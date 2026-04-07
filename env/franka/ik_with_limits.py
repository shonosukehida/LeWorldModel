# pldm_envs/franka/ik_with_limits.py

import collections
from absl import logging
from dm_control.mujoco.wrapper import mjbindings
import numpy as np

mjlib = mjbindings.mjlib

IKResult = collections.namedtuple(
    'IKResult', ['qpos', 'err_norm', 'steps', 'success']
)

_INVALID_JOINT_NAMES_TYPE = (
    '`joint_names` must be either None, a list, a tuple, or a numpy array; '
    'got {}.'
)
_REQUIRE_TARGET_POS_OR_QUAT = (
    'At least one of `target_pos` or `target_quat` must be specified.'
)


def qpos_from_site_pose(
    physics,
    site_name,
    target_pos=None,
    target_quat=None,
    joint_names=None,
    tol=1e-14,
    rot_weight=1.0,
    regularization_threshold=0.1,
    regularization_strength=3e-2,
    max_update_norm=2.0,
    progress_thresh=20.0,
    max_steps=100,
    inplace=False,
    limit_margin=1e-4,  # joint 範囲端から少しだけ内側に寄せるマージン
):
    """Joint 制約を毎ステップ投影しながら IK を解くバージョン"""

    dtype = physics.data.qpos.dtype

    if target_pos is not None and target_quat is not None:
        jac = np.empty((6, physics.model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, physics.model.nv), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if target_pos is not None:
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif target_quat is not None:
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError(_REQUIRE_TARGET_POS_OR_QUAT)

    update_nv = np.zeros(physics.model.nv, dtype=dtype)

    if target_quat is not None:
        site_xquat = np.empty(4, dtype=dtype)
        neg_site_xquat = np.empty(4, dtype=dtype)
        err_rot_quat = np.empty(4, dtype=dtype)

    if not inplace:
        physics = physics.copy(share_model=True)

    model = physics.model
    data = physics.data

    # Ensure that the Cartesian position of the site is up to date.
    mjlib.mj_fwdPosition(model.ptr, data.ptr)

    # Convert site name to index.
    site_id = model.name2id(site_name, 'site')

    # These are views onto the underlying MuJoCo buffers.
    site_xpos = physics.named.data.site_xpos[site_name]
    site_xmat = physics.named.data.site_xmat[site_name]

    # DOF indices for allowed joints
    if joint_names is None:
        dof_indices = slice(None)  # 全 DOF

        joint_ids_for_limits = list(range(model.njnt))
    elif isinstance(joint_names, (list, np.ndarray, tuple)):
        if isinstance(joint_names, tuple):
            joint_names = list(joint_names)
        indexer = physics.named.model.dof_jntid.axes.row
        dof_indices = indexer.convert_key_item(joint_names)

        joint_ids_for_limits = [
            model.name2id(jn, 'joint') for jn in joint_names
        ]
    else:
        raise ValueError(_INVALID_JOINT_NAMES_TYPE.format(type(joint_names)))

    steps = 0
    success = False


    for steps in range(max_steps):

        err_norm = 0.0

        if target_pos is not None:
            # Translational error.
            err_pos[:] = target_pos - site_xpos
            err_norm += np.linalg.norm(err_pos)
        if target_quat is not None:
            # Rotational error.
            mjlib.mju_mat2Quat(site_xquat, site_xmat)
            mjlib.mju_negQuat(neg_site_xquat, site_xquat)
            mjlib.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
            mjlib.mju_quat2Vel(err_rot, err_rot_quat, 1)
            err_norm += np.linalg.norm(err_rot) * rot_weight

        if err_norm < tol:
            logging.debug('Converged after %i steps: err_norm=%3g', steps, err_norm)
            success = True
            break
        else:
            # ヤコビアン計算
            mjlib.mj_jacSite(
                model.ptr, data.ptr, jac_pos, jac_rot, site_id
            )
            jac_joints = jac[:, dof_indices]

            reg_strength = (
                regularization_strength if err_norm > regularization_threshold
                else 0.0
            )
            update_joints = nullspace_method(
                jac_joints, err, regularization_strength=reg_strength
            )

            update_norm = np.linalg.norm(update_joints)

            # 進捗が悪すぎれば打ち切り
            progress_criterion = err_norm / update_norm
            if progress_criterion > progress_thresh:
                logging.debug(
                    'Step %2i: err_norm / update_norm (%3g) > '
                    'tolerance (%3g). Halting due to insufficient progress',
                    steps, progress_criterion, progress_thresh)
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            # DOF ベクトルに書き戻し
            update_nv[dof_indices] = update_joints

            # ここからjoint limit を毎ステップ適用する
            mjlib.mj_integratePos(model.ptr, data.qpos, update_nv, 1)

            # joint 範囲に投影
            qpos = data.qpos  
            for jid in joint_ids_for_limits:
                if model.jnt_limited[jid]:
                    adr = model.jnt_qposadr[jid]
                    lo, hi = model.jnt_range[jid]
                    if lo < hi:
                        lo_eff = lo + limit_margin
                        hi_eff = hi - limit_margin
                        if lo_eff > hi_eff:  
                            lo_eff, hi_eff = lo, hi
                        qpos[adr] = np.clip(qpos[adr], lo_eff, hi_eff)


            mjlib.mj_fwdPosition(model.ptr, data.ptr)

            logging.debug(
                'Step %2i: err_norm=%-10.3g update_norm=%-10.3g',
                steps, err_norm, update_norm)

    if not success and steps == max_steps - 1:
        logging.warning('Failed to converge after %i steps: err_norm=%3g',
                        steps, err_norm)

    if not inplace:
        qpos = data.qpos.copy()
    else:
        qpos = data.qpos

    return IKResult(qpos=qpos, err_norm=err_norm, steps=steps, success=success)


def nullspace_method(jac_joints, delta, regularization_strength=0.0):
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if regularization_strength > 0:
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
