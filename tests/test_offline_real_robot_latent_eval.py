"""Hardware-free saved-rollout diagnostics regression tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from offline_real_robot_latent_eval import evaluate_rollout, load_wm_process, fingerprint, validate_recorded_config
from offline_latent_dist_eval import _l2


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.bad = False

    def encode(self, info):
        value = torch.as_tensor(info['proprio'][..., :2])
        return {'emb': value[:, 0] if self.bad else value}


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'rollout.h5'
        with h5py.File(self.source, 'w') as file:
            for key in ('pixels', 'wrist_pixels'):
                file[key] = np.zeros((3, 4, 4, 3), dtype=np.uint8)
            for key in ('goal', 'goal_wrist_pixels'):
                file[key] = np.zeros((4, 4, 3), dtype=np.uint8)
            file['proprio'] = np.array([[3, 4, 0, 0, 0, 0, 0, 0], [0]*8, [6, 8, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
            file['goal_proprio'] = np.zeros(8, dtype=np.float32)
        self.model = Model().eval()
        self.policy = Mock(world_model=self.model)
        self.policy.encode_goal = Mock(return_value=torch.zeros(1, 1, 2))
        self.policy._prepare_info = lambda info: info

    def test_l2(self):
        np.testing.assert_array_equal(_l2(torch.tensor([[3., 4.]]), torch.zeros(1, 2)), [5.])

    def test_goal_once_count_reproducibility_readonly_rng(self):
        before = fingerprint(self.source)
        rng = torch.random.get_rng_state().clone()
        out = evaluate_rollout(self.policy, self.source)
        self.policy.encode_goal.assert_called_once()
        with h5py.File(out / 'latent_distance.h5') as file:
            first = file['observed_latent_distance'][:]
            np.testing.assert_array_equal(first, [5, 0, 10])
            self.assertEqual(file['observed_latents'].shape, (3, 2))
            self.assertEqual(len(file['step_idx']), 3)
        second = evaluate_rollout(self.policy, self.source, self.root / 'repeat')
        with h5py.File(second / 'latent_distance.h5') as file:
            np.testing.assert_array_equal(first, file['observed_latent_distance'][:])
        self.assertEqual(before, fingerprint(self.source))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue((out / 'observed_latent_distance.png').is_file())
        self.assertEqual(evaluate_rollout(self.policy, self.source), out)

    def test_repeat_replaces_all_evaluation_outputs_and_preserves_sources(self):
        out = evaluate_rollout(self.policy, self.source, metadata={'run': 'old'})
        with h5py.File(out / 'latent_distance.h5', 'a') as file:
            file['obsolete_dataset'] = [1]
        (out / 'observed_latent_distance.png').write_bytes(b'old graph')
        originals = [self.source]
        for parent in (self.root, out):
            for name in ('rollout.mp4', 'robot.log'):
                path = parent / name
                path.write_bytes(b'original robot data')
                originals.append(path)
        before = {path: fingerprint(path) for path in originals}
        self.assertEqual(evaluate_rollout(self.policy, self.source, metadata={'run': 'new'}, max_steps=2), out)
        with h5py.File(out / 'latent_distance.h5') as file:
            self.assertNotIn('obsolete_dataset', file)
            np.testing.assert_array_equal(file['step_idx'][:], [0, 1])
            np.testing.assert_array_equal(file['observed_latent_distance'][:], [5, 0])
            self.assertEqual(json.loads(file.attrs['metadata'])['run'], 'new')
        metrics = json.loads((out / 'latent_distance_metrics.json').read_text())
        self.assertEqual(metrics['run'], 'new')
        self.assertEqual(metrics['evaluation_steps'], 2)
        self.assertTrue((out / 'observed_latent_distance.png').read_bytes().startswith(b'\x89PNG'))
        self.assertEqual(before, {path: fingerprint(path) for path in originals})
        self.assertFalse(list(out.glob('.latent_eval_*')))

    def test_output_symlink_does_not_modify_source(self):
        out = self.root / 'latent_eval'
        out.mkdir()
        (out / 'latent_distance.h5').symlink_to(self.source)
        before = fingerprint(self.source)
        evaluate_rollout(self.policy, self.source)
        self.assertFalse((out / 'latent_distance.h5').is_symlink())
        self.assertEqual(before, fingerprint(self.source))

    def test_real_gpc_preprocessing_and_goal_api(self):
        from types import SimpleNamespace
        from stable_worldmodel.policy import GPCPolicy
        from utils import get_eval_img_preprocessor
        from action_normalization import SafeStandardScaler
        from unittest.mock import patch
        scaler = SafeStandardScaler()
        scaler.mean_ = np.zeros((1, 8), dtype=np.float32)
        scaler.scale_ = np.ones((1, 8), dtype=np.float32)
        transform = get_eval_img_preprocessor(8)
        policy = GPCPolicy(SimpleNamespace(action_horizon=1), self.model, None,
                           process={'proprio': scaler, 'goal_proprio': scaler},
                           transform={key: transform for key in ('pixels', 'wrist_pixels', 'goal', 'goal_wrist_pixels')})
        with patch.object(policy, 'encode_goal', wraps=policy.encode_goal) as encode:
            out = evaluate_rollout(policy, self.source)
            encode.assert_called_once()
        with h5py.File(out / 'latent_distance.h5') as file:
            np.testing.assert_array_equal(file['observed_latent_distance'][:], [5, 0, 10])

    def test_saved_camera_and_goal_resolutions_are_independent(self):
        # RealSense frames and cv2-loaded goal files are persisted without resize.
        # Use the actual GPC preprocessing: both become encoder-sized images.
        with h5py.File(self.source, 'a') as file:
            for key, shape in [('pixels', (3, 48, 64, 3)),
                               ('wrist_pixels', (3, 60, 80, 3)),
                               ('goal', (24, 32, 3)),
                               ('goal_wrist_pixels', (40, 40, 3))]:
                del file[key]
                file[key] = np.zeros(shape, dtype=np.uint8)
        self.test_real_gpc_preprocessing_and_goal_api()

    def test_invalid_image_shapes_are_rejected(self):
        from offline_real_robot_latent_eval import validate_rollout
        for key, shape in [('goal', (1, 4, 4, 3)), ('goal', (3, 4, 4)),
                           ('goal_wrist_pixels', (4, 4, 1)), ('pixels', (2, 4, 4, 3)),
                           ('wrist_pixels', (3, 3, 4, 4))]:
            with self.subTest(key=key, shape=shape):
                with h5py.File(self.source, 'a') as file:
                    original = file[key][:]
                    del file[key]
                    file[key] = np.zeros(shape, dtype=np.uint8)
                    with self.assertRaisesRegex(ValueError, key):
                        validate_rollout(file)
                    del file[key]
                    file[key] = original

    def test_recorded_configuration_mismatch(self):
        from omegaconf import OmegaConf
        stats = self.root / 'stats.npz'
        checkpoint = self.root / 'model_object.ckpt'
        config = dict(policy=str(self.root / 'model'),
                      eval=dict(img_size=8, normalization_stats_path=str(stats)))
        with h5py.File(self.source, 'a') as file:
            file.attrs['config'] = OmegaConf.to_yaml(config)
        validate_recorded_config(self.source, checkpoint, stats, 8)
        with self.assertRaisesRegex(ValueError, 'image-size'):
            validate_recorded_config(self.source, checkpoint, stats, 16)
        with self.assertRaisesRegex(ValueError, 'statistics differ'):
            validate_recorded_config(self.source, checkpoint, self.root / 'dp.npz', 8)
        with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
            validate_recorded_config(self.source, self.root / 'other_object.ckpt', stats, 8)

    def test_bad_shape(self):
        self.model.bad = True
        with self.assertRaisesRegex(ValueError, 'no pooling or flattening'):
            evaluate_rollout(self.policy, self.source)

    def test_missing_stats(self):
        with self.assertRaisesRegex(FileNotFoundError, 'statistics'):
            load_wm_process(self.root / 'missing.npz')

    def test_training_mode(self):
        self.model.train()
        with self.assertRaisesRegex(ValueError, 'eval'):
            evaluate_rollout(self.policy, self.source)

    def test_missing_input_and_dimension(self):
        self.policy.encode_goal.return_value = torch.zeros(1, 1, 3)
        with self.assertRaisesRegex(ValueError, 'match exactly'):
            evaluate_rollout(self.policy, self.source)
        with h5py.File(self.source, 'a') as file:
            del file['goal_proprio']
        with self.assertRaisesRegex(ValueError, 'goal_proprio'):
            evaluate_rollout(self.policy, self.source)


if __name__ == '__main__':
    unittest.main()
