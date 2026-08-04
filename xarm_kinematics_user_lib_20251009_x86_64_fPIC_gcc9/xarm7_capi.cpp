// C wrapper exposing simple symbols for ctypes binding from Python.
// Statically links libkinematics_xarm_user.a so the resulting .so has
// no glibc-2.34 dependency (the .so shipped in this dir does — it was
// built on a newer host than the target system here, Ubuntu 20.04 /
// glibc 2.31).
//
// Build (see Makefile.python):
//   g++ -std=c++11 -fPIC -shared xarm7_capi.cpp \
//       libkinematics_xarm_user.a -o libxarm7_capi.so -fopenmp

#include "xarm_kinematics_interface.h"
#include <cstring>
#include <cmath>

extern "C" {

// Default joint range macros expand to {a,b,c,...}; we can't take the
// address of a brace-init, so copy into a stack array per call.
static double g_qmax[7] = XARM7_ANGLE_MAX;
static double g_qmin[7] = XARM7_ANGLE_MIN;

// Initialize the kinematic engine.
//   q_max, q_min  : 7-vec each. Pass NULL for the default
//                   XARM7_ANGLE_MAX / _MIN.
//   tcp_offset    : 6-vec [x_mm, y_mm, z_mm, roll, pitch, yaw] of
//                   the user TCP frame relative to the flange.
//                   Pass NULL for "no offset".
//   world_offset  : 6-vec base->world frame transform. Pass NULL for
//                   no transform.
// Returns 0 on success.
int xarm7_init(double *q_max, double *q_min,
               double *tcp_offset, double *world_offset) {
    double qmax_use[7];
    double qmin_use[7];
    if (q_max != NULL) memcpy(qmax_use, q_max, sizeof(qmax_use));
    else                memcpy(qmax_use, g_qmax, sizeof(qmax_use));
    if (q_min != NULL) memcpy(qmin_use, q_min, sizeof(qmin_use));
    else                memcpy(qmin_use, g_qmin, sizeof(qmin_use));
    return xarm7_config(qmax_use, qmin_use, tcp_offset, world_offset);
}

// theta  (in,  7-vec, rad)
// pose   (out, 6-vec, [x_mm, y_mm, z_mm, roll, pitch, yaw])
int xarm7_fk(double *theta, double *pose) {
    return xarm7_forward_kinematics(theta, pose);
}

// pose   (in,  6-vec, [x_mm, y_mm, z_mm, roll, pitch, yaw])
// q_pre  (in,  7-vec, rad) — seed / reference joints
// theta  (out, 7-vec, rad)
int xarm7_ik(double *pose, double *q_pre, double *theta) {
    return xarm7_inverse_kinematics(pose, q_pre, theta);
}

}  // extern "C"
