from xarm.wrapper import XArmAPI

arm = XArmAPI("192.168.1.240", is_radian=True)

code, pose = arm.get_forward_kinematics(
    joints.tolist(),
    input_is_radian=True,
)