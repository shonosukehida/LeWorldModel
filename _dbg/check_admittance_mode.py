
import time
from xarm.wrapper import XArmAPI

ROBOT_IP = "192.168.1.240"

arm = XArmAPI(ROBOT_IP)


def check(code, name):
    if code != 0:
        raise RuntimeError(f"{name} failed: code={code}")


try:
    # 1. 通常の位置制御モード
    check(arm.set_mode(0), "set_mode")
    check(arm.set_state(0), "set_state")

    # 2. インピーダンスパラメータ
    M = 0.06
    J = M * 0.01

    K_pos = 300
    K_ori = 4

    mass = [M, M, M, J, J, J]

    stiffness = [
        K_pos, K_pos, K_pos,
        K_ori, K_ori, K_ori
    ]

    damping = [0] * 6

    check(
        arm.set_ft_sensor_admittance_parameters(
            mass,
            stiffness,
            damping
        ),
        "set_admittance_parameters"
    )

    # 3. Z方向のみ柔軟にする
    ref_frame = 0
    c_axis = [0, 0, 1, 0, 0, 0]

    check(
        arm.set_ft_sensor_admittance_parameters(
            ref_frame,
            c_axis
        ),
        "set_admittance_axes"
    )

    # 4. F/Tセンサ有効化
    check(
        arm.set_ft_sensor_enable(1),
        "set_ft_sensor_enable"
    )

    time.sleep(0.2)

    # 5. アドミッタンス制御開始
    check(
        arm.set_ft_sensor_mode(1),
        "set_ft_sensor_mode"
    )

    check(arm.set_state(0), "set_state")

    print("Admittance control enabled")

    # 6. 10秒間動作
    time.sleep(30)

finally:
    # 7. 通常モードに戻す
    arm.set_ft_sensor_mode(0)
    arm.set_ft_sensor_enable(0)
    arm.disconnect()