from xarm.wrapper import XArmAPI

ROBOT_IP = "192.168.1.240"

arm = XArmAPI(ROBOT_IP, is_radian=True)

try:
    # ロボット本体のエラー
    code, err_warn = arm.get_err_warn_code()
    print("Robot error/warn:", code, err_warn)
    print("arm.error_code:", arm.error_code)

    # F/T Sensorのエラー
    code, ft_error = arm.get_ft_sensor_error()
    print("FT sensor error:", code, ft_error)

    # F/T Sensorの設定
    code, config = arm.get_ft_sensor_config()

    print("FT config code:", code)

    if code == 0:
        print("FT mode:", config[0])
        print("FT enabled:", config[1])
        print("FT type:", config[2])
        print("FT ID:", config[3])
        print("FT frequency:", config[4])
        print("FT mass:", config[5])

    # 読み出し結果とAPI戻り値を確認
    code, ft_data = arm.get_ft_sensor_data()

    print("FT read code:", code)
    print("FT data:", ft_data)

finally:
    arm.disconnect()