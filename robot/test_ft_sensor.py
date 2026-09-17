import time

from xarm.wrapper import XArmAPI


ROBOT_IP = "192.168.1.240"


def check_code(code: int, name: str) -> None:
    if code != 0:
        raise RuntimeError(f"{name} failed with code={code}")


def main() -> None:
    arm = XArmAPI(ROBOT_IP, enable_report=True)

    try:
        print("Connecting to xArm...")
        print(f"SDK version: {arm.version}")

        # Robot / sensor initialization
        check_code(arm.motion_enable(enable=True), "motion_enable")

        # 一度無効化してから有効化
        check_code(arm.set_ft_sensor_enable(0), "disable FT sensor")

        check_code(arm.clean_error(), "clean_error")
        check_code(arm.clean_warn(), "clean_warn")

        check_code(arm.set_ft_sensor_enable(1), "enable FT sensor")

        time.sleep(0.5)

        print("\nFT sensor enabled.")

        # 現在のセンサ設定確認
        code, config = arm.get_ft_sensor_config()
        check_code(code, "get_ft_sensor_config")

        print("\nFT sensor config:")
        print(f"  mode:       {config[0]}")
        print(f"  started:    {config[1]}")
        print(f"  type:       {config[2]}")
        print(f"  id:         {config[3]}")
        print(f"  frequency:  {config[4]}")

        print("\nReading force/torque data...")
        print("Lightly push/pull the gripper by hand and observe the values.")
        print("Press Ctrl+C to stop.\n")

        while True:
            code, ft = arm.get_ft_sensor_data()
            check_code(code, "get_ft_sensor_data")

            fx, fy, fz, tx, ty, tz = ft

            print(
                f"Fx={fx:8.3f} N, "
                f"Fy={fy:8.3f} N, "
                f"Fz={fz:8.3f} N | "
                f"Tx={tx:8.3f} Nm, "
                f"Ty={ty:8.3f} Nm, "
                f"Tz={tz:8.3f} Nm"
            )

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nStopping FT sensor test...")

    finally:
        try:
            arm.set_ft_sensor_enable(0)
        except Exception:
            pass

        arm.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()