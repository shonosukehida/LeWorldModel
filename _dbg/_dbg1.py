
import time

import numpy as np
from xarm.wrapper import XArmAPI


# ============================================================
# Configuration
# ============================================================

ROBOT_IP = "192.168.1.240"
READ_HZ = 10


def main():

    # --------------------------------------------------------
    # Connect to xArm
    # --------------------------------------------------------

    arm = XArmAPI(ROBOT_IP, is_radian=True)

    sensor_enabled = False

    try:

        # ----------------------------------------------------
        # Enable F/T Sensor
        # ----------------------------------------------------

        code = arm.set_ft_sensor_enable(1)

        if code != 0:
            raise RuntimeError(
                f"Failed to enable F/T sensor: code={code}"
            )

        sensor_enabled = True

        time.sleep(0.5)

        print("F/T Sensor enabled.")
        print("Press Ctrl+C to stop.\n")

        # ----------------------------------------------------
        # Read F/T Sensor
        # ----------------------------------------------------

        while True:

            code, ft_data = arm.get_ft_sensor_data()

            if code != 0:
                raise RuntimeError(
                    f"Failed to read F/T sensor: code={code}"
                )

            ft = np.asarray(ft_data, dtype=np.float32)

            fx, fy, fz, tx, ty, tz = ft

            # Force magnitude
            force = np.linalg.norm(ft[:3])

            # Torque magnitude
            torque = np.linalg.norm(ft[3:])

            print(
                f"Fx={fx:8.3f} N | "
                f"Fy={fy:8.3f} N | "
                f"Fz={fz:8.3f} N | "
                f"Tx={tx:8.3f} Nm | "
                f"Ty={ty:8.3f} Nm | "
                f"Tz={tz:8.3f} Nm | "
                f"Force={force:8.3f} N | "
                f"Torque={torque:8.3f} Nm"
            )

            time.sleep(1.0 / READ_HZ)

    except KeyboardInterrupt:
        print("\nF/T Sensor test stopped.")

    finally:

        if sensor_enabled:
            arm.set_ft_sensor_enable(0)

        arm.disconnect()

        print("Disconnected.")


if __name__ == "__main__":
    main()