from robopy.config.sensor_config.visual_config.camera_config import (
    RealsenseCameraConfig,
)
from robopy.sensors.visual.realsense_camera import RealsenseCamera


def test_by_serial(serial_no: str) -> None:
    print("=== Test: serial number ===")

    config = RealsenseCameraConfig(
        name="serial_test",
        width=640,
        height=480,
        fps=30,
        serial_no=serial_no,
    )

    camera = RealsenseCamera(config)

    try:
        camera.connect()
        print(f"Connected by serial: {camera.serial_number}")

        frame = camera.read()
        print(f"Frame shape: {frame.shape}")

    finally:
        camera.disconnect()


def test_by_index(index: int) -> None:
    print("=== Test: index ===")

    config = RealsenseCameraConfig(
        name="index_test",
        width=640,
        height=480,
        fps=30,
        index=index,
    )

    camera = RealsenseCamera(config)

    try:
        camera.connect()
        print(f"Connected by index {index}")
        print(f"Resolved serial: {camera.serial_number}")

        frame = camera.read()
        print(f"Frame shape: {frame.shape}")

    finally:
        camera.disconnect()


if __name__ == "__main__":
    SERIAL_NO = "138422071505"
    INDEX = 0

    test_by_serial(SERIAL_NO)
    test_by_index(INDEX)
