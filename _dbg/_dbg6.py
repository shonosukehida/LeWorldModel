from xarm.wrapper import XArmAPI

IP = "192.168.1.240"

arm = XArmAPI(IP, is_radian=True)

print("connected:", arm.connected)
print("TCP offset:", arm.tcp_offset)

code, pose = arm.get_position(is_radian=True)
print("get_position code:", code)
print("TCP pose:", pose)

code, pose_aa = arm.get_position_aa(is_radian=True)
print("TCP pose (axis-angle):", pose_aa)

arm.disconnect()