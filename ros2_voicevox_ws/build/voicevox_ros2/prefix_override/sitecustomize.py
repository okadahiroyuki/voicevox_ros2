import sys
if sys.prefix == '/home/roboworks/ros2_voicevox_ws/.venv':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/roboworks/ros2_voicevox_ws/install/voicevox_ros2'
