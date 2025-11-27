#!/bin/bash
set -e

# ROS2
source /opt/ros/jazzy/setup.bash

# uv venv
source /ros2_voicevox_ws/.venv/bin/activate

# colcon
source /ros2_voicevox_ws/install/setup.bash

exec "$@"
