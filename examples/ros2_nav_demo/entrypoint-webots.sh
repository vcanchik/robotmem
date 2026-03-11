#!/bin/bash
set -e

# 启动虚拟显示器（Webots 需要 OpenGL context）
Xvfb :99 -screen 0 1024x768x16 &
XVFB_PID=$!
export DISPLAY=:99
sleep 1

# 检查 Xvfb 是否存活
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb 启动失败，无法创建虚拟显示器"
    exit 1
fi

# Source ROS 2
source /opt/ros/humble/setup.bash

# 后台启动 Webots + TurtleBot3
echo "启动 Webots + TurtleBot3 Burger ..."
ros2 launch webots_ros2_turtlebot robot_launch.py &
SIM_PID=$!

# 等待 /odom topic 就绪（最多 60 秒）
echo "等待仿真就绪 ..."
SIM_READY=false
for i in $(seq 1 60); do
    if ros2 topic list 2>/dev/null | grep -q "/odom"; then
        sleep 3
        SIM_READY=true
        echo "仿真就绪"
        break
    fi
    sleep 1
done

if [ "$SIM_READY" = false ]; then
    echo "ERROR: 仿真 60 秒内未就绪（/odom topic 未出现）"
    kill $SIM_PID 2>/dev/null || true
    kill $XVFB_PID 2>/dev/null || true
    exit 1
fi

# 运行 Demo（透传所有参数，无参数时 demo.py 使用自身默认值）
echo ""
python3 /demo.py "$@"

# 清理
kill $SIM_PID 2>/dev/null || true
kill $XVFB_PID 2>/dev/null || true
