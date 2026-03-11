#!/bin/bash
set -e

# 强制软件渲染（Docker 无 GPU）
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

# 启动虚拟显示器（Gazebo 需要 OpenGL context）
Xvfb :99 -screen 0 1024x768x24 +extension GLX &
XVFB_PID=$!
export DISPLAY=:99
sleep 2

# 检查 Xvfb 是否存活
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb 启动失败，无法创建虚拟显示器"
    exit 1
fi

# Source ROS 2
source /opt/ros/humble/setup.bash

# 后台启动 Gazebo（server only，无 GUI）+ TurtleBot3
echo "启动 Gazebo + TurtleBot3 Burger (headless) ..."
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py use_sim_time:=true &
SIM_PID=$!

# 等待 /odom topic 就绪（最多 120 秒，Rosetta 模拟较慢）
echo "等待仿真就绪（首次启动可能需要 1-2 分钟）..."
SIM_READY=false
for i in $(seq 1 120); do
    if ros2 topic list 2>/dev/null | grep -q "/odom"; then
        sleep 5
        SIM_READY=true
        echo "仿真就绪"
        break
    fi
    sleep 1
done

if [ "$SIM_READY" = false ]; then
    echo "ERROR: 仿真 120 秒内未就绪（/odom topic 未出现）"
    echo "可能原因：Gazebo 物理引擎在 Rosetta/软件渲染下启动较慢"
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
