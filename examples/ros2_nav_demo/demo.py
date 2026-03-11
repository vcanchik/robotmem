#!/usr/bin/env python3
"""robotmem ROS 2 Demo — 记忆驱动导航对比

Session 1: 反应式避障探索（前进 + 撞墙后退转向），robotmem 记住航点
Session 2: recall 航点 → 反转 → 原路返航

证明：走过的路不用再走第二次 — 记忆驱动直接返航。

环境要求：
- ROS 2 Humble
- 有 cmd_vel (Twist) + odom (Odometry) 的移动机器人
- pip install git+https://github.com/robotmem/robotmem.git

已测试平台：
- Docker Gazebo headless — TurtleBot3 Burger (turtlebot3_world)
- Docker Webots headless — TurtleBot3 Burger (webots_ros2_turtlebot)
- The Construct (app.theconstruct.ai) — Originbot Racing Circuit

运行方式：
  # 方式 1: Docker（推荐）
  cd examples/ros2_nav_demo
  docker compose run gazebo-demo    # Gazebo 仿真
  docker compose run webots-demo    # Webots 仿真

  # 方式 2: 直接运行（需 ROS 2 环境）
  pip3 install git+https://github.com/robotmem/robotmem.git
  source /opt/ros/humble/setup.bash
  python3 demo.py                   # TurtleBot3 标准 topic
  python3 demo.py 40                # 40 秒探索
  python3 demo.py 25 --prefix /originbot_1  # The Construct Originbot

已知限制：
- Session 2 导航为航点跟踪 + 简单避障（后退转向），无全局路径规划
- 狭窄赛道环境中，反转航点导航可能因墙壁阻挡而跳过部分航点
- 建议在开阔环境或宽赛道中运行以获得最佳效果
- 运行前建议重置仿真（Gazebo reload 或 ros2 service call /reset_simulation）

robotmem API 使用：
- learn(): 记录航点路线 + spatial context
- save_perception(): 记录 odometry 轨迹数据
- recall(): 检索航点用于返航导航
- session(): 区分探索/返航两个 episode
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math, time, json, os, sys, argparse, random

from robotmem.sdk import RobotMemory

# Demo 用独立 DB，不污染主 ~/.robotmem/memory.db
DB_DIR = os.path.expanduser("~/.robotmem-ros-demo")
DB_PATH = os.path.join(DB_DIR, "memory.db")
COLLECTION = "ros_nav_demo"


class ExploreNode(Node):
    """Session 1: 前进 + 撞墙后退转向，记录航点"""

    FORWARD = 0
    BACKUP = 1
    TURN = 2

    def __init__(self, duration=25.0, cmd_topic="/cmd_vel",
                 odom_topic="/odom"):
        super().__init__("explore_node")
        self.pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_timer(0.1, self._loop)

        self.duration = duration
        self.t0 = time.time()
        self.explore_t0 = None  # 首次收到 odom 时设置
        self.positions = []
        self.waypoints = []
        self.pos = None
        self.prev_pos = None
        self.stuck_count = 0
        self.last_wp_t = 0
        self.done = False
        self.state = self.FORWARD
        self.state_t = time.time()
        self.turn_dir = 1.0

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self.pos = (p.x, p.y)
        if self.explore_t0 is None:
            self.explore_t0 = time.time()
            self.get_logger().info("收到 odom，开始探索")
        self.positions.append((p.x, p.y, time.time() - self.explore_t0))

        now = time.time()
        if now - self.last_wp_t > 2.0:
            self.waypoints.append((p.x, p.y))
            self.last_wp_t = now

    def _loop(self):
        if self.done:
            return
        # 从首次收到 odom 开始计时
        if self.explore_t0 is None:
            # 安全超时：120s 内没收到 odom 就退出
            if time.time() - self.t0 > 120:
                self.done = True
                self.get_logger().error("120s 内未收到 odom，退出")
            return
        if time.time() - self.explore_t0 > self.duration:
            self.pub.publish(Twist())
            self.done = True
            self.get_logger().info(
                f"探索完成: {len(self.waypoints)} 航点, "
                f"{len(self.positions)} 位置, "
                f"{time.time()-self.explore_t0:.1f}s"
            )
            return

        cmd = Twist()
        now = time.time()

        if self.state == self.FORWARD:
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
            # 位置变化检测卡住（比速度更可靠）
            if self.prev_pos and self.pos:
                d = math.hypot(self.pos[0] - self.prev_pos[0],
                               self.pos[1] - self.prev_pos[1])
                if d < 0.01:
                    self.stuck_count += 1
                else:
                    self.stuck_count = 0
            if self.stuck_count > 8:
                self.state = self.BACKUP
                self.state_t = now
                self.turn_dir = random.choice([-1.0, 1.0])
                self.stuck_count = 0

        elif self.state == self.BACKUP:
            cmd.linear.x = -0.2
            cmd.angular.z = 0.0
            if now - self.state_t > 1.5:
                self.state = self.TURN
                self.state_t = now

        elif self.state == self.TURN:
            cmd.linear.x = 0.0
            cmd.angular.z = self.turn_dir * 1.2
            if now - self.state_t > random.uniform(1.0, 2.5):
                self.state = self.FORWARD
                self.state_t = now

        self.prev_pos = self.pos
        self.pub.publish(cmd)


class NavNode(Node):
    """Session 2: 记忆导航（含避障：卡住后退转向再继续）"""

    NAVIGATE = 0
    BACKUP = 1
    TURN = 2

    def __init__(self, waypoints, cmd_topic="/cmd_vel",
                 odom_topic="/odom", timeout=120.0):
        super().__init__("nav_node")
        self.pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_timer(0.1, self._loop)

        self.wps = waypoints
        self.idx = 0
        self.pos = None
        self.prev_pos = None
        self.yaw = 0.0
        self.t0 = time.time()
        self.nav_t0 = None  # 导航开始时间（首次收到 odom 时设置）
        self.timeout = timeout
        self.positions = []
        self.done = False
        self.skipped = 0
        self.state = self.NAVIGATE
        self.state_t = time.time()
        self.stuck_count = 0
        self.wp_retries = 0
        self.turn_dir = 1.0

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self.pos = (p.x, p.y)
        if self.nav_t0 is None:
            self.nav_t0 = time.time()
            self.get_logger().info("收到 odom，开始导航")
        self.positions.append((p.x, p.y, time.time() - (self.nav_t0 or self.t0)))
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _finish(self, reason="done"):
        self.pub.publish(Twist())
        self.done = True
        t = time.time() - (self.nav_t0 or self.t0)
        self.get_logger().info(
            f"导航{reason}: {t:.1f}s, {self.idx}/{len(self.wps)} 航点, "
            f"跳过 {self.skipped}")

    def _next_wp(self):
        self.idx += 1
        self.stuck_count = 0
        self.wp_retries = 0
        self.state = self.NAVIGATE

    def _loop(self):
        if self.done:
            return
        # 安全超时：即使没收到 odom 也不能无限等待
        if time.time() - self.t0 > self.timeout + 120:
            self._finish("odom 超时")
            return
        if self.pos is None:
            return
        # 导航超时：从首次收到 odom 开始计算
        if self.nav_t0 and time.time() - self.nav_t0 > self.timeout:
            self._finish("超时")
            return
        if self.idx >= len(self.wps):
            self._finish("完成")
            return

        now = time.time()
        cmd = Twist()

        if self.state == self.NAVIGATE:
            tx, ty = self.wps[self.idx]
            cx, cy = self.pos
            dist = math.hypot(tx - cx, ty - cy)

            if dist < 0.8:
                self._next_wp()
                return

            # 位置变化检测卡住
            if self.prev_pos:
                d = math.hypot(self.pos[0] - self.prev_pos[0],
                               self.pos[1] - self.prev_pos[1])
                if d < 0.01:
                    self.stuck_count += 1
                else:
                    self.stuck_count = 0

            if self.stuck_count > 8:
                self.wp_retries += 1
                if self.wp_retries > 3:
                    self.skipped += 1
                    self._next_wp()
                    return
                self.state = self.BACKUP
                self.state_t = now
                self.turn_dir = random.choice([-1.0, 1.0])
                self.stuck_count = 0
            else:
                # 朝航点导航
                target_yaw = math.atan2(ty - cy, tx - cx)
                err = target_yaw - self.yaw
                while err > math.pi:
                    err -= 2 * math.pi
                while err < -math.pi:
                    err += 2 * math.pi
                if abs(err) > 0.3:
                    cmd.angular.z = max(-1.5, min(1.5, 1.5 * err))
                    cmd.linear.x = 0.1
                else:
                    cmd.linear.x = min(0.4, dist)
                    cmd.angular.z = 0.6 * err

        elif self.state == self.BACKUP:
            cmd.linear.x = -0.2
            if now - self.state_t > 1.0:
                self.state = self.TURN
                self.state_t = now

        elif self.state == self.TURN:
            cmd.angular.z = self.turn_dir * 1.0
            if now - self.state_t > random.uniform(0.5, 1.5):
                self.state = self.NAVIGATE
                self.state_t = now

        self.prev_pos = self.pos
        self.pub.publish(cmd)


def path_len(pts):
    """轨迹总距离"""
    return sum(
        math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        for i in range(1, len(pts))
    )


def main():
    parser = argparse.ArgumentParser(description="robotmem ROS 2 导航记忆 Demo")
    parser.add_argument("duration", nargs="?", type=float, default=25.0,
                        help="探索时长（秒），默认 25")
    parser.add_argument("--prefix", default="",
                        help="Topic 前缀，如 /originbot_1（The Construct）或空（TurtleBot3 标准）")
    args = parser.parse_args()

    # prefix 校验：非空时必须以 / 开头
    prefix = args.prefix
    if prefix and not prefix.startswith("/"):
        prefix = f"/{prefix}"

    # duration 校验
    if args.duration <= 0:
        print(f"ERROR: 探索时长必须 > 0，当前值: {args.duration}")
        return
    if args.duration < 3:
        print(f"WARNING: 探索时长 {args.duration}s 太短，可能采集不到航点（建议 >= 10s）")

    cmd_topic = f"{prefix}/cmd_vel"
    odom_topic = f"{prefix}/odom"

    # 清理旧数据（只删 DB 文件，不删整个目录）
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    os.makedirs(DB_DIR, exist_ok=True)

    rclpy.init()

    print("=" * 55)
    print("robotmem ROS 2 Demo — 记忆驱动导航")
    print(f"Topics: {cmd_topic}, {odom_topic}")
    print("=" * 55)

    # ── Session 1: 探索 ──
    print(f"\n--- Session 1: 探索 ({args.duration}s) ---")
    exp = ExploreNode(duration=args.duration, cmd_topic=cmd_topic,
                      odom_topic=odom_topic)
    while rclpy.ok() and not exp.done:
        rclpy.spin_once(exp, timeout_sec=0.1)

    wps = exp.waypoints
    e_pos = exp.positions
    e_time = e_pos[-1][2] if e_pos else 0
    e_dist = path_len(e_pos)
    print(f"  航点: {len(wps)}, 距离: {e_dist:.2f}m, 用时: {e_time:.1f}s")

    if not wps:
        print("  ERROR: 探索未采集到航点（可能 odom 未发布或探索时间太短）")
        exp.destroy_node()
        rclpy.shutdown()
        return

    # 存入 robotmem
    print("  存入 robotmem ...")
    learn_ok = False
    with RobotMemory(db_path=DB_PATH, collection=COLLECTION,
                     embed_backend="none") as mem:
        with mem.session(context={"task": "navigation", "phase": "explore"}) as sid:
            try:
                mem.learn(
                    f"导航路线 {len(wps)} 航点",
                    context={
                        "spatial": {
                            "waypoints": wps,
                            "start": wps[0],
                            "end": wps[-1],
                        },
                        "params": {"distance": e_dist, "duration": e_time},
                        "task": {"type": "navigation", "result": "success"},
                    },
                )
                print("  learn ✓")
                learn_ok = True
            except Exception as e:
                print(f"  learn ERROR: {e}")

            try:
                mem.save_perception(
                    f"探索轨迹 {len(e_pos)} 点",
                    perception_type="proprioceptive",
                    data=json.dumps({"positions": e_pos[:200], "waypoints": wps}),
                )
                print("  save_perception ✓")
            except Exception as e:
                print(f"  save_perception ERROR: {e}")

    if not learn_ok:
        print("  ERROR: learn 失败，无法进行返航 Demo")
        exp.destroy_node()
        rclpy.shutdown()
        return

    exp.destroy_node()

    print("\n  暂停 2s（模拟新任务：返回出发点）...")
    time.sleep(2)

    # ── Session 2: 记忆导航（recall 起点，直接返航） ──
    print("\n--- Session 2: 记忆返航 ---")
    return_wps = []
    with RobotMemory(db_path=DB_PATH, collection=COLLECTION,
                     embed_backend="none") as mem:
        with mem.session(context={"task": "navigation", "phase": "return"}) as sid:
            results = mem.recall("导航路线 航点")
            print(f"  recall: {len(results)} 条记忆")
            for r in results:
                ctx = r.get("context")
                if isinstance(ctx, str):
                    try:
                        ctx = json.loads(ctx)
                    except json.JSONDecodeError:
                        continue
                if isinstance(ctx, dict):
                    spatial = ctx.get("spatial", {})
                    start = spatial.get("start")
                    all_wps = spatial.get("waypoints", [])
                    if start and len(all_wps) >= 2:
                        # 反转关键航点：从当前位置直接回起点
                        reversed_wps = [tuple(p) for p in all_wps if isinstance(p, (list, tuple)) and len(p) == 2][::-1]
                        if not reversed_wps:
                            continue
                        # 间隔采样（每隔 2 个取 1 个），减少中间航点开销
                        sampled = reversed_wps[::2]
                        # 确保最后一个是起点
                        start_pt = tuple(start)
                        if sampled[-1] != start_pt:
                            sampled.append(start_pt)
                        return_wps = sampled
                        print(f"  全部航点: {len(all_wps)} → 采样: {len(return_wps)}（反转，直接返航）")
                        break

    if not return_wps:
        print("  ERROR: 未能提取航点!")
        rclpy.shutdown()
        return

    nav = NavNode(return_wps, cmd_topic=cmd_topic, odom_topic=odom_topic)
    while rclpy.ok() and not nav.done:
        rclpy.spin_once(nav, timeout_sec=0.1)

    n_pos = nav.positions
    n_time = n_pos[-1][2] if n_pos else 0
    n_dist = path_len(n_pos)
    nav.destroy_node()
    rclpy.shutdown()

    # ── 结果 ──
    print("\n" + "=" * 55)
    print("结果对比")
    print("=" * 55)
    print(f"  Session 1 (探索):   {e_time:6.1f}s  {e_dist:6.2f}m  (随机探索)")
    print(f"  Session 2 (返航):   {n_time:6.1f}s  {n_dist:6.2f}m  (记忆导航)")
    if e_time > 0:
        pct_time = (1 - n_time / e_time) * 100
        print(f"  时间节省:          {pct_time:6.1f}%")
    if e_dist > 0:
        pct_dist = (1 - n_dist / e_dist) * 100
        print(f"  距离节省:          {pct_dist:6.1f}%")
    print(f"\n  证明: 走过的路不用再走第二次 — 记忆驱动直接返航")
    print(f"\n  DB: {DB_DIR}")


if __name__ == "__main__":
    main()
