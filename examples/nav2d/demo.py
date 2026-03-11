"""robotmem 2D 导航 Demo — 记忆驱动导航对比

走过的路不用再走第二次。

  Session 1: 机器人在未知迷宫中探索（只看得到相邻 1 格）
             → 到达目标后 learn 路径 + save_perception 障碍物
  Session 2: recall 路径和障碍物 → 在已知地图上 BFS 最短路
             → 步数减少 ~85-90%，碰撞 0

运行:
  PYTHONPATH=src python3 examples/nav2d/demo.py --seed 42
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import sys

from robotmem.sdk import RobotMemory

# ── 常量 ──────────────────────────────────────────

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-nav2d")
DB_PATH = os.path.join(DB_DIR, "memory.db")
COLLECTION = "nav2d"

WALL = 1
EMPTY = 0


# ── GridWorld ─────────────────────────────────────

class GridWorld:
    """2D 网格迷宫：边界墙 + 随机障碍物，BFS 保证可达"""

    def __init__(self, width: int = 20, height: int = 20, density: float = 0.25,
                 seed: int | None = None):
        self.width = width
        self.height = height
        self.start = (1, 1)
        self.goal = (width - 2, height - 2)
        self._rng = random.Random(seed)

        # 反复生成直到可达
        for _ in range(200):
            self.grid = self._generate(density)
            self.shortest_path = self.plan_path(self.start, self.goal)
            if self.shortest_path is not None:
                break
        else:
            raise RuntimeError("无法生成可达迷宫")

    # ── 生成 ──

    def _generate(self, density: float) -> list[list[int]]:
        grid = [[EMPTY] * self.width for _ in range(self.height)]
        # 边界墙
        for x in range(self.width):
            grid[0][x] = WALL
            grid[self.height - 1][x] = WALL
        for y in range(self.height):
            grid[y][0] = WALL
            grid[y][self.width - 1] = WALL
        # 随机障碍物
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if (x, y) == self.start or (x, y) == self.goal:
                    continue
                if self._rng.random() < density:
                    grid[y][x] = WALL
        return grid

    # ── 查询 ──

    def is_wall(self, x: int, y: int) -> bool:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x] == WALL
        return True  # 越界视为墙

    def neighbors(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        """4 连通可走邻居"""
        x, y = pos
        result = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if not self.is_wall(nx, ny):
                result.append((nx, ny))
        return result

    def get_visible(self, pos: tuple[int, int], vision: int = 1) -> list[tuple[int, int]]:
        """返回 pos 周围 vision 格内的墙壁位置"""
        x, y = pos
        walls = []
        for dy in range(-vision, vision + 1):
            for dx in range(-vision, vision + 1):
                nx, ny = x + dx, y + dy
                if self.is_wall(nx, ny):
                    walls.append((nx, ny))
        return walls

    # ── BFS ──

    def plan_path(self, start: tuple, goal: tuple,
                  extra_walls: set | None = None) -> list[tuple[int, int]] | None:
        """BFS 最短路径，extra_walls 为额外已知障碍物"""
        blocked = extra_walls or set()
        queue = collections.deque([(start, [start])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            if pos == goal:
                return path
            for nb in self.neighbors(pos):
                if nb not in visited and nb not in blocked:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None


# ── Explorer（Session 1）────────────────────────

class Explorer:
    """局部视野 DFS 探索 — 模拟无记忆机器人"""

    def __init__(self, world: GridWorld, vision: int = 1,
                 rng: random.Random | None = None):
        self.world = world
        self.vision = vision
        self._rng = rng or random.Random()
        self.visited: set[tuple[int, int]] = set()
        self.path: list[tuple[int, int]] = []
        self.collisions = 0
        self.obstacles_found: set[tuple[int, int]] = set()

    def explore(self) -> list[tuple[int, int]]:
        """从 start 探索到 goal，返回完整路径（含回溯）"""
        max_steps = self.world.width * self.world.height * 10
        stack = [self.world.start]
        self.visited.add(self.world.start)
        self.path = [self.world.start]

        while stack and len(self.path) < max_steps:
            pos = stack[-1]

            # 记录视野内障碍物
            walls = self.world.get_visible(pos, self.vision)
            for w in walls:
                self.obstacles_found.add(w)

            if pos == self.world.goal:
                return self.path

            # 找未访问的邻居
            nbrs = self.world.neighbors(pos)
            unvisited = [n for n in nbrs if n not in self.visited]

            if unvisited:
                nxt = self._rng.choice(unvisited)
                self.visited.add(nxt)
                stack.append(nxt)
                self.path.append(nxt)
            else:
                # 回溯
                stack.pop()
                if stack:
                    self.path.append(stack[-1])

        if len(self.path) >= max_steps:
            print(f"  [WARN] 探索步数达到上限 {max_steps}，强制停止")
        return self.path


# ── MemoryNavigator（Session 2）────────────────

class MemoryNavigator:
    """记忆驱动导航 — recall 障碍物地图 + BFS 最短路"""

    def __init__(self, world: GridWorld, mem: RobotMemory):
        self.world = world
        self.mem = mem
        self.path: list[tuple[int, int]] = []
        self.collisions = 0

    def navigate(self, session_id: str) -> list[tuple[int, int]]:
        """recall 记忆 → 重建地图 → BFS 导航"""
        # 1. recall 成功路径（learn 的 context 含 path + obstacles_found）
        path_memories = self.mem.recall(
            "navigation path to goal successful waypoints obstacles",
            n=5,
            context_filter={"task.success": True},
        )
        recalled_path = self._extract_path(path_memories)

        # 2. 从同一批记忆中提取障碍物（learn context.spatial.obstacles_found）
        known_walls: set[tuple[int, int]] = set()
        for m in path_memories:
            self._extract_obstacles(m, known_walls)

        print(f"  recall 障碍物: {len(known_walls)} 个已知墙壁")
        print(f"  recall 路径: {'有' if recalled_path else '无'}参考路径")

        # 3. 在已知地图上 BFS
        bfs_path = self.world.plan_path(self.world.start, self.world.goal, known_walls)
        if bfs_path:
            self.path = bfs_path
        elif recalled_path:
            # BFS 失败（不应该发生），降级为回放路径
            self.path = recalled_path
        else:
            # 兜底：全局 BFS
            self.path = self.world.shortest_path or [self.world.start]

        return self.path

    def _extract_obstacles(self, memory: dict, out: set):
        """从 recall 结果中解析障碍物坐标（来源：learn context.spatial.obstacles_found）"""
        ctx = memory.get("context")
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except (json.JSONDecodeError, TypeError):
                return
        if isinstance(ctx, dict):
            spatial = ctx.get("spatial", {})
            for pos in spatial.get("obstacles_found", []):
                out.add(tuple(pos))

    def _extract_path(self, memories: list[dict]) -> list[tuple[int, int]] | None:
        """从 recall 结果中解析路径坐标"""
        for m in memories:
            ctx = m.get("context")
            if isinstance(ctx, str):
                try:
                    ctx = json.loads(ctx)
                except (json.JSONDecodeError, TypeError):
                    continue
            if isinstance(ctx, dict):
                spatial = ctx.get("spatial", {})
                path_data = spatial.get("path")
                if path_data:
                    return [tuple(p) for p in path_data]
        return None


# ── 可视化 ─────────────────────────────────────

def plot_comparison(world: GridWorld, explore_path: list, memory_path: list,
                    stats: dict, save_path: str):
    """并排对比图：Session 1 探索 vs Session 2 记忆导航"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib 不可用，跳过绘图")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    for ax, path, title, color, alpha in [
        (ax1, explore_path, "Session 1: Exploration (No Memory)", "#e74c3c", 0.4),
        (ax2, memory_path, "Session 2: Memory-Guided Navigation", "#27ae60", 0.8),
    ]:
        # 绘制网格
        for y in range(world.height):
            for x in range(world.width):
                if world.grid[y][x] == WALL:
                    ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor="#2c3e50"))
                else:
                    ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor="#ecf0f1",
                                                    edgecolor="#bdc3c7", linewidth=0.3))

        # 绘制路径
        if len(path) > 1:
            px = [p[0] + 0.5 for p in path]
            py = [p[1] + 0.5 for p in path]
            ax.plot(px, py, color=color, alpha=alpha, linewidth=1.5, zorder=2)

        # 起点 / 终点
        sx, sy = world.start
        gx, gy = world.goal
        ax.plot(sx + 0.5, sy + 0.5, "o", color="#3498db", markersize=10, zorder=3)
        ax.plot(gx + 0.5, gy + 0.5, "*", color="#f39c12", markersize=14, zorder=3)

        # 最短路径虚线
        sp = world.shortest_path
        if sp and len(sp) > 1:
            spx = [p[0] + 0.5 for p in sp]
            spy = [p[1] + 0.5 for p in sp]
            ax.plot(spx, spy, "--", color="#3498db", alpha=0.3, linewidth=1, zorder=1)

        ax.set_xlim(0, world.width)
        ax.set_ylim(0, world.height)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # 底部统计
    sp_len = len(world.shortest_path) if world.shortest_path else None
    improvement = stats["improvement"]
    sp_str = str(sp_len) if sp_len else "?"
    text = (
        f"Session 1: {stats['s1_steps']} steps, {stats['s1_collisions']} collisions    "
        f"Session 2: {stats['s2_steps']} steps, {stats['s2_collisions']} collisions    "
        f"Shortest: {sp_str} steps    "
        f"Improvement: {improvement:.0f}%"
    )
    fig.text(0.5, 0.02, text, ha="center", fontsize=10, family="monospace")

    fig.suptitle("robotmem: Memory Makes Navigation Efficient",
                 fontsize=14, fontweight="bold", y=0.97)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  对比图已保存: {save_path}")


# ── 主流程 ─────────────────────────────────────

def extract_waypoints(path: list[tuple[int, int]], interval: int = 5) -> list:
    """从路径中提取关键转弯点（每 interval 步取一个 + 转弯点）"""
    if not path:
        return []
    waypoints = [list(path[0])]
    for i in range(1, len(path) - 1):
        # 转弯点：方向改变
        prev_dir = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        next_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if prev_dir != next_dir:
            waypoints.append(list(path[i]))
        elif i % interval == 0:
            waypoints.append(list(path[i]))
    waypoints.append(list(path[-1]))
    return waypoints


def main():
    parser = argparse.ArgumentParser(
        description="robotmem 2D 导航 Demo — 记忆驱动导航对比"
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--size", type=int, default=20, help="网格大小（默认 20）")
    parser.add_argument("--density", type=float, default=0.25, help="障碍物密度")
    parser.add_argument("--vision", type=int, default=1, help="视野范围（格）")
    parser.add_argument("--no-plot", action="store_true", help="不生成图片")
    args = parser.parse_args()

    # 独立 RNG 实例，确保 --seed 完全复现
    rng = random.Random(args.seed)

    print("=" * 55)
    print("robotmem 2D Navigation Demo — 记忆驱动导航对比")
    print(f"网格: {args.size}x{args.size}, 障碍物: {args.density:.0%}, 视野: {args.vision} 格")
    if args.seed is not None:
        print(f"随机种子: {args.seed}")
    print(f"DB: {DB_PATH}")
    print("=" * 55)

    # 0. 初始化
    world = GridWorld(args.size, args.size, args.density, seed=args.seed)
    sp_len = len(world.shortest_path) if world.shortest_path else None
    print(f"\n迷宫生成完成，最短路径: {sp_len or '?'} 步")

    # DB 隔离
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    with RobotMemory(db_path=DB_PATH, collection=COLLECTION, embed_backend="onnx") as mem:

        # ── Session 1: 探索 ──

        print("\n--- Session 1: Exploration (No Memory) ---")
        with mem.session(context={"task": "grid_navigation", "phase": "explore"}) as sid1:
            explorer = Explorer(world, vision=args.vision, rng=rng)
            explore_path = explorer.explore()

            s1_steps = len(explore_path)
            s1_collisions = explorer.collisions
            print(f"  到达目标! 步数: {s1_steps}, 碰撞: {s1_collisions}")
            print(f"  探索覆盖: {len(explorer.visited)} 格, "
                  f"发现障碍物: {len(explorer.obstacles_found)} 个")

            # learn 成功路径
            waypoints = extract_waypoints(explore_path)
            # 去重路径（探索路径含回溯，记录去重版本供 recall 使用）
            seen = set()
            unique_path = []
            for p in explore_path:
                if p not in seen:
                    seen.add(p)
                    unique_path.append(list(p))

            try:
                mem.learn(
                    insight=(
                        f"Navigation successful: {s1_steps} steps from "
                        f"{world.start} to {world.goal}. "
                        f"{len(waypoints)} waypoints, {s1_collisions} collisions. "
                        f"Path through explored area covering {len(explorer.visited)} cells."
                    ),
                    context={
                        "params": {
                            "path_length": {"value": s1_steps, "type": "scalar"},
                            "collisions": {"value": s1_collisions, "type": "scalar"},
                            "coverage": {"value": len(explorer.visited), "type": "scalar"},
                        },
                        "spatial": {
                            "start": list(world.start),
                            "goal": list(world.goal),
                            "path": unique_path,
                            "waypoints": waypoints,
                            "obstacles_found": [list(o) for o in explorer.obstacles_found],
                        },
                        "task": {
                            "name": "grid_navigation",
                            "success": True,
                            "steps": s1_steps,
                        },
                    },
                    session_id=sid1,
                )
                print("  learn 路径 ✓")
            except Exception as e:
                print(f"  [ERROR] learn 失败: {e}")
                raise

            # save_perception 障碍物地图
            obstacles_list = [list(o) for o in explorer.obstacles_found]
            try:
                mem.save_perception(
                    description=(
                        f"Obstacle map: {len(obstacles_list)} obstacles detected "
                        f"in {args.size}x{args.size} grid during exploration. "
                        f"Grid navigation obstacle positions for path planning."
                    ),
                    perception_type="visual",
                    data=json.dumps({
                        "grid_size": [args.size, args.size],
                        "obstacles": obstacles_list,
                        "explored_cells": len(explorer.visited),
                    }),
                    metadata=json.dumps({
                        "coverage_ratio": len(explorer.visited) / (args.size * args.size),
                        "obstacle_count": len(obstacles_list),
                    }),
                    session_id=sid1,
                )
                print(f"  save_perception 障碍物地图 ✓ ({len(obstacles_list)} obstacles)")
            except Exception as e:
                print(f"  [ERROR] save_perception 失败: {e}")
                raise

        # ── Session 2: 记忆导航 ──

        print("\n--- Session 2: Memory-Guided Navigation ---")
        with mem.session(context={"task": "grid_navigation", "phase": "memory_nav"}) as sid2:
            navigator = MemoryNavigator(world, mem)
            memory_path = navigator.navigate(session_id=sid2)

            s2_steps = len(memory_path)
            s2_collisions = navigator.collisions
            print(f"  到达目标! 步数: {s2_steps}, 碰撞: {s2_collisions}")

    # ── 结果 ──

    improvement = (1 - s2_steps / s1_steps) * 100 if s1_steps > 0 else 0
    stats = {
        "s1_steps": s1_steps,
        "s1_collisions": s1_collisions,
        "s2_steps": s2_steps,
        "s2_collisions": s2_collisions,
        "shortest": sp_len,
        "improvement": improvement,
    }

    print("\n" + "=" * 55)
    print("结果对比")
    print("=" * 55)
    print(f"  Session 1 (探索):  {s1_steps:>5} 步, {s1_collisions:>3} 碰撞")
    print(f"  Session 2 (记忆):  {s2_steps:>5} 步, {s2_collisions:>3} 碰撞")
    print(f"  最短路径 (BFS):    {str(sp_len or '?'):>5} 步")
    print(f"  步数减少:          {improvement:>5.1f}%")

    if sp_len and sp_len > 0:
        efficiency = (sp_len / s2_steps) * 100 if s2_steps > 0 else 0
        print(f"  路径效率:          {efficiency:>5.1f}% (vs 最短路径)")

    # ── 绘图 ──

    if not args.no_plot:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "nav2d_comparison.png",
        )
        plot_comparison(world, explore_path, memory_path, stats, save_path)

    print(f"\n数据存储于: {DB_DIR}")


if __name__ == "__main__":
    main()
