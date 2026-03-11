"""PushT 策略模块 — PushTHeuristicPolicy + PushTMemoryPolicy

PushT 环境:
- Action: [x, y] 绝对位置 (0~512)，agent 向目标位置移动（速度受限）
- Obs: [agent_x, agent_y, block_x, block_y, block_angle]
- Reward: coverage (0~1)，T 形块与目标重叠率
- Episode: 300 步
- 目标: 固定 T-pose [256, 256, π/4]
"""

import numpy as np


class PushTHeuristicPolicy:
    """启发式策略: 温和推块向目标 + 可调噪声

    action 是 agent 的目标位置，agent 每步以有限速度向目标移动。

    策略:
    1. 接近: 小步移向 block 背面（推方向反向）
    2. 推送: 到位后轻推（小步穿过 block）
    3. 保持: block 接近目标后停止推送
    4. 绕行: agent 在 block 前面时绕到背面
    """

    TARGET = np.array([256.0, 256.0])
    BEHIND_DIST = 15.0
    APPROACH_SPEED = 15.0
    PUSH_SPEED = 8.0
    ARRIVE_THRESH = 5.0
    STOP_DIST = 15.0

    def __init__(self, noise_scale=15.0):
        self.noise_scale = noise_scale

    def act(self, obs):
        agent = np.array([obs[0], obs[1]])
        block = np.array([obs[2], obs[3]])

        to_target = self.TARGET - block
        dist_to_target = np.linalg.norm(to_target)

        if dist_to_target < self.STOP_DIST:
            return agent.tolist()

        push_dir = to_target / dist_to_target
        push_strength = np.clip(dist_to_target / 80.0, 0.2, 1.0)

        agent_to_block = block - agent
        dot = np.dot(agent_to_block, push_dir)

        if dot > 0:
            behind_pos = block - push_dir * self.BEHIND_DIST
            to_behind = behind_pos - agent
            behind_dist = np.linalg.norm(to_behind)

            if behind_dist > self.ARRIVE_THRESH:
                step_dir = to_behind / behind_dist
                step = agent + step_dir * min(behind_dist, self.APPROACH_SPEED)
            else:
                step = agent + push_dir * self.PUSH_SPEED * push_strength
        else:
            perp = np.array([-push_dir[1], push_dir[0]])
            side = 1.0 if np.dot(agent - block, perp) > 0 else -1.0
            detour = block - push_dir * 40 + perp * side * 30
            to_detour = detour - agent
            detour_dist = np.linalg.norm(to_detour)
            if detour_dist > 1:
                step = agent + (to_detour / detour_dist) * self.APPROACH_SPEED
            else:
                step = detour

        noise = np.random.randn(2) * self.noise_scale
        return np.clip(step + noise, 0, 512).tolist()


class PushTMemoryPolicy:
    """记忆驱动策略: stop-at-peak

    核心机制 — 记忆教 agent "什么时候该停":
    - Phase B 记录每个 episode 的 peak_coverage 和 peak_step
    - Phase C 从记忆学到:
      1. peak 通常多高 → 停推阈值（达到后保持）
      2. peak 通常在哪步 → 超时保护

    与 FetchPush PhaseAwareMemoryPolicy 的对比:
    - FetchPush: 推方向微调（成功率 42→67%）
    - PushT: 停推时机学习（peak 被保留而非被过冲破坏）
    - 原因: PushT 启发式经常达到高 peak 后过冲到 0
    """

    def __init__(self, base_policy, recalled_memories, memory_weight=0.3):
        self.base = base_policy
        self.weight = memory_weight

        # 从记忆学习停推参数
        self.stop_coverage = self._learn_stop_coverage(recalled_memories)
        self.max_push_steps = self._learn_max_steps(recalled_memories)

        # Episode 内状态
        self.peak = 0.0
        self.holding = False
        self.step_count = 0

    def _learn_stop_coverage(self, memories):
        """学习停推阈值: 成功 episode 的 peak coverage 分布"""
        if not memories:
            return 0.03  # 无记忆时的默认阈值

        peaks = []
        for m in memories:
            task = m.get("task", {})
            peak = task.get("peak_coverage", 0)
            if peak > 0:
                peaks.append(peak)

        if not peaks:
            return 0.03

        # 用 30th percentile 作为停推阈值（保守：比大多数 peak 低）
        return float(np.percentile(peaks, 30)) * 0.7

    def _learn_max_steps(self, memories):
        """学习最大推送步数: peak 通常出现在前 X 步"""
        if not memories:
            return 200

        steps = []
        for m in memories:
            params = m.get("params", {})
            ps = params.get("peak_step", {})
            if isinstance(ps, dict) and "value" in ps:
                steps.append(ps["value"])

        if not steps:
            return 200

        # 用 80th percentile: 大多数 peak 在此之前
        return int(np.percentile(steps, 80)) + 30

    def reset_episode(self):
        """每个 episode 重置"""
        self.peak = 0.0
        self.holding = False
        self.step_count = 0

    def update_coverage(self, reward):
        """每步调用: 追踪 coverage，决定是否停推"""
        self.step_count += 1

        if self.holding:
            return

        if reward > self.peak:
            self.peak = reward

        # 停推条件（至少推 10 步才允许停推，防止初始噪声误触发）:
        # 1. peak 达到阈值 AND coverage 从 peak 下降 50%+
        # 2. 超过最大推送步数
        if self.step_count < 10:
            return
        if self.peak > self.stop_coverage and reward < self.peak * 0.5:
            self.holding = True
        elif self.step_count > self.max_push_steps:
            self.holding = True

    def act(self, obs):
        if self.holding:
            # 停推: 保持当前位置（不再接触 block）
            return [float(obs[0]), float(obs[1])]

        return self.base.act(obs)
