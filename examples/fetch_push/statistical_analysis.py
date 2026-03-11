"""robotmem 5-seed 实验统计分析 — 配对 t-test + 效果量

用 numpy 手动实现（scipy 不可用），输出：
  - 配对 t-test: t 统计量, p-value
  - 95% 置信区间（Delta 均值）
  - Cohen's d 效果量
  - Phase A vs B vs C 全量统计

运行:
  cd examples/fetch_push
  source .venv/bin/activate
  python statistical_analysis.py
"""

from __future__ import annotations

import math
import os

import numpy as np

# ── 原始数据（5-seed ROS Node 实验） ──

SEEDS = [42, 123, 456, 789, 2026]
PHASE_A = np.array([53, 48, 54, 46, 51], dtype=float)  # 基线（无记忆）
PHASE_B = np.array([55, 56, 57, 50, 47], dtype=float)  # 写入记忆
PHASE_C = np.array([63, 65, 75, 67, 63], dtype=float)  # 利用记忆

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def paired_t_test(x: np.ndarray, y: np.ndarray) -> dict:
    """配对 t-test（双尾）— 纯 numpy 实现

    H0: mean(y - x) = 0
    H1: mean(y - x) ≠ 0
    """
    n = len(x)
    diff = y - x
    d_bar = np.mean(diff)
    s_d = np.std(diff, ddof=1)  # 样本标准差
    se = s_d / math.sqrt(n)
    t_stat = d_bar / se
    df = n - 1

    # p-value（双尾）— 用 t 分布的正则不完全 Beta 函数近似
    p_value = _t_pvalue_two_sided(t_stat, df)

    # 95% 置信区间
    t_crit = _t_critical(0.025, df)  # 双尾 5%
    ci_low = d_bar - t_crit * se
    ci_high = d_bar + t_crit * se

    # Cohen's d（配对版本）
    cohens_d = d_bar / s_d

    return {
        "n": n,
        "mean_diff": d_bar,
        "std_diff": s_d,
        "se": se,
        "t_stat": t_stat,
        "df": df,
        "p_value": p_value,
        "ci_95": (ci_low, ci_high),
        "cohens_d": cohens_d,
    }


def _t_pvalue_two_sided(t: float, df: int) -> float:
    """用 Beta 正则化不完全函数计算 t 分布的双尾 p-value"""
    x = df / (df + t * t)
    p_one_tail = 0.5 * _regularized_incomplete_beta(df / 2, 0.5, x)
    return 2 * p_one_tail


def _t_critical(alpha: float, df: int) -> float:
    """牛顿法求 t 分布临界值（精度 1e-8）"""
    # 初始估计：正态分布分位数
    t = 2.0 if alpha < 0.05 else 1.5
    for _ in range(50):
        p = _t_pvalue_two_sided(t, df) / 2  # 单尾
        # 用数值微分
        dt = 0.0001
        p2 = _t_pvalue_two_sided(t + dt, df) / 2
        dp = (p2 - p) / dt
        if abs(dp) < 1e-15:
            break
        t -= (p - alpha) / dp
        t = max(0.01, t)
    return t


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a,b) — 连分式展开（Lentz 算法）"""
    if x < 0 or x > 1:
        return 0.0
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    # 使用对称性提高收敛
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularized_incomplete_beta(b, a, 1.0 - x)

    # log(x^a * (1-x)^b / (a * Beta(a,b)))
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a

    # Lentz 连分式
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, 200):
        # 偶数项
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= d * c

        # 奇数项
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta

        if abs(delta - 1.0) < 1e-10:
            break

    return front * f


def effect_size_label(d: float) -> str:
    """Cohen's d 效果量标签"""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("robotmem 5-seed 实验 — 统计分析")
    print("=" * 60)

    # 描述统计
    print("\n--- 描述统计 ---")
    for name, data in [("Phase A", PHASE_A), ("Phase B", PHASE_B), ("Phase C", PHASE_C)]:
        print(f"  {name}: mean={np.mean(data):.1f}%, std={np.std(data, ddof=1):.1f}%, "
              f"min={np.min(data):.0f}%, max={np.max(data):.0f}%")

    delta = PHASE_C - PHASE_A
    print(f"\n  Delta(C-A): mean={np.mean(delta):.1f}%, std={np.std(delta, ddof=1):.1f}%")
    print(f"  全部正向: {all(d > 0 for d in delta)} (min={np.min(delta):.0f}%)")

    # 配对 t-test: Phase A vs Phase C
    print("\n--- 配对 t-test: Phase A vs Phase C ---")
    result_ac = paired_t_test(PHASE_A, PHASE_C)
    _print_ttest(result_ac)

    # 配对 t-test: Phase A vs Phase B
    print("\n--- 配对 t-test: Phase A vs Phase B ---")
    result_ab = paired_t_test(PHASE_A, PHASE_B)
    _print_ttest(result_ab)

    # 配对 t-test: Phase B vs Phase C
    print("\n--- 配对 t-test: Phase B vs Phase C ---")
    result_bc = paired_t_test(PHASE_B, PHASE_C)
    _print_ttest(result_bc)

    # 保存结果
    lines = _format_results(result_ac, result_ab, result_bc)
    result_file = os.path.join(RESULTS_DIR, "statistical_analysis.txt")
    with open(result_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\n结果已保存: {result_file}")


def _print_ttest(r: dict):
    print(f"  n = {r['n']}")
    print(f"  mean diff = {r['mean_diff']:.1f}%")
    print(f"  std diff  = {r['std_diff']:.1f}%")
    print(f"  SE        = {r['se']:.2f}")
    print(f"  t({r['df']})    = {r['t_stat']:.3f}")
    print(f"  p-value   = {r['p_value']:.6f} {'***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else 'ns'}")
    print(f"  95% CI    = [{r['ci_95'][0]:.1f}%, {r['ci_95'][1]:.1f}%]")
    print(f"  Cohen's d = {r['cohens_d']:.2f} ({effect_size_label(r['cohens_d'])})")


def _format_results(ac: dict, ab: dict, bc: dict) -> list[str]:
    lines = [
        "=" * 60,
        "robotmem 5-seed 实验 — 统计分析结果",
        "=" * 60,
        f"日期: 2026-03-11",
        f"数据: ROS 2 Node FetchPush-v4, 每阶段 100 episodes, 5 seeds",
        "",
        "--- 原始数据 ---",
        f"Seeds:   {SEEDS}",
        f"Phase A: {PHASE_A.tolist()} (mean={np.mean(PHASE_A):.1f}%)",
        f"Phase B: {PHASE_B.tolist()} (mean={np.mean(PHASE_B):.1f}%)",
        f"Phase C: {PHASE_C.tolist()} (mean={np.mean(PHASE_C):.1f}%)",
        f"Delta:   {(PHASE_C - PHASE_A).tolist()} (mean={np.mean(PHASE_C - PHASE_A):.1f}%)",
        "",
        "--- 配对 t-test ---",
        "",
        "Phase A vs Phase C (核心结论):",
        f"  t({ac['df']}) = {ac['t_stat']:.3f}, p = {ac['p_value']:.6f}",
        f"  95% CI = [{ac['ci_95'][0]:.1f}%, {ac['ci_95'][1]:.1f}%]",
        f"  Cohen's d = {ac['cohens_d']:.2f} ({effect_size_label(ac['cohens_d'])})",
        f"  结论: {'显著' if ac['p_value'] < 0.05 else '不显著'} (p {'<' if ac['p_value'] < 0.05 else '>'} 0.05)",
        "",
        "Phase A vs Phase B:",
        f"  t({ab['df']}) = {ab['t_stat']:.3f}, p = {ab['p_value']:.6f}",
        f"  Cohen's d = {ab['cohens_d']:.2f} ({effect_size_label(ab['cohens_d'])})",
        "",
        "Phase B vs Phase C:",
        f"  t({bc['df']}) = {bc['t_stat']:.3f}, p = {bc['p_value']:.6f}",
        f"  Cohen's d = {bc['cohens_d']:.2f} ({effect_size_label(bc['cohens_d'])})",
    ]
    return lines


if __name__ == "__main__":
    main()
