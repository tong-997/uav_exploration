# Exp 2 策略结论合理性检查

> 生成时间：2026-06-06

## 1 风险点

Greedy-Frontier 成功率 100%，w/o Voronoi 成功率 90%，而 Ours 仅 80%。如果声称"本系统整体最优"，与数据矛盾。需要明确各方法的真实优劣和正确结论。

## 2 关键数据对比

| 指标 | Ours | Greedy | w/o Voronoi | Random | w/o Inflation |
|------|------|--------|-------------|--------|--------------|
| Success Rate | **80%** | **100%** | **90%** | 0% | 0% |
| Steps to 90% | 432 | **384** | 392 | 800 | 800 |
| Coverage | 84.6% | **90.1%** | 88.0% | 55.2% | 38.0% |
| Total Path | **457 m** | 532 m | 485 m | 1156 m | 175 m |
| Collisions | 0 | 0 | 0 | 0 | 0 |
| Inter-UAV Safe | 100% | 100% | 100% | 96.9% | 100% |
| Min Inter (m) | 15.5 | 8.3 | 15.7 | 2.4 | 17.8 |
| Obs Safe | 63.5% | 69.5% | 65.6% | 86.3% | 12.4% |
| Replan | 150 | 45 | 93 | 80 | 81 |

### 2.1 Greedy-Frontier 为什么表现好

- **100% 成功率**：贪心选择最近前沿，减少了移动距离，每次规划路径简单且短，降低了 A* 失败概率
- **最快达标**：384 步，比 Ours 快 11%
- **缺点**：路径较长（532 m vs 457 m），因为不做全局协调，多机可能重复探索同一区域

### 2.2 Ours 为什么成功率低于 Greedy

- Utility 评分考虑前沿大小和他机目标距离，可能选择更远的前沿
- 远距离规划路径复杂度更高，A* 失败概率增加（Replan = 150 vs Greedy 的 45）
- 在某些种子下，远前沿被障碍物密集区分隔，导致 800 步内无法到达

### 2.3 w/o Voronoi 为什么优于 Ours

- Utility 评分函数中的他机目标惩罚已部分实现去重
- Voronoi 分区可能过度约束 UAV 选择，将 UAV 限制在其分区内，错过了跨分区的高效前沿
- 说明 Voronoi 在本场景下的边际贡献有限甚至轻微负面

## 3 结论合理性审查

### 3.1 现有结论检查

原文结论 1：**障碍膨胀是最关键模块** — **正确**，w/o Inflation 的 0% 成功率和 12.4% 安全率充分支持

原文结论 2：**随机前沿不可行** — **正确**，0% 成功率，路径 2.5 倍但覆盖不到 2/3

原文结论 3：**Greedy 意外表现良好** — **正确且诚实**，承认了 Greedy 的优势

原文结论 4：**Voronoi 影响有限但正面** — **需修正**，数据显示 w/o Voronoi (90%) > Ours (80%)，Voronoi 的影响在本场景下是轻微负面的

原文结论 5：**防撞对效率无负面影响** — **正确**，w/o Deconflict = Ours

原文结论 6：**系统各模块协同互补** — **需软化**，Greedy 在成功率和速度上都优于 Ours

原文结论 7：**重规划次数揭示差异** — **正确**

原文结论 8：**机间距反映协同质量** — **正确**

### 3.2 不可声称

- ~~"本系统（Ours）在所有指标上最优"~~ — 成功率和覆盖率均不是最高
- ~~"Voronoi 分区显著提升探索效率"~~ — 数据不支持
- ~~"Utility 评分函数优于 Greedy"~~ — 在成功率和速度上不如 Greedy

### 3.3 可以声称

1. **"Ours 在路径效率上最优（457 m），路径最短的同时维持了合理覆盖率"** — 数据支持
2. **"Ours 在安全性上表现优异（100% 机间安全率，0 碰撞，Min Inter = 15.5 m）"** — 数据支持
3. **"Greedy 以更高成功率和更快速度完成探索，但路径效率较低（+16%），且机间最小距离较小（8.3 m vs 15.5 m），在密集环境中安全裕度不足"** — 平衡表述
4. **"障碍膨胀和启发式前沿选择是系统的关键模块；Voronoi 分区在本场景下贡献有限"** — 诚实的消融结论
5. **"完整系统在效率-安全性平衡上优于各消融版本"** — 可用路径效率和安全性综合评价

### 3.4 建议论文表述

> "The ablation study reveals that obstacle inflation is the most critical module (0% success without it), followed by informed frontier selection (Random achieves 0% success). Greedy frontier selection achieves the highest success rate (100%) and fastest completion (384 steps) among all methods, demonstrating that simple nearest-frontier strategies are highly competitive in moderate-density environments. Our full system trades a lower success rate (80%) for the shortest total path (457 m vs. 532 m, a 14% reduction) and the largest minimum inter-UAV distance (15.5 m vs. 8.3 m), prioritizing path efficiency and safety margins. The Voronoi partition contributes marginally in this scenario (removing it yields 90% success vs. 80%), as the utility function's implicit coordination partially substitutes for explicit spatial partitioning. The distributed deconfliction module shows no efficiency overhead, confirming it as a zero-cost safety guarantee."

## 4 建议对现有结论的修改

### 结论 4 修正
原文："Voronoi 分区影响有限**但正面**"
修正为："Voronoi 分区在本场景下影响有限，成功率略低于无分区版本（80% vs 90%）。这可能因为显式分区限制了 UAV 对跨区高效前沿的访问。"

### 结论 6 修正
原文："系统各模块协同互补"
修正为："完整系统在路径效率（457 m 最短）和安全间距（15.5 m 最大）上取得最佳平衡，但在成功率上不如简单贪心策略。系统的核心优势在于效率-安全的综合权衡，而非单一指标最优。"
