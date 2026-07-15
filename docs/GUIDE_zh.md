# 论文与实验完全指南（中文版）

> 面向作者本人的理解、复现与校验手册。每一节先讲"是什么、为什么"，
> 再给"代码在哪、怎么跑、预期看到什么数字"。所有命令都在仓库根目录执行，
> Python 一律用 `./.venv/bin/python`。

---

## 1. 论文在证明什么（一句话 + 三幕）

**一句话**：车队调度里被当作"解释"的图注意力权重，会**比它所依据的遥测数据
活得更久**——数据过期时，解释既不报警、性能也不掉，但解释的内容已经悄悄
漂移到过期节点上；训练时见过退化也治不了它，换一条解耦的解释通道才有实测
改善。

**三幕结构**（正文叙事；提案的 RQ1–RQ4 / H1–H5 在方法章保留形式化定义）：

| 幕 | 问题 | 答案 | 主图 |
|---|---|---|---|
| 1 | 干净数据上注意力解释忠实吗？(RQ1a) | **不忠实**：margin-DEF = −0.541，比随机解释还差 | `story_act1_clean_def.png` |
| 2 | 遥测变陈旧时会发生什么？(H1–H4) | 注意力流向过期节点（WAMSN↑），忠实度躺地板、性能平坦——**无声漂移** | `story_act2_decoupling.png` |
| 3a | 训练时见过退化能治吗？(H5) | **不能**：H5′ 的忠实度反而更差（−0.77 vs −0.54） | `story_act3a_training_mitigation.png` |
| 3b | 换解释通道能治吗？（新增贡献） | **有改善**：解耦蒸馏头 +0.11（p=0.0001），但仍未达随机基线 | `story_act3b_coupled_vs_decoupled.png` |

---

## 2. 核心概念，用人话讲

### 2.1 AoI（Age of Information，信息年龄）

一辆车上一次可信上报距现在过了多少秒。`AoI = 当前仿真时间 − 最后可信读数时间`。
进隧道 → 信号丢失 → AoI 开始增长；出隧道且信号恢复 → AoI 归零。
观测里归一化为 `aoi / 60s`（60 s = 提案严重度阶梯的"极端"级）。

代码：`src/dispatch_marl/degradation.py` 的 `DegradationLayer.observe()`。

### 2.2 冻结式退化（本文的退化机制，提案 §7.2）

信号丢失的车**停止传输**：所有观察者（它自己 + 所有邻车）看到的都是它
**最后一次可信上报的位置和速度**（冻结快照），直到信号恢复。注意三点：

- 仿真器内部永远是真值，只有"观测"被替换——所以性能损失可以精确归因于
  "策略看到了什么"；
- 冻结对**所有人**生效（邻车特征、邻车排序、订单相对坐标都基于观测态）；
- 严重度旋钮 `outage_duration_s`：触发后信号持续丢失直到 AoI 达到该秒数
  （= 阶梯的"最大 AoI"），出隧道后也不立刻恢复（物理解释：接收机重捕获延迟）。

⚠️ 历史注意：2026-07-15 之前的结果用的是"噪声机制"（只给自身位置加 20 m
高斯噪声、邻车永远真值），信息损失几乎为零。旧结果保存在
`results/b1b2b3_sumo120_seed42_v1/`，其中"H5 逆转"结论在冻结机制下**不成立**，
只能作为机制敏感性引用。

### 2.3 DEF（调度解释忠实度）——"遮挡检验"

对一个决策：策略在完整图 G 上给它选的动作 a\* 的概率是 π(a\*|G)。
把解释（注意力）认为最重要的 top-k 节点做两种反事实：

- **Comprehensiveness（必要性）**：`Comp = π(a*|G) − π(a*|G去掉Rk)`
  ——遮掉重要节点，信心应该大跌（越大越忠实）；
- **Sufficiency（充分性）**：`Suff = π(a*|G) − π(a*|只留Rk)`
  ——只留重要节点，信心应该不怎么掉（越小越忠实）。

再和**同样大小的随机节点集**对比得到归一化增益：
`g_comp = Comp − Comp_rand`，`g_suff = Suff_rand − Suff`，
**DEF = ½(g_comp + g_suff)**。DEF > 0 ⇔ 注意力比随机解释更忠实。
对 k ∈ {1,2,3} 取平均，每个 k 采 5 个随机子集。

**margin-DEF（本文的方法学扩展）**：策略熵坍缩后 π(a\*)≈1，遮什么概率都
不动（概率版 DEF 幅度只剩 ~0.0015）。改在 **logit 边际**
`m = logit[a*] − max其他动作logit`（截断到 ±10）上做同样的检验，
信号恢复约 500 倍。两个版本共享同一批反事实前向，零额外开销。
正文用 margin-DEF，概率版（提案原定义）同表报告。

代码：`src/dispatch_marl/faithfulness.py`（`evaluate_decision`、
`decision_margin`）。每个决策约 26 次前向传播。

### 2.4 WAMSN（过期节点上的注意力质量）

`WAMSN = Σ αi·(AoIi/AoImax) / Σ αi`，只对车辆类节点求和（订单无 AoI）。
0 = 注意力全在新鲜节点；越接近 1 = 越集中在最陈旧的节点。
它回答"注意力**流去了哪**"；JS 漂移（附录）回答"分布**变了多少**"。

### 2.5 耦合 vs 解耦解释通道

- **耦合**：驱动决策的注意力权重本身当解释——模型免费送的，但没人保证忠实；
- **解耦**：一个旁路 MLP 读取（detach 的）GAT 节点嵌入、输出各节点重要性，
  **不参与决策**。训练目标是"遮挡蒸馏"：每个节点的真实重要性 = 单独遮掉它
  后决策边际的跌幅 Δm，头学习拟合这个排序（KL 损失）。

代码：`src/dispatch_marl/models/explainer.py`、`scripts/distill_explainer.py`、
`scripts/eval_explainer.py`（配对评估：同一决策、同一随机基线，只有 top-k
的选取来源不同）。

---

## 3. 系统架构速览

```
SUMO 1.20（渝北中央公园 OSM 路网，20 出租车，50 乘客，1200 s/回合）
  │  traci（socket）
  ▼
DispatchEnv（PettingZoo ParallelEnv, src/dispatch_marl/env.py）
  │  每个空闲出租车一张自中心异构图：1 自身 + 5 邻车 + 5 订单（各 5 维特征）
  │  动作 Discrete(6)：0=不动，k=接第 k 近订单
  │  奖励（团队共享）= 10×完成接客 + 0.5×成功派单 − 0.001×平均等待
  ▼
DegradationLayer（观测边界，冻结语义，见 §2.2）
  ▼
DispatchGATPolicy（手写 2 层 4 头 GAT + actor + critic，CTDE 中心化 critic）
  │  forward() 直接返回注意力张量 (层,批,头,N,N) —— 忠实度管线的输入
  ▼
MAPPO（rollout → GAE → clipped PPO，参数共享，src/dispatch_marl/training.py）
```

模型条件表：

| 代号 | 是什么 | 用途 |
|---|---|---|
| B0 | SUMO 内建贪心匹配器 | 性能上界参照（32 接单，学习策略只有 6–8——论文按提案风险条款定位：贡献是测量忠实度，不是造 SOTA 调度器） |
| B1 | MAPPO+MLP（无图） | 性能背景表；无注意力通道，不进忠实度分析 |
| B2 | GAT-MAPPO（主角），干净训练 | 第一、二幕的被测对象 |
| B3 | B2 去掉 AoI 特征 | 干净训练下该消融是空转的（AoI 恒 0），只作背景 |
| H5′ | B2 同配置 + 训练时开隧道冻结退化（outage 30 s） | 第三幕 a |
| 解耦头 | B2 之上的旁路解释器 | 第三幕 b |

---

## 4. 实验协议（每个选择的理由）

- **严重度阶梯**：最大 AoI ∈ {0(clean), 5, 15, 30, 60} s，单轴（提案 §7.2）。
  丢包轴只留作附录（`--with-dropout-axis`）。
- **需求划分**：`add_taxis.py --variants 20` 生成 20 份乘客需求变体
  （车队初始位置完全相同、只有乘客不同），按时间顺序 70/15/15 切
  训练/验证/测试。**所有正文数字都在 3 份测试需求上测**（回合 i 用第
  i mod 3 份）。清单：`scenarios/yubei/central_park/demand_manifest.json`。
- **种子**：8 个（42–49），控制环境与随机基线；每 (级别×种子) 3 回合。
- **随机评估（重要）**：这些策略都会熵坍缩，argmax 评估退化为全不动
  （0 接单、奖励 −45.59）。所有评估都用采样动作——和训练时行为一致。
- **统计**：H1/H3 = 逐决策 Spearman + 1 万次置换检验；H2 = 格子级
  配对符号翻转检验（忠实度衰减率 vs 性能衰减率）；H4 = 合并相关。
  CI 用 bootstrap。已知局限：逐决策检验未处理回合内聚类（写论文时注明）。

---

## 5. 手动复现步骤

### 第 0 步：环境（Intel Mac 专用路径）

```bash
cd "<仓库根目录>"
/usr/local/bin/python3.11 -m venv .venv
./.venv/bin/pip install --upgrade pip setuptools wheel
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install torch==2.2.2 "numpy==1.26.4"   # Intel mac 最后的 torch 版本
./.venv/bin/pip install eclipse-sumo==1.20.0 traci==1.20.0 sumolib==1.20.0
```

注意事项（README "Intel-mac note" 有完整版）：
- SUMO 1.27 没有 Intel 版；PyPI 的 1.20 wheel 需要 xerces-c **3.2**
  （brew 的 3.3 不兼容，需 `brew extract` 装 3.2.5 后软链）；
- 不装 libsumo（缺 GDAL 依赖），包自动回落到 socket traci；
- `netconvert` 在本机不可用 → **不能重建场景**，但已提交的场景可直接用；
- `.env` 里 `SUMO_HOME` 指向 venv 内的 wheel 路径。

**校验**：
```bash
./.venv/bin/python scripts/test_policy.py        # 期望结尾: OK — GAT-MAPPO policy runs end-to-end
./.venv/bin/python scripts/test_faithfulness.py  # 期望结尾: OK — faithfulness pipeline works end-to-end
```

### 第 1 步：理解冻结机制（不训练，30 秒）

```bash
./.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, 'src')
import numpy as np
from dispatch_marl.degradation import DegradationLayer, DegradationConfig
cfg = DegradationConfig(mode='tunnel_triggered', outage_duration_s=30.0)
layer = DegradationLayer(cfg, frozenset({'TUNNEL'}), np.random.default_rng(0))
print(layer.observe('t', 100, 200, 5, 'open',   0.0))   # 可信: (100,200,5,False,0)
print(layer.observe('t', 150, 250, 5, 'TUNNEL', 10.0))  # 隧道内: 冻结在(100,200), AoI=10
print(layer.observe('t', 180, 280, 5, 'open',   20.0))  # 已出隧道仍丢失: AoI=20 < 30
print(layer.observe('t', 195, 295, 5, 'open',   31.0))  # AoI 达 30 → 恢复, AoI=0
EOF
```
逐行看输出即可验证 §2.2 的全部语义。

### 第 2 步：训练（或跳过，直接用已有 checkpoint）

已提交的可用 checkpoint（跳过训练直接到第 3 步）：
- B2：`results/b1b2b3_sumo120_seed42_v1/B2_gat/ckpt_best.pt`（epoch 91）
- H5′：`results/story_freeze_v1/H5b_train/ckpt_best.pt`（epoch 17）

自己重训（每个 300 轮，本机 CPU 约 1–2 小时）：
```bash
# B2（干净训练）
./.venv/bin/python scripts/train.py --area central_park --epochs 300 --seed 42 \
  --save-every 50 --faith-every-epochs 10 --faith-sample-size 64
# H5′（退化感知训练，唯一区别是这两个参数）
./.venv/bin/python scripts/train.py --area central_park --epochs 300 --seed 42 \
  --save-every 50 --degradation tunnel_triggered --outage-duration 30 \
  --faith-every-epochs 10 --faith-sample-size 64
```
**校验**：训练日志里熵（H 列）会在 ~25–130 轮间坍缩到 <0.05、接单归零——
这是已知现象，`ckpt_best.pt`（滚动均值选出）才是可用模型。B2 最佳滚动均值
参考 ~14、H5′ ~10（种子相同应完全一致；若 SUMO 版本不同会有差异）。

### 第 3 步：单点评估

```bash
CKPT=results/b1b2b3_sumo120_seed42_v1/B2_gat/ckpt_best.pt
./.venv/bin/python scripts/eval_policy.py $CKPT --episodes 5 --stochastic --demand-split test
```
**校验**：B2 约 **6.7±1.9** 接单（随机采样，允许 ±2 波动）；H5′ 约 **7.8±1.3**。
反向校验：去掉 `--stochastic` → 应看到 0 接单、reward ≈ −45.6（argmax 退化，§4）。

非学习基线对照表：
```bash
./.venv/bin/python scripts/run_baselines.py --areas central_park
# 期望: random 0 / nearest 1 / sumo_greedy 32
```

### 第 4 步：主扫描（第一、二幕的数据；约 2 小时/个）

```bash
./.venv/bin/python scripts/sweep_severity.py $CKPT \
  --episodes 3 --seeds 42 43 44 45 46 47 48 49 --faithfulness-every 8 \
  --out runs/sweeps/my_B2_ladder
```
（默认即冻结机制 + max-AoI 阶梯 + 测试需求。H5′ 同命令换 checkpoint。）
每格打印一行，可中断续跑（已存在的格子自动跳过）。

### 第 5 步：假设检验

```bash
./.venv/bin/python scripts/analyze_hypotheses.py runs/sweeps/my_B2_ladder
```
**校验目标**（B2，允许小幅波动；精确参考值在
`results/story_freeze_v1/B2_sweep/analysis.json`）：

| 检验 | 期望结论 | 参考统计量 |
|---|---|---|
| H1 / H1m（DEF 随严重度降） | 不显著（地板效应） | ρ≈+0.00, p>0.5 |
| H2 / H2m（忠实度先于性能衰减） | 显著 | p≈0.02–0.03, n=32 |
| H3（WAMSN 随严重度升） | 显著 | ρ≈+0.09, p=0.0001 |
| H4（WAMSN–DEF 负相关） | 显著 | ρ≈−0.14, p=0.0001 |

表格部分：margin-DEF 各级 ≈ **−0.54**（H5′ ≈ **−0.77**）、pickups 平坦
6.5–7.4、WAMSN 从 0 跳到 ~0.013–0.017、经验退化暴露率 ≈ 2%。

### 第 6 步：解耦头（第三幕 b）

```bash
# 蒸馏（~15 分钟；或直接用 results/b1b2b3_sumo120_seed42_v1/explainer/explainer_head.pt）
./.venv/bin/python scripts/distill_explainer.py $CKPT
# 配对对比（clean 和 tunnel 各一次）
./.venv/bin/python scripts/eval_explainer.py $CKPT --episodes 2 --every 5
./.venv/bin/python scripts/eval_explainer.py $CKPT --episodes 2 --every 5 \
  --degradation tunnel_triggered --out compare_tunnel.json
```
**校验**：蒸馏报告 val Spearman ≈ **+0.60±0.33**；对比输出
`def_m: coupled ≈ −0.50, decoupled ≈ −0.39, Δ ≈ +0.11, p=0.0001`（两个条件都如此）。

### 第 7 步：出四张主图

```bash
./.venv/bin/python scripts/plot_story.py \
  --b2-sweep runs/sweeps/my_B2_ladder --h5-sweep runs/sweeps/my_H5_ladder \
  --compare <clean对比.json> --compare-tunnel <tunnel对比.json> \
  --out-dir runs/figs/story
```
和 `results/story_freeze_v1/figs/` 里的成品逐张对照。

---

## 6. 校验清单（每个论文数字 ↔ 文件位置)

| 论文数字 | 出处文件 |
|---|---|
| Act 1: clean margin-DEF −0.541 [−0.556, −0.527], n=3565 | `story_freeze_v1/B2_sweep/analysis.json` + 图 act1 |
| Act 2 表（DEF/pickups/WAMSN × 5 级） | 同上 analysis.json 的 descriptive 部分 |
| H1–H4 检验值 | 同上（H5′ 在 `H5b_sweep/analysis.json`） |
| H5′ 更差：−0.77 vs −0.54 | 两个 analysis.json 对照；图 act3a |
| 解耦头 +0.107/+0.113, p=0.0001 | `story_freeze_v1/explainer/compare_freeze_*.json` |
| 蒸馏质量 Spearman +0.60 | `b1b2b3_sumo120_seed42_v1/explainer/explainer_head.report.json` |
| B0 贪心 32 / random 0 / nearest 1 | `b1b2b3_sumo120_seed42_v1/baselines_sumo120.json` |
| 熵坍缩曲线 | 各训练目录 `train_log.jsonl`（`entropy` 字段）；图 `B2_convergence.png` |
| 与提案的全部偏离 | `docs/DEVIATIONS.md`（9 条） |

**随机性说明**：环境种子、需求文件、随机基线子集都被种子固定；唯一不固定的
是策略的动作采样，所以逐回合接单数会波动，但均值应落在报告的 ±std 内，
统计检验的结论（显著/不显著方向）应稳定复现。

## 7. 常见坑

1. **argmax 评估全是 0 接单** —— 不是 bug，见 §4，加 `--stochastic`。
2. **训练后期接单归零** —— 熵坍缩，用 `ckpt_best.pt` 而非最后一个快照。
3. **概率版 DEF 全是 ±0.002** —— 饱和效应，看 `def_m`（margin 版）。
4. **WAMSN 数值很小（~0.01）** —— 隧道只覆盖 ~2% 决策暴露，是地理事实，
   不是错误；重点是 0 → 0.013 的跳变和单调性（H3 的 ρ 用秩，不受量级影响）。
5. **旧 keeper 的 H5 逆转结论** —— 噪声机制伪象，冻结机制下不成立，勿引用。
6. **本机不能 `build_yubei.py`** —— netconvert 缺 GDAL；场景已提交，无需重建。
