# LogicMD SCI-VisNet 10-case evaluation

## 结论

这批 LogicMD 轨迹在前约 100 ns 的 quick 视图下尚有部分短程结构信号，但 full
500-frame（声明时间轴 0–499 ns）评估显示明显的长程退化：骨架运动放大、配体驻留
下降、蛋白–配体接触丢失。因而 quick 适合做快速筛查，不能代替 full。

10 个 case 的 quick/full 均成功完成，无评估失败。相同主机、相同环境的完整复跑中，
所有纳入核验的 CSV 输出逐值完全一致，最大绝对数值差为 0。

## 数据与抽样

- 预测：LogicMD SCI-VisNet，checkpoint
  `epoch1_step40000.ckpt`，`inf_step=499`，`ODE=10`。
- 参考：对应 PDB 的原始 MD 轨迹，使用与预测一致的 sliced topology/重原子集合。
- 时间轴：按 LogicMD lig10/SCI 协议声明的 `dt=1 ns`；没有混入约 80 ps 时钟的
  MISATO zero-shot 数据。
- 抽样：从固定 test50 表按参考配体运动分层 Q1–Q5，每层抽 2 条；先按 PDB 排序，
  Python PRNG seed `20260903`，共 10 条。
- quick：每条前 100 帧；full：每条全部 500 帧。每个 case 均为 1 条预测 replica
  对 1 条参考轨迹。

## 总体效果（10-case 中位数）

| 指标 | quick | full | 解释 |
|---|---:|---:|---|
| 物理有效帧率 | 0.700 | 0.615 | full 更差；部分 case 很早即出现几何异常 |
| 时间配对骨架 RMSD (Å) | 2.619 | 5.276 | 对原 MD 的结构偏离随长程明显增大 |
| 时间配对配体 RMSD (Å) | 3.665 | 8.217 | 配体长程偏离明显 |
| 骨架 overmove 比 | 1.384 | 2.954 | full 的生成运动幅度约为参考的 3 倍 |
| 配体 stay rate | 0.895 | 0.228 | quick 显著高估长程驻留 |
| 骨架 RMSF Pearson | 0.620 | 0.375 | 空间柔性模式一致性下降 |
| RMSF 幅度比 | 1.520 | 3.462 | full 中波动幅度过大 |
| 接触频率 Pearson | 0.325 | 0.171 | 蛋白–配体接触频率一致性偏弱 |
| 接触频率 F1 | 0.462 | 0.195 | full 接触保持显著变差 |
| PCA Gaussian W2 | 31.370 | 29.321 | 单项略降，不代表总体改善 |
| PCA1 ACF RMSE | — | 0.264 | full 的时间相关性诊断 |

## 逐条记录

表中箭头为 quick → full；`valid` 是物理有效帧率，`overmove` 是生成/参考骨架
运动幅度比，`stay` 是配体驻留率，`contact F1` 是接触频率 F1。`推理耗时`
来自内部既有推理运行的计时文件，并非本次重跑生成。

| case | 分层 | 推理耗时 (s) | valid | overmove | stay | contact F1 |
|---|---|---:|---:|---:|---:|---:|
| 2A14 | Q1 tight | 405.50 | .950 → .802 | 2.70 → 5.19 | .960 → .230 | .481 → .249 |
| 3L3M | Q1 tight | 541.10 | .300 → .150 | 1.30 → 2.74 | .990 → .226 | .572 → .338 |
| 2HMV | Q2 stable | 434.87 | .770 → .722 | 1.52 → 3.77 | .540 → .108 | .457 → .136 |
| 2JGS | Q2 stable | 400.97 | .000 → .000 | .48 → 1.28 | 1.000 → .348 | .467 → .177 |
| 5FSC | Q3 mid | 345.63 | 1.000 → .958 | 1.76 → 2.97 | .960 → .504 | .569 → .468 |
| 6ABX | Q3 mid | 1123.47 | .010 → .002 | 1.22 → 2.81 | 1.000 → .534 | .352 → .213 |
| 3TWP | Q4 mobile | 491.19 | .970 → .924 | 1.47 → 3.03 | .380 → .266 | .307 → .073 |
| 4K6Z | Q4 mobile | 468.37 | .990 → .934 | 1.51 → 3.08 | .830 → .166 | .627 → .346 |
| 1M2Q | Q5 loose | 536.44 | .630 → .508 | 1.30 → 2.94 | .650 → .130 | .382 → .059 |
| 2UYQ | Q5 loose | 460.84 | .500 → .440 | 1.24 → 2.87 | .420 → .084 | .277 → .095 |

重点异常：

- `2JGS` 的物理有效帧率 quick/full 都为 0；full bond relative MAE 为 0.909，
  属于明确的几何失效。
- `6ABX` 的有效帧率为 1.0%/0.2%，也不是单纯的长程漂移问题。
- `5FSC` 是这 10 条中相对最稳的一条，但 full 的 overmove 仍为 2.97。
- Q5 的 `1M2Q`、`2UYQ` 在 full 中接触频率 Pearson 分别为 -0.113、-0.007。

## 长程窗口趋势（10-case 中位数）

| 窗口末端 (ns) | valid | 预测骨架 RMSD0 (Å) | 预测配体 RMSD0 (Å) | overmove |
|---:|---:|---:|---:|---:|
| 10 | 0.591 | 0.868 | 1.124 | 0.727 |
| 50 | 0.627 | 1.625 | 2.438 | 1.125 |
| 100 | 0.708 | 2.212 | 3.224 | 1.388 |
| 250 | 0.723 | 3.486 | 5.717 | 2.079 |
| 499 | 0.615 | 5.122 | 8.196 | 2.954 |

骨架 overmove 的中位数在约 50 ns 已超过 1，随后持续放大；配体 RMSD0 同样随窗口
稳定上升。这是 full 比 quick 更能揭示的主要问题。

## 耗时与资源

| profile | 首轮 wall | 复跑 wall | 首轮最大 RSS | 复跑最大 RSS | 状态 |
|---|---:|---:|---:|---:|---|
| quick | 20.74 s | 20.96 s | 403,520 KB | 416,116 KB | 10/10，exit 0 |
| full | 48.01 s | 48.32 s | 1,025,192 KB | 1,043,532 KB | 10/10，exit 0 |

full 相对 quick 的首轮 wall time 为 2.32×，峰值内存约 2.54×。10 条既有 LogicMD
推理生成记录合计 5,208.38 s（86.81 min），中位数 464.60 s/条，范围
345.63–1,123.47 s/条。

## 可复现性

- quick 与 full 均按相同协议完整复跑一次。
- 比较了 `per_replica.csv`、`per_case.csv`、`summary_by_model.csv`、
  `metrics_long.csv`、`frame_metrics.csv`、`motion_curves.csv`、
  `failures.csv`，full 额外比较 `window_metrics.csv`。
- 上述文件的解析后内容全部完全一致，最大绝对数值差为 0；审计结果见
  `reproducibility.json`。
- 已记录协议、抽样表、输入轨迹、checkpoint、推理配置、推理计时表和评估源码的
  SHA-256；环境记录见 `environment.txt`。
- 局限：这是同机同软件环境复现，尚未验证跨硬件/跨 BLAS 的 bitwise 一致性；当前
  评估 Python 环境未安装 pytest，因此没有在本次环境重新执行包级单元测试，但两份
  协议均通过 CLI validate，且两轮评估均 exit 0。
- 每个 case 只有单预测 replica 和单参考轨迹，本结果适合工程筛查，不应据此给出
  publication-grade 的动力学速率或不确定性结论。

## 复跑

```bash
cd /home/xincheng/xincheng-test
./complexmd-eval/evaluations/logicmd_sci_visnet_10_20260903/run.sh
./complexmd-eval/evaluations/logicmd_sci_visnet_10_20260903/replay.sh
```
