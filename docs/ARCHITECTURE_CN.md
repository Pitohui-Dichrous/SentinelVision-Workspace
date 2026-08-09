# SentinelVision 时序安全管线架构

## 目标与当前范围

本阶段不修改 YOLOv5 网络和生产模型，而是在相同检测输出之上增加可关闭的 PPE 时序安全管线。目标是把“每帧画框”升级为“同一目标的连续风险状态”，同时保留原有基线用于回归和实验对照。

当前已经实现的研究纵向切片：

- `hat` / `person` 互斥类别冲突消解；
- 与 PPE 类别无关的逻辑头部跟踪、短时运动辅助重关联；
- 内部候选 ID 与确认后公开 Track ID 分层；
- N-of-M、EMA、短暂丢失容忍和状态滞回；
- 每个 Track 独立的 PPE 风险状态机；
- 每个违规 episode 独立 Event ID 和冷却状态；
- 结构化状态转换、告警和人工复核日志；
- 兼容基线 / PPE 时序增强显式切换。

本阶段没有宣称完成 Phase 6 Fire 专用时序确认、Phase 7 动态风险等级、Phase 8 Hard Example Collector、Phase 9 System Watchdog、Phase 11 自动训练优化或 Phase 12 Dynamic ROI。Phase 10 目前只有指标字段、事件日志和实验方法定义，自动 detection trace、Baseline / Proposed 双重放、CSV / JSON 汇总与可读报告生成器也仍待后续实现。

## 两条可切换的数据流

### 兼容基线

```text
视频源
  → YOLOv5 AutoShape / class-aware NMS
  → canonical 类别映射
  → 同类别跨模型 NMS
  → 原始框 Mask / ROI 空间过滤
  → 类别开关
  → 旧轻量同类别 IoU Tracker
  → 每 (source, class) 30 帧窗口、15 次命中
  → 每 (source, class) 10 秒冷却
  → 告警
```

该路径保留现有行为，供日常回退和 Baseline 实验使用。

### PPE 时序增强

```text
视频源
  → YOLOv5 AutoShape / class-aware NMS
  → canonical 类别映射
  → 同类别跨模型 NMS
  → 原始框 Mask / ROI 空间过滤
  → PPEConflictResolver
  ├─ PPE HeadObservation
  │    → HeadTracker 严格 IoU 关联
  │    → 未匹配的已确认轨迹短时重关联
  │    → 候选确认 / 分配公开 Track ID
  │    → TemporalEvidenceEngine
  │    → PPERiskStateMachine
  │    → 稳定类别显示开关
  │    → Track/Event 告警与稳定绘框
  └─ Fire / 其他类别
       → 类别开关
       → 兼容基线路径
```

Mask / ROI 是物理监控域，所以先对每个原始框做空间过滤，再进行冲突消解。这样域外框不会把域内违规框的融合中心拖出边界。`hat` / `person` 开关则是显示策略：只要其中任一项启用，空间域内的完整互斥证据都会进入 resolver 与时序引擎，最终才按稳定类别决定是否绘制；未戴帽告警还会独立服从 `person` 的告警配置。增强模式只接管 PPE，减少对现有 Fire 路径的回归风险。

## 真实类别语义

历史 checkpoint 的 canonical ID 必须按生产注册表解释，不能按英文单词猜测：

| canonical ID | 业务语义 | 是否告警 |
| --- | --- | --- |
| `person` | 未戴安全帽 | 是，warning |
| `hat` | 已戴安全帽 | 否 |
| `fire` | 火焰 | 是，critical |
| `ppe_uncertain` | 时序管线的逻辑待确认状态，不是模型类别 | 否 |

## 纯算法模块

`safety_pipeline` 不依赖 PySide6、Torch、OpenCV 或网络，可对保存的检测序列进行确定性测试和重放。

| 模块 | 职责 |
| --- | --- |
| `config.py` | 版本化默认值、范围校验、配置指纹、失败回退 |
| `spatial.py` | 原始框级 ROI / Mask 监控域过滤 |
| `policy.py` | PPE 证据门控与稳定类别显示策略 |
| `matching.py` | 无第三方依赖的确定性二分匹配 |
| `ppe_conflict.py` | 一对一跨类别配对、几何约束、分数判定、保留 raw evidence |
| `tracking.py` | 类别无关的逻辑头部跟踪、速度估计、歧义拒绝、候选晋升与短暂丢失容忍 |
| `temporal.py` | N-of-M、EMA、稳定状态、missing tolerance、恢复滞回 |
| `risk.py` | NORMAL 到 ALARMED/COOLDOWN/RECOVERING 的每轨迹状态机 |
| `pipeline.py` | 组合跟踪、时序、风险、事件和累计指标 |
| `clock.py` | 实时单调时钟与离线视频 PTS / 帧索引回退 |
| `journal.py` | 低频 JSONL 状态转换、告警、人工复核审计记录 |

`VideoWorker` 仍负责采帧、模型推理、ROI、绘制和 Qt 信号；业务判断下沉到纯算法包，没有重写 YOLOv5 或生产部署链。

## PPE 冲突消解

1. 只比较配置中的 `hat` / `person` 互斥对。
2. 同时满足 IoU、中心距离和面积比约束才允许配对。
3. Resolver 采用“最大匹配数优先、总权重次优”的二分匹配；不使用最高 IoU 贪心，避免一个局部最优框阻断两个本可消解的冲突对。
4. 合并后的 `HeadObservation` 保留两个分数、两个原始 detection、模型来源和逻辑框。
5. 分差严格超过配置边界时选择胜者；等于或低于边界时输出 `UNCERTAIN`。
6. UI 只画一个逻辑框，raw evidence 仍供时序与调试使用。

## 跟踪、时序与风险

- Tracker 按“head”匹配，不把 `hat` / `person` 当成不同实体，所以类别抖动不会自动换 Track ID。
- 第一阶段关联允许目标保持未匹配，并最大化总 IoU；它不会为了增加匹配数量而牺牲近乎完美的旧轨迹关联。
- 已经发生检测间隙的轨迹即使重新达到严格 IoU，也必须通过双向歧义差值；无法区分时不继承旧状态。严格关联仍失败时，才对短时丢失的已确认轨迹使用有界 EMA 速度预测，并继续检查宽松 IoU、中心距离、面积比例和歧义差值，避免旧告警状态绑到新人。
- 已确认 Track 和未确认候选分别使用秒制保留时间。丢失轨迹不会继续绘制；等于保留上限时仍保留，超过后才退休。
- `candidate_id` 是内部关联键，从首帧起承载 Temporal/Risk 证据；达到 `min_hits` 后才分配公开 `track_id`。单帧噪声死亡不会消耗公开编号，晋升也不会重置确认延迟、Event 或冷却。
- Temporal Engine 保存每轨迹最近 M 帧的 HELMET / NO_HELMET / UNCERTAIN / UNKNOWN 与分数，使用 N-of-M 和 EMA 形成稳定状态。
- 已稳定状态只有在持续反证达到恢复条件后才改变；单帧抖动不会立刻翻转。
- 未戴帽风险按每个 Track 独立经历 `NORMAL → SUSPECT → CONFIRMED → ALARMED → COOLDOWN → RECOVERING → NORMAL`。
- 同一个 episode 只创建一个 Event ID；另一名违规人员拥有不同 Track/Event，不受前一人的冷却压制。

所有时序判断使用调用方提供的单调时间。实时模式传入 `time.monotonic()`；本地录像优先使用单调前进的 PTS，遇到恒为 0、NaN、回退等无效 PTS 时按真实解码帧索引和 FPS 回退，因此视频时间不会随 GPU 墙钟推理速度漂移。N-of-M 和连续确认仍按采样到的推理帧计数，`DETECT_EVERY_N` / 推理采样率会改变证据密度和确认延迟，正式对照实验必须固定。

## UI 与事件契约

检测 Inspector 中只有两个新增控制：

- `分析模式`：兼容基线 / PPE 时序增强；
- `显示 raw / stable 调试信息`：只在增强模式可用。

普通画面只显示已确认目标的稳定类别与公开 `Track #`；未确认候选不得显示内部数字。调试模式额外显示 raw、stable 和 risk 状态。视频源或模型变化会清空全部旧轨迹与证据；分析模式或 PPE 类别开关变化只重置 PPE 管线，保留 Fire 的旧轨迹与冷却，避免无关设置导致重复火焰告警。

增强告警增加以下字段：

- `session_id`、`track_id`、`event_id`、`risk_type`；
- `stable_state`、`risk_state`；
- `confirmation_delay`；
- `pipeline_mode`、`config_fingerprint`。

告警详情和 CSV/JSON 导出保留这些字段。公开 Track 编号以 `session_id` 为命名空间；内部 `candidate_id` 不进入告警、人工复核或普通导出。低频审计记录写入 `.runtime/events/ppe_events.jsonl`，该目录在 Git 保护边界内，不上传模型画面或运行日志。

## 不变的安全边界

- `RESULTS` 仍只包含人工批准的生产模型；时序管线不会部署模型。
- 动态 `RESULTS` 扫描、模型默认不选中规则和模型冲突检查不变。
- 所有新路径均从工作库根目录计算，不绑定移动硬盘盘符。
- 配置缺失或旧设置没有新字段时进入兼容基线。
- 模式关闭后不经过 PPE resolver、HeadTracker 或风险状态机。
- 当前重关联是短时几何与运动预测，不包含外观 ReID。长时间遮挡、多人完全交叉或新人进入旧位置时无法保证身份恢复；关系含糊时优先新建候选，不继承旧风险状态。
