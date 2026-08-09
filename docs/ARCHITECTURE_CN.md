# SentinelVision 时序安全管线架构

## 目标与当前范围

本阶段不修改 YOLOv5 网络和生产模型，而是在相同检测输出之上增加可关闭的生产时序安全管线。目标是把“每帧画框”升级为“真实证据确认 + 同一目标连续风险状态”，同时保留原有基线用于回归和实验对照。

当前已经实现的研究纵向切片：

- `hat` / `person` 互斥类别冲突消解；
- 与 PPE 类别无关的逻辑头部跟踪、Alpha-Beta 中心运动预测与短时重关联；
- 内部候选 ID 与确认后公开 Track ID 分层；
- N-of-M、EMA、短暂丢失容忍和状态滞回；
- 每个 Track 独立的 PPE 风险状态机；
- 每个违规 episode 独立 Event ID 和冷却状态；
- 结构化状态转换、告警和人工复核日志；
- 低置信关联、候选容量和公开编号发布门；
- Fire / 未来告警类别的按对象真实证据门；
- 实验兼容基线 / 生产时序防护显式切换。

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

### 生产时序防护

```text
视频源
  → YOLOv5 AutoShape / class-aware NMS
  → canonical 类别映射
  → 同类别跨模型 NMS
  → 原始框 Mask / ROI 空间过滤
  → PPEConflictResolver
  ├─ PPE HeadObservation
  │    → HeadTracker 统一预测状态关联
  │    → 高置信建轨 / 低置信仅续已公开轨迹
  │    → 候选命中、持续时间、置信 EMA 发布门
  │    → TemporalEvidenceEngine
  │    → 秒级证据资格门 / PPERiskStateMachine
  │    → 稳定类别显示开关
  │    → Track/Event 告警与稳定绘框
  └─ Fire / 其他类别
       → 类别开关
       → 同类别对象跟踪
       → AlertEvidenceGate（真实命中/持续时间/出现比例/EMA）
       → 每对象独立冷却与告警
```

Mask / ROI 是物理监控域，所以先对每个原始框做空间过滤，再进行冲突消解。这样域外框不会把域内违规框的融合中心拖出边界。`hat` / `person` 开关是显示策略：只要其中任一项启用，空间域内的完整互斥证据都会进入 resolver 与时序引擎，最终才按稳定类别决定是否绘制；未戴帽告警仍独立服从 `person` 的告警配置。生产模式同时接管其他告警类别的证据验证，但不把预测框、丢失框或不同 Track 的检测合并为告警证据。

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
| `alert_validation.py` | 非 PPE 告警类别的按对象真实证据验证与冷却 |
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
- schema 3 在每次关联前先用带速度衰减、最小 `dt` 和速度夹限的 Alpha-Beta 状态预测所有 Track；只预测中心，宽高只做 EMA，避免框尺寸抖动把预测框放大。严格与宽松关联都使用预测状态，并继续检查 IoU、中心距离、面积比例和双向歧义差值。
- 高置信检测才能新建候选和形成告警证据；低置信检测只能在几何门内续接已公开 Track。生产模式让模型保留到独立 association floor，再以 UI confidence 与安全配置下限的较高者判定高置信，避免上游提前截断低分续跟证据。预测位置只用于关联，绝不作为 detection、时序票数或告警证据。
- 已确认 Track 和未确认候选分别使用秒制保留时间。丢失轨迹不会继续绘制；等于保留上限时仍保留，超过后才退休。
- `candidate_id` 是内部关联键；候选同时达到 `min_hits`、最短持续时间和置信 EMA 后才分配公开 `track_id`。单帧噪声不会显示或消耗公开编号，晋升也不会重置确认延迟、Event 或冷却。
- Temporal Engine 保存每轨迹最近 M 帧的 HELMET / NO_HELMET / UNCERTAIN / UNKNOWN 与分数，使用 N-of-M 和 EMA 形成稳定状态。
- 已稳定状态只有在持续反证达到恢复条件后才改变；单帧抖动不会立刻翻转。
- 未戴帽风险只有当前 raw 与 stable 一致、当前支持分数、稳定 EMA、稳定度和最短持续秒数全部达标时才推进；缺失、预测、低置信或冲突帧不会推进违规或恢复。
- 风险按每个 Track 独立经历 `NORMAL → SUSPECT → CONFIRMED → ALARMED → COOLDOWN → RECOVERING → NORMAL`。
- 同一个 episode 只创建一个 Event ID；另一名违规人员拥有不同 Track/Event，不受前一人的冷却压制。

所有时序判断使用调用方提供的单调时间。实时模式在 `cap.read()` 成功后、模型推理开始前立即记录捕获时间，避免推理耗时波动放大速度；本地录像优先使用单调前进的 PTS，遇到恒 0、NaN、回退等无效 PTS 时按真实解码帧索引和 FPS 回退。N-of-M 仍按采样到的推理帧计数，`DETECT_EVERY_N` / 推理采样率会改变证据密度和确认延迟，正式实验必须固定。

## UI 与事件契约

检测 Inspector 中只有两个新增控制：

- `分析模式`：实验兼容基线 / 生产时序防护（推荐）；
- `显示 raw / stable 调试信息`：只在增强模式可用。

普通画面只显示通过发布门的稳定 PPE 类别与公开 `Track #`；非 PPE 类别也只显示本帧真实、已确认的检测，丢失轨迹不再伪装成当前证据。调试模式额外显示 raw、stable 和 risk。视频源、模型或分析模式变化会清空全部相关轨迹与证据；单独切换 PPE 类别只重置 PPE 管线，不擦除 Fire 的冷却。

增强告警增加以下字段：

- `session_id`、`track_id`、`event_id`、`risk_type`；
- `stable_state`、`risk_state`；
- `confirmation_delay`；
- `evidence_hits`、`evidence_duration`、`evidence_ema`、`evidence_stability`、`decision_reason`；
- `pipeline_mode`、`config_fingerprint`。

告警详情和 CSV/JSON 导出保留这些字段。公开 Track 编号以 `session_id` 为命名空间；内部 `candidate_id` 不进入告警、人工复核或普通导出。低频审计记录写入 `.runtime/events/ppe_events.jsonl`，该目录在 Git 保护边界内，不上传模型画面或运行日志。

## 不变的安全边界

- `RESULTS` 仍只包含人工批准的生产模型；时序管线不会部署模型。
- 动态 `RESULTS` 扫描、模型默认不选中规则和模型冲突检查不变。
- 所有新路径均从工作库根目录计算，不绑定移动硬盘盘符。
- 配置缺失或旧设置没有新字段时进入兼容基线。
- 模式关闭后不经过 PPE resolver、HeadTracker 或风险状态机。
- 当前重关联是短时几何与运动预测，不包含外观 ReID。长时间遮挡、多人完全交叉或新人进入旧位置时无法保证身份恢复；关系含糊时优先新建候选，不继承旧风险状态。
