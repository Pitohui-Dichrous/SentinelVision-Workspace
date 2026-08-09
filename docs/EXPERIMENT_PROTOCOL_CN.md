# Baseline / Proposed 实验方法

## 研究命题

在使用相同 YOLOv5 权重、相同视频和相同原始检测结果的条件下，PPE 类别冲突消解、逻辑头部跟踪、多帧证据融合和每轨迹风险状态机能否降低冲突、闪烁、误报和重复告警，同时保持可接受的告警延迟。

实验重点是系统级稳定性，不把 mAP 的变化当作本阶段主要贡献。

## 对照组

### Baseline

```text
YOLOv5
+ confidence threshold
+ 现有同类别 IoU tracker
+ 每 (source, class) 30 帧中 15 次命中
+ 每 (source, class) 10 秒 cooldown
```

### Proposed

```text
同一 YOLOv5 detection
+ PPEConflictResolver
+ 类别无关 HeadTracker（严格关联 + 受限 Alpha-Beta 预测重捕获 + 歧义拒绝）
+ 高低置信双阈值；低置信结果只允许续接已公开目标
+ 连续高置信命中数 / 连续持续时间 / EMA 达标后才分配公开 Track 编号
+ N-of-M / EMA / missing tolerance / evidence duration / hysteresis
+ 每 Track 风险状态机
+ 每 episode Event ID
```

Fire 等非 PPE 告警类在生产模式下已有按 Track 的通用时序证据门禁，但不进入本阶段 PPE Proposed 主对比，避免把不同问题混入同一实验。专用 FireTemporalVerifier 仍需另设空间聚类、形变和持续时间实验。

## 控制变量

每组实验必须固定：

- 同一视频文件和相同起止帧；
- 同一个生产模型文件及 SHA-256；
- 相同 YOLO confidence、NMS IoU 和输入尺寸；
- 相同 `DETECT_EVERY_N`、解码 FPS 与推理采样率；
- 相同 ROI / Mask；
- 相同类别映射；
- 相同安全管线配置指纹；
- 相同算法 schema、重关联阈值和会话重置边界；
- 相同 ground-truth 事件区间。

最佳方法是“推理一次、检测 trace 双重放”：先保存 canonical detection，再把完全相同的 trace 分别送入 Baseline 和 Proposed。这样 GPU 速度、视频解码和模型随机性不会成为混杂变量。当前版本已经把纯算法层与 Qt/Torch 分离，并提供确定性单元测试；自动 trace 录制和双重放 CLI 是实验工具的下一阶段，不能在尚未实现时声称已经自动生成完整对比报告。

当前 UI 中手工把同一录像播放两次只能用于探索，不能作为最终受控实验：Proposed 本地文件路径使用 PTS / 帧索引视频时钟，而兼容基线仍沿用 wall time；两次运行也未保证得到逐项完全相同的 canonical detection。正式结果必须等待同 trace 双重放工具，或使用等价的离线评估器统一时间源。

## 测试视频与标注

建议至少建立以下固定场景：

1. 单人全程正确戴帽；
2. 单人全程未戴帽；
3. 戴帽/未戴帽分数短时冲突；
4. 遮挡造成 1–6 帧丢失；
5. 两人先后或同时进入，其中第二人违规；
6. ROI 边缘经过；
7. 夜间、模糊、逆光和特殊角度；
8. 从未戴帽恢复为戴帽，再次违规。
9. 静态背景物体产生 1–8 个高置信短闪误检；
10. 低置信框只能续接已有目标，不能单独形成候选或告警；
11. 匀速移动目标短暂丢失后，在预测窗口内回归；
12. 两人交叉、预测关系含糊、框尺寸抖动和异常短帧间隔。

ground truth 不能只标“这段视频有违规”，至少应包含：

```json
{
  "source_id": "ppe_case_001",
  "events": [
    {
      "event_id": "gt-001",
      "risk_type": "no_helmet",
      "start_time": 12.4,
      "end_time": 19.8,
      "subject_id": "person-a"
    }
  ]
}
```

没有事件起止和主体标注时，可以计算已报警中的误报，但不能可靠计算漏报率或每事件重复告警。

## 系统级指标定义

| 指标 | 定义 | 所需数据 |
| --- | --- | --- |
| False alarms / hour | 人工判定为误报的事件数 / 有效监控小时 | 告警、人工结论、有效时长 |
| Missed event rate | 未匹配任何系统 Event 的 ground-truth 事件 / 全部 ground-truth 事件 | 区间标注 |
| Average alarm latency | `alarmed_at - ground_truth.start_time` 的均值 | ground truth 与事件时间 |
| Duplicate alerts / event | 同一 ground-truth episode 匹配到的额外通知数 | 稳定 Event ID |
| PPE conflict rate | 高重叠互斥 raw pair / PPE raw observations | resolver 计数 |
| Detection flicker rate | 单位时间或单位 Track 的状态翻转次数 | raw/stable trace |
| Confirmed events | 风险状态机创建的 Event 数 | Event journal |
| Suppressed transients | SUSPECT 后恢复、未升级为 Event 的 episode 数 | 状态转换记录 |
| ID switch rate | 同一 ground-truth 主体错误变更公开 Track ID 的次数 / 有效轨迹时长 | 主体标注与 Track trace |
| Track fragmentation | 同一 ground-truth 主体被拆成的公开 Track 段数 | 主体标注与 Track trace |
| False inheritance rate | 新主体错误继承旧公开 Track / 风险状态的次数 | 主体标注与关联 trace |
| Reacquisition success rate | 短时丢失后正确继承旧 Track 的次数 / 可重关联案例 | 丢失区间与主体标注 |

当前 UI 已实时显示处理帧数、活动公开 Track、候选数、重捕获、歧义拒绝、过滤/拒绝数和管线 p95；事件导出包含 Session/Track/Event、稳定状态、风险状态、确认延迟与证据摘要；`.runtime/events/ppe_events.jsonl` 可追踪状态转换和人工结论。统计必须先按 `session_id` 分组，不能把不同会话中相同的裸 `track_id` 当成同一目标。在线候选退休数只能视为噪声代理指标，不能在没有人工标注时直接宣称为 false positive。

False alarm 需要人工复核；Missed event rate 必须依赖独立 ground truth。不得把“没有触发告警”自动当成正确负样本。

## 参数调节原则

1. 先固定验证集和评价脚本，再调阈值。
2. 不在同一录像上既调参又报告最终结果。
3. 一次只改变一个参数组：冲突、tracking、temporal 或 risk。
4. 报告默认值、搜索范围、最终值和配置指纹。
5. 同时报告误报下降与新增延迟，不能只展示有利指标。
6. 0.60、0.20、8 帧、6 票等都是调试起点，不应写成行业标准。

## 第一阶段验收

- 0.95 `hat` 与 0.75 `person` 高重叠时 UI 只产生一个 UNCERTAIN 逻辑头部，raw evidence 保留；
- 0.91 与 0.90 输出 UNCERTAIN，不触发未戴帽事件；
- 两个相邻头部不被错误合并；
- `hat ↔ person` 抖动仍保持同一 Track ID；
- 单人短时漏检且普通 IoU 不足、重关联约束满足时继承同一公开 Track ID；
- 超过保留时间后重现会获得新 Track ID；
- 多个旧 Track 对同一观测关系含糊时拒绝继承；
- 单帧及未满足连续发布门的高置信候选误检不显示也不消耗公开编号，首个稳定目标仍是 `Track #1`；
- 单帧或不足持续时间的高置信短闪不产生告警；持续低置信噪声既不能创建新候选，也不能推进风险证据；
- 缺失框、预测框和 coasting 状态只用于内部关联，不渲染为当前识别结果，也不计入发布命中数、发布持续时间或告警证据；
- 移动目标在预测窗口内回归且关系唯一时继承原 Track/Event/Cooldown，超窗、瞬移或多人歧义时拒绝继承；
- 候选洪泛受确定性容量上限约束，不驱逐已确认目标，并在调试统计中显式记录拒绝数；
- 在固定真实丢失时长下，不同 `DETECT_EVERY_N` 使用相同秒制退休边界，但证据采样率仍作为控制变量固定；
- 单帧错误不翻转稳定状态；
- Track 17 在冷却时，Track 26 可独立报警；
- 同一 episode 不重复创建 Event；
- 关闭增强模式后回到兼容基线；
- 单元测试、UI smoke、工作库验证和便携环境自检通过。
