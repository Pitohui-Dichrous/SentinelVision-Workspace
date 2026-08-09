# PPE 时序安全管线配置

## 配置分层

可审查的算法默认值位于：

```text
config/safety_pipeline.yaml
```

当前电脑的用户选择位于：

```text
.runtime/config/sentinel_settings.json
```

前者进入 Git，后者属于可重建运行状态并被 `.gitignore` 排除。两者的 schema 相互独立：算法 YAML 当前是 schema 3，用户设置在保存分析选项后也是 schema 3，但字段含义不同。算法代码不包含盘符路径。

## 运行模式与兼容性

`config/safety_pipeline.yaml` 的 `default_mode` 当前为 `baseline`。旧的用户设置 schema 1/2 没有 `safety_pipeline` 字段时仍进入兼容基线，不会静默改变已部署系统的报警行为。

用户在检测 Inspector 中明确选择增强模式后，设置升级为 schema 3，例如：

```json
{
  "schema_version": 3,
  "selected_model_ids": ["combined"],
  "class_enabled": {
    "fire": true,
    "person": true,
    "hat": true
  },
  "safety_pipeline": {
    "mode": "ppe_temporal",
    "debug_overlay": false
  }
}
```

保存时会保留未知字段，避免未来模块的设置被旧代码无意删除。可用模式只有：

- `baseline`：原有同类别 tracker、30/15 滑窗和按来源/类别冷却，只供回退和实验对照；
- `ppe_temporal`：UI 显示为“生产时序防护”。PPE 使用冲突消解、预测跟踪、时序证据和每 Track 风险状态；Fire 与其他告警类别使用通用按对象真实证据门。

## 参数说明

下列默认值只是实验起点，不是行业标准、法规阈值或安全认证结论。修改后必须使用固定录像和人工标注重新评估。

### `ppe.conflict`

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `protected_class_id` | `hat` | 已戴帽 canonical ID |
| `unprotected_class_id` | `person` | 未戴帽 canonical ID |
| `uncertain_class_id` | `ppe_uncertain` | 逻辑待确认 ID，不是模型输出 |
| `min_iou` | 0.60 | 跨类别框允许合并的最小 IoU |
| `score_margin` | 0.20 | 两个分数判定胜者的最小差值 |
| `max_center_distance_ratio` | 0.35 | 中心距离相对框尺度上限 |
| `min_area_ratio` | 0.50 | 小框/大框面积比下限 |

分差边界采用严格语义：`abs(helmet_score - nohelmet_score) > score_margin` 时选择高分状态，否则为 `UNCERTAIN`。因此默认值下 0.95 与 0.75 正好位于 0.20 边界，结果为 UNCERTAIN；0.96 与 0.75 才确定为 HELMET。边界已写入单元测试，避免浮点误差暗中决定结果。

### `ppe.tracking`

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `iou_threshold` | 0.30 | 逻辑头部轨迹匹配阈值；与 YOLO NMS IoU 分离 |
| `min_hits` | 2 | Track 进入风险判断前的最少命中 |
| `smoothing_alpha` | 0.70 | 新框在平滑框中的权重 |
| `loss_tolerance.value` | 1.00 s | 已确认 Track 的最长丢失保留时间 |
| `tentative_loss_tolerance.value` | 0.30 s | 未确认候选的最长丢失保留时间 |
| `admission.association_min_confidence` | 0.20 | 可用于续接已公开 Track 的最低置信度 |
| `admission.new_candidate_min_confidence` | 0.45 | 新建候选和形成证据的最低置信度 |
| `admission.max_tentative_candidates` | 128 | 未确认候选容量；满载时只拒绝新候选，不驱逐已确认 Track |
| `publication.min_duration_seconds` | 0.20 s | 分配公开 Track ID 前的最短观测时间 |
| `publication.min_ema_confidence` | 0.50 | 分配公开 Track ID 前的最低置信 EMA |
| `publication.ema_alpha` | 0.35 | 候选置信 EMA 的当前样本权重 |
| `reacquisition.enabled` | true | 是否对严格匹配失败的已确认 Track 尝试短时重关联 |
| `reacquisition.min_iou` | 0.10 | 预测框重关联的最低 IoU |
| `reacquisition.max_center_distance_ratio` | 0.80 | 中心距离相对框尺度上限 |
| `reacquisition.min_area_ratio` | 0.65 | 新旧框最小面积比例 |
| `reacquisition.ambiguity_margin` | 0.10 | 最优与次优候选不足该差值时拒绝继承 |
| `reacquisition.motion_max_seconds` | 0.85 s | 运动预测允许外推的最长时间 |
| `reacquisition.motion_model` | `alpha_beta` | schema 3 必需的有界中心运动模型；关闭宽松重捕获请用 `reacquisition.enabled: false` |
| `reacquisition.position_gain` | 0.75 | 观测残差校正中心位置的比例 |
| `reacquisition.velocity_gain` | 0.20 | 观测残差校正速度的比例 |
| `reacquisition.velocity_decay_per_second` | 0.35 | 丢失期间速度衰减率 |
| `reacquisition.min_velocity_dt_seconds` | 0.02 s | 防止极小时间差制造异常速度的下限 |
| `reacquisition.max_center_speed_ratio` | 8.0 | 每秒最大中心速度相对目标尺度 |
| `public_id_policy` | `confirmed_only` | 仅确认后分配公开 Track 编号 |

schema 3 在匹配前先预测全部活动 Track 的中心，严格和宽松阶段都使用预测状态。尺寸只做 EMA、不外推速度；中心速度具有最小 `dt`、尺度相关上限和时间衰减。经历检测间隙的匹配还必须通过双向唯一性差值；多人关系含糊时拒绝继承并创建私有候选。Resolver 使用不同目标：先最大化可消解冲突对数量，再比较总权重。

内部 `candidate_id` 不会进入普通 UI、告警或导出。候选达到 `min_hits`、发布最短时间和置信 EMA 后才获得会话内连续、单调且不复用的公开 `track_id`；一帧误检不会显示或消耗编号。schema 3 强制 `max_tentative_candidates >= 1`，显式配置为 0 会校验失败；只有 schema 1/2 的内部兼容默认允许无限值。低置信检测只更新已公开 Track 的几何关联，不增加 Temporal/Risk 证据。公开编号必须与 `session_id` 组合使用。

生产模式采用真正的双阈值：YOLO 推理保留到 `association_min_confidence`，避免低分框在进入 tracker 前就被丢弃；Inspector 的 confidence 与 `new_candidate_min_confidence` 取较高者，作为新建候选和有效证据的门槛。因此提高 UI 阈值会更严格，但不会切断已公开 Track 在关联下限以上的短时低置信续接。兼容基线仍把 UI confidence 直接传给旧推理路径，行为不变。修改 confidence 或 NMS IoU 会清空生产模式下的旧轨迹与证据，避免一个 episode 混用两套判定策略。

算法 schema 1 仍使用帧制生命周期、关闭重关联并立即分配公开 ID；schema 2 保留此前的秒制生命周期和旧速度 EMA。schema 3 强制使用 Alpha-Beta 核心以保证高低置信门禁、容量与发布语义不会被旧 tracker 绕过；可用 `reacquisition.enabled: false` 关闭宽松重捕获，或显式切换兼容基线完成整链回退。schema 3 才启用候选门、秒级风险资格和通用告警证据门。schema 1/2 的旧配置指纹保持不变，便于复现实验。`DETECT_EVERY_N` 不改变秒制单位，但会改变证据密度，正式实验必须固定推理采样率。

### `ppe.temporal`

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `window_size` | 8 | 每 Track 证据窗口长度 |
| `min_votes` | 6 | HELMET 或 NO_HELMET 成为候选稳定状态的最少票数 |
| `ema_alpha` | 0.35 | 当前分数进入 EMA 的权重 |
| `score_margin` | 0.12 | 两类 EMA 形成稳定候选的最小差值 |
| `min_stable_frames` | 1 | UNKNOWN/UNCERTAIN 接受候选前的连续帧数 |
| `recovery_frames` | 3 | 已稳定状态接受相反候选前的滞回帧数 |
| `missing_tolerance` | 3 | 丢失后仍保留稳定状态的帧数 |

`min_votes` 不得大于 `window_size`。缺帧不参与有效票数，超过 `missing_tolerance` 后稳定状态回到 UNKNOWN。

### `ppe.risk`

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `suspect_frames` | 1 | 稳定未戴帽进入 SUSPECT 前的帧数 |
| `confirm_frames` | 2 | SUSPECT 后进入确认的持续帧数 |
| `recovery_frames` | 4 | 稳定已戴帽完成恢复的持续帧数 |
| `cooldown_seconds` | 10.0 | 每 Track 告警后的冷却状态时长 |
| `verification.min_violation_seconds` | 0.80 s | 进入 CONFIRMED 前连续合格违规证据的最短时间 |
| `verification.min_stable_confidence` | 0.50 | 当前支持分与稳定 EMA 的共同下限 |
| `verification.min_stability` | 0.65 | 稳定状态在有效窗口中的最低比例 |
| `verification.max_evidence_gap_seconds` | 0.35 s | 超过后清空连续资格进度，但不删除 Event/冷却 |

同一 Track 的同一违规 episode 只生成一个 Event ID。冷却到期后状态可回到 ALARMED，但不会为未恢复的同一 episode 制造重复 Event；恢复完成后再次违规会创建新 Event，但同一 Track 的上次告警时间仍保留，新 Event 若落在 10 秒窗口内会先进入 COOLDOWN，窗口届满且违规仍持续时才告警。另一 Track 不受该冷却影响。

风险只有在“本帧是实际检测、raw 与 stable 一致、当前类别支持分达标、稳定 EMA 达标、稳定度达标”时推进。低置信关联、预测位置、缺帧和冲突帧都不会推进违规，也不会错误完成恢复。

### `alert_validation.defaults`

该门用于生产模式中的 Fire 和其他 `alert_enabled` 类别。不同 `(source, class, track)` 证据完全隔离。

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `min_high_confidence_hits` | 4 | 确认前最少真实高置信命中 |
| `min_duration_seconds` | 0.60 s | 命中跨度下限 |
| `window_seconds` | 1.20 s | 统计窗口 |
| `max_gap_seconds` | 0.25 s | 超过后重启候选 episode |
| `min_presence_ratio` | 0.60 | 真实命中占观测机会的最低比例 |
| `min_ema_confidence` | 0.45 | 置信 EMA 下限 |
| `cooldown_seconds` | 10.0 s | 每对象独立冷却 |
| `max_active_events` | 128 | 活动证据记录容量；不会驱逐已确认/冷却事件 |

missing、predicted、低置信或未确认观测贡献 0 个正证据。该通用门已经阻止单帧反光或闪框直接报警，但不等同于完整的火焰形变与空间簇专用 `FireTemporalVerifier`。

## 校验和失败策略

- 算法 schema 1/2/3、类型、数值范围、时间单位和类别 ID 会在启动时校验；
- 配置文件缺失、YAML 错误或字段越界时记录 warning、回退内置参数并锁定兼容基线；此前保存的增强模式不会覆盖该 fail-closed 状态，修复配置后重启才可重新选择；
- 旧程序不认识算法 schema 3 时按既有 fail-closed 规则回到基线，不会静默忽略新门限；
- UI 只暴露模式和调试显示，算法阈值不堆在主页面；
- 分析模式、视频源或模型变化会清空全部相关时序状态；只改变 PPE 类别开关时仅清空 PPE 状态，不抹除 Fire 的证据与冷却；
- 每个增强事件保存配置 SHA-256 指纹，便于实验复现。

实时流使用单调时钟。本地录像优先使用有效且前进的 PTS；若后端返回恒 0、NaN、回退时间或无效 FPS，系统按真实解码帧索引与 FPS（最终回退为受校验的 30 FPS）生成确定性时间。

## 审计日志

以下低频记录追加到 `.runtime/events/ppe_events.jsonl`：

- 风险状态转换；
- 告警事件；
- 人工“有效/误报”结论。

生产记录使用 `session_id + track_id + event_id` 串联，并附带有效证据帧、持续时间、EMA、稳定度和判定依据；`candidate_id` 永不写入普通事件记录。日志不会逐帧保存画面或 detection，避免 24 小时运行时无界增长。完整逐帧实验 trace 应在后续实验工具中显式开启，并继续写入 `.runtime/experiments`。

## 测试命令

```powershell
& '.\RUNTIME\python\python.exe' -I -B -X utf8 -m unittest discover -s tests -p 'test_*.py' -v
```

测试不依赖 pytest，也不加载生产模型或启动训练。
