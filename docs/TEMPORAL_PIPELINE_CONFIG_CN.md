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

前者进入 Git，后者属于可重建运行状态并被 `.gitignore` 排除。两者的 schema 相互独立：算法 YAML 当前是 schema 2，用户设置在保存分析选项后是 schema 3。算法代码不包含盘符路径。

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

- `baseline`：现有同类别 tracker、30/15 滑窗和按来源/类别冷却；
- `ppe_temporal`：PPE 使用冲突消解、逻辑头部 tracker、时序证据和每 Track 风险状态；其他类别仍走兼容路径。

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
| `tentative_loss_tolerance.value` | 0.40 s | 未确认候选的最长丢失保留时间 |
| `reacquisition.enabled` | true | 是否对严格匹配失败的已确认 Track 尝试短时重关联 |
| `reacquisition.min_iou` | 0.10 | 预测框重关联的最低 IoU |
| `reacquisition.max_center_distance_ratio` | 0.65 | 中心距离相对框尺度上限 |
| `reacquisition.min_area_ratio` | 0.65 | 新旧框最小面积比例 |
| `reacquisition.ambiguity_margin` | 0.10 | 最优与次优候选不足该差值时拒绝继承 |
| `reacquisition.motion_max_seconds` | 0.50 s | 速度预测允许外推的最长时间 |
| `reacquisition.velocity_alpha` | 0.65 | 当前观测速度进入 EMA 的权重 |
| `public_id_policy` | `confirmed_only` | 仅确认后分配公开 Track 编号 |

Tracker 先在普通 IoU 门限内求“允许不匹配的最大总 IoU”；轨迹一旦经历检测间隙，即使重新达到严格 IoU，也必须先通过唯一性差值。严格关联失败后，只对尚未匹配的已确认 Track 使用预测框尝试短时重关联。重关联必须同时满足 IoU、中心距离、面积比例和双向唯一性差值；多人关系含糊时拒绝继承并创建新候选。Resolver 使用不同目标：先最大化可消解冲突对数量，再比较总权重。

内部 `candidate_id` 从首帧开始积累时序证据，但不会进入普通 UI、告警或导出。候选达到 `min_hits` 后才获得会话内连续、单调且不复用的公开 `track_id`；因此一帧误检不会再消耗用户看到的编号。公开编号必须与 `session_id` 组合使用，不能跨会话仅按裸 `track_id` 汇总。

算法 schema 1 仍可读取：旧 `max_lost_frames` 按“未匹配的推理更新次数”计数，重关联关闭，公开 ID 立即分配，从而保留旧实验行为。schema 2 改为管线时间的秒制容忍；`DETECT_EVERY_N` 不再改变单位，但仍会改变实际可用于重关联和 N-of-M 的证据密度，正式实验必须固定推理采样率。

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

同一 Track 的同一违规 episode 只生成一个 Event ID。冷却到期后状态可回到 ALARMED，但不会为未恢复的同一 episode 制造重复 Event；恢复完成后再次违规会创建新 Event，但同一 Track 的上次告警时间仍保留，新 Event 若落在 10 秒窗口内会先进入 COOLDOWN，窗口届满且违规仍持续时才告警。另一 Track 不受该冷却影响。

## 校验和失败策略

- 算法 schema 1/2、类型、数值范围、时间单位和类别 ID 会在启动时校验；
- 配置文件缺失、YAML 错误或字段越界时记录 warning、回退内置参数并锁定兼容基线；此前保存的增强模式不会覆盖该 fail-closed 状态，修复配置后重启才可重新选择；
- 旧程序不认识算法 schema 2 时同样按既有 fail-closed 规则回到基线，不会静默忽略重关联字段；
- UI 只暴露模式和调试显示，算法阈值不堆在主页面；
- 模式或 PPE 类别开关变化只清空 PPE 轨迹、证据、状态机和 session；视频源或模型变化会清空全部时序状态；Fire 的旧冷却不会被无关 PPE 开关抹除；
- 每个增强事件保存配置 SHA-256 指纹，便于实验复现。

实时流使用单调时钟。本地录像优先使用有效且前进的 PTS；若后端返回恒 0、NaN、回退时间或无效 FPS，系统按真实解码帧索引与 FPS（最终回退为受校验的 30 FPS）生成确定性时间。

## 审计日志

以下低频记录追加到 `.runtime/events/ppe_events.jsonl`：

- 风险状态转换；
- 告警事件；
- 人工“有效/误报”结论。

增强记录使用 `session_id + track_id + event_id` 串联：`track_id` 只在单个 PPE 管线会话内有效，`candidate_id` 永不写入普通事件记录。它不会逐帧保存画面或 detection，避免 24 小时运行时无界增长。完整逐帧实验 trace 应在后续实验工具中显式开启，并继续写入 `.runtime/experiments`。

## 测试命令

```powershell
& '.\RUNTIME\python\python.exe' -I -B -X utf8 -m unittest discover -s tests -p 'test_*.py' -v
```

测试不依赖 pytest，也不加载生产模型或启动训练。
