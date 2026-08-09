# 硕士课程设计技术实现摘要

## 题目建议

**基于时空一致性与风险状态机的全天候工业视觉安全监测系统**

## 可直接用于报告的摘要

本项目在既有 YOLOv5 工业视觉检测系统基础上，针对安全帽互斥类别重叠、逐帧分类抖动、不同目标共享冷却时间以及告警缺少事件生命周期等问题，设计并实现了一套可配置、可回退的 PPE 时序安全分析管线。系统不改变原有检测网络和生产模型，而是在 canonical detection 后增加类别冲突消解、逻辑头部跟踪、多帧证据融合及每轨迹风险状态机。冲突消解器以 IoU、中心距离和面积比例作为几何约束，对 `hat` 与 `person` 进行一对一配对，同时保留两个原始检测分数和模型来源；得分差不足时输出 UNCERTAIN，避免任意选择。逻辑头部跟踪器不依赖当前 PPE 类别，因此短时分类翻转不会产生新的目标身份。时序证据引擎结合 N-of-M 投票、置信度指数移动平均、短暂丢失容忍和恢复滞回，输出 HELMET、NO_HELMET、UNCERTAIN 或 UNKNOWN 稳定状态。风险状态机按 Track 独立维护 NORMAL、SUSPECT、CONFIRMED、ALARMED、COOLDOWN 与 RECOVERING 状态，并为每次完整违规 episode 生成 Event ID，从而避免一名人员的冷却抑制另一名新进入人员的告警。

为保证工程可用性，新增算法被实现为不依赖 Qt、Torch 和 OpenCV 的纯 Python 模块，所有参数通过版本化 YAML 配置并进行范围校验；旧设置默认进入兼容基线，用户可以在 UI 中显式切换增强模式。ROI / Mask 先作用于原始框，防止域外冲突框改变融合框的监控域归属；PPE 类别开关只控制稳定结果显示，完整互斥证据仍用于时序判断。视频源或模型变化会清空全部旧时序状态，PPE 模式或类别变化只重置 PPE 管线并保留 Fire 冷却。事件详情与 CSV/JSON 导出增加 Track ID、Event ID、稳定状态、风险状态、确认延迟和配置指纹；风险状态转换、告警与人工复核以 JSONL 追加写入本地运行目录。本地录像使用 PTS 并在异常时按真实解码帧索引回退，以保证实验时间可重复。该实现保持 `RESULTS` 人工批准边界、动态模型扫描和移动硬盘离线运行能力不变。

后续实验将在完全相同的 YOLOv5 detection 上对比兼容基线与 Proposed 管线，重点评价每小时误报、漏报事件率、平均告警延迟、每事件重复告警、安全帽类别冲突率和状态闪烁率。由此可以验证：即使模型 mAP 不变，基于时空一致性和风险状态管理的后处理仍可提升 24 小时工业视觉监测的稳定性与实际可用性。

## 本阶段技术贡献

1. 将互斥类别从两个独立 detection 重构为保留双分数证据的 `HeadObservation`。
2. 实现 PPE 类别无关的一对一轻量跟踪，以允许不匹配的最大总 IoU 保持同一头部身份，避免拥挤画面中的基数优先 ID 交换。
3. 结合 N-of-M、EMA、missing tolerance 和 hysteresis 形成稳定 PPE 状态。
4. 建立每 Track / 每 episode 风险状态和事件身份，消除全局冷却造成的目标间干扰。
5. 建立兼容基线、配置指纹、结构化审计日志和可确定性测试，为对照实验提供工程基础。

## 研究边界与诚实说明

- 当前没有改变 YOLOv5 网络或声称提高 mAP。
- 当前增强管线只覆盖 PPE；Fire 专用时序确认仍待实现。
- False alarm 可以通过人工复核统计；Missed event 必须另有带起止区间的 ground truth。
- 当前参数是实验初值，尚未构成特定工厂的标定或安全认证。
- Phase 6 Fire Temporal、Phase 7 Dynamic Risk、Phase 8 Hard Example Collector、Phase 9 Watchdog、Phase 11 自动训练优化和 Phase 12 Dynamic ROI 属于后续阶段。
- Phase 10 当前只完成指标字段、事件日志与实验协议基础；自动 trace、同 detection 双重放、CSV / JSON 汇总和可读报告工具尚未实现。

## 代码与验证入口

- 算法：`safety_pipeline/`
- 默认配置：`config/safety_pipeline.yaml`
- UI / 实时接入：`SentinelVision4.py`
- 测试：`tests/`
- 架构：[ARCHITECTURE_CN.md](ARCHITECTURE_CN.md)
- 配置：[TEMPORAL_PIPELINE_CONFIG_CN.md](TEMPORAL_PIPELINE_CONFIG_CN.md)
- 实验：[EXPERIMENT_PROTOCOL_CN.md](EXPERIMENT_PROTOCOL_CN.md)
