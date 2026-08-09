# 硕士课程设计技术实现摘要

## 题目建议

**基于时空一致性与风险状态机的全天候工业视觉安全监测系统**

## 可直接用于报告的摘要

本项目在既有 YOLOv5 工业视觉检测系统基础上，针对安全帽互斥类别重叠、逐帧分类抖动、低识别率造成轨迹碎片、不同目标共享冷却时间以及告警缺少事件生命周期等问题，设计并实现了一套可配置、可回退的生产时序安全分析管线。系统不改变原有检测网络和生产模型，而是在 canonical detection 后增加类别冲突消解、逻辑头部跟踪、多帧证据融合及每轨迹风险状态机。冲突消解器以 IoU、中心距离和面积比例作为几何约束，对 `hat` 与 `person` 进行一对一配对，同时保留两个原始检测分数和模型来源；得分差不足时输出 UNCERTAIN，避免任意选择。逻辑头部跟踪器不依赖当前 PPE 类别，并采用严格关联、受限 Alpha-Beta 中心运动预测和歧义拒绝策略：预测只帮助下一次实际检测继承身份，不会被当作新的风险证据；多人关系含糊时则宁可新建候选，也不错误继承旧风险与冷却。内部候选 ID 从首帧积累证据，只有同时达到连续高置信命中数、连续持续时间和置信度 EMA 门槛后才分配会话内连续的公开 Track ID；缺失会重置未发布候选的发布连续证据，使短暂高置信误检既不显示识别框，也不造成用户看到的编号跳跃。低置信检测只能续接已公开目标，不能创建候选或推进违规证据。时序证据引擎结合 N-of-M 投票、置信度指数移动平均、短暂丢失容忍和恢复滞回，输出 HELMET、NO_HELMET、UNCERTAIN 或 UNKNOWN 稳定状态。风险状态机按 Track 独立维护 NORMAL、SUSPECT、CONFIRMED、ALARMED、COOLDOWN 与 RECOVERING 状态，并要求真实、高质量且持续足够时间的证据才能确认事件，从而过滤短闪噪声；每次完整违规 episode 具有独立 Event ID，避免一名人员的冷却抑制另一名新进入人员的告警。

为保证工程可用性，新增算法被实现为不依赖 Qt、Torch 和 OpenCV 的纯 Python 模块，所有参数通过版本化 YAML 配置并进行有限数、范围和兼容性校验；schema 1/2 保持实验基线语义，只有 schema 3 显式启用生产门禁。ROI / Mask 先作用于原始框，防止域外冲突框改变融合框的监控域归属；PPE 类别开关只控制稳定结果显示，完整互斥证据仍用于时序判断。视频源、模型或分析模式变化会清空全部相关时序状态；只改变 PPE 类别开关时仅重置 PPE 管线并保留 Fire 冷却。对于 Fire 等非 PPE 告警类，新增按实际 Track 独立的通用时序证据门禁，以命中数、持续时间、窗口出现率和 EMA 置信度过滤单帧反光或短闪框；它不等同于后续专用 FireTemporalVerifier。事件详情与 CSV/JSON 导出增加 Session ID、Track ID、Event ID、稳定状态、风险状态、确认延迟、证据命中数、持续时间、稳定度、判定原因和配置指纹；风险状态转换、告警与人工复核以 JSONL 追加写入本地运行目录。本地录像使用 PTS 并在异常时按真实解码帧索引回退，直播使用捕获时刻而不是推理结束时刻，避免推理时延污染速度估计。该实现保持 `RESULTS` 人工批准边界、动态模型扫描和移动硬盘离线运行能力不变。

后续实验将在完全相同的 YOLOv5 detection 上对比兼容基线与 Proposed 管线，重点评价每小时误报、漏报事件率、平均告警延迟、每事件重复告警、安全帽类别冲突率和状态闪烁率。由此可以验证：即使模型 mAP 不变，基于时空一致性和风险状态管理的后处理仍可提升 24 小时工业视觉监测的稳定性与实际可用性。

## 本阶段技术贡献

1. 将互斥类别从两个独立 detection 重构为保留双分数证据的 `HeadObservation`。
2. 实现 PPE 类别无关的一对一轻量跟踪，以允许不匹配的最大总 IoU、受限 Alpha-Beta 运动预测和歧义拒绝减少类别抖动、检测闪烁造成的轨迹碎片。
3. 分离内部候选 ID 与公开 Track ID，并增加高低置信双阈值、连续发布命中/持续时间、置信度 EMA 和候选容量上限，使暂态误检不渲染、不占用公开编号或无限消耗资源。
4. 结合 N-of-M、EMA、missing tolerance、证据持续时间和 hysteresis 形成稳定 PPE 状态；缺失、预测和不合格低置信结果不推进风险。
5. 建立每 Track / 每 episode 风险状态和事件身份，并为非 PPE 告警类提供通用时序证据门禁，消除全局冷却和跨目标证据聚合造成的干扰。
6. 建立兼容基线、schema-aware 配置指纹、结构化证据审计日志和可确定性测试，为对照实验与现场标定提供工程基础。

## 研究边界与诚实说明

- 当前没有改变 YOLOv5 网络或声称提高 mAP。
- 当前 PPE 使用完整冲突、跟踪、证据融合与风险状态机；Fire 等非 PPE 告警类已有通用时序防闪门禁，但 Fire 专用空间聚类、形变建模和持续火焰验证仍待实现。
- 当前短时重关联不使用外观 ReID；长时间遮挡、多人完全交叉或新人进入旧位置时不能保证身份恢复，歧义场景优先新建候选。
- False alarm 可以通过人工复核统计；Missed event 必须另有带起止区间的 ground truth。
- 当前参数是实验初值，尚未构成特定工厂的标定或安全认证。
- Phase 6 专用 Fire Temporal（当前仅完成通用告警门禁）、Phase 7 Dynamic Risk、Phase 8 Hard Example Collector、Phase 9 Watchdog、Phase 11 自动训练优化和 Phase 12 Dynamic ROI 属于后续阶段。
- Phase 10 当前只完成指标字段、事件日志与实验协议基础；自动 trace、同 detection 双重放、CSV / JSON 汇总和可读报告工具尚未实现。

## 代码与验证入口

- 算法：`safety_pipeline/`
- 默认配置：`config/safety_pipeline.yaml`
- UI / 实时接入：`SentinelVision4.py`
- 测试：`tests/`
- 架构：[ARCHITECTURE_CN.md](ARCHITECTURE_CN.md)
- 配置：[TEMPORAL_PIPELINE_CONFIG_CN.md](TEMPORAL_PIPELINE_CONFIG_CN.md)
- 实验：[EXPERIMENT_PROTOCOL_CN.md](EXPERIMENT_PROTOCOL_CN.md)
