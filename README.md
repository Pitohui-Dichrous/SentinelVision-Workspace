# SentinelVision 工业视觉安全监测

SentinelVision 是一个可离线部署的 YOLOv5 工业视觉安全监测工作库。它保留“训练候选 → 人工审核 → 批准部署 → 实时检测”的闭环，并在不改动已部署模型的前提下提供可选的时序安全分析。

## 先确认你处于哪种工作环境

| 场景 | 可以做什么 | 不应做什么 |
| --- | --- | --- |
| **便携运行成品** | 目录中同时存在 `RUNTIME`、`TOOLS`、`DATASETS`、`RESULTS` 和脚本；可启动、训练、审核和部署。 | 不要把运行时、模型、数据集或训练产物加入 Git。 |
| **源码克隆或临时工作树** | 维护源码、配置、文档与单元测试。 | 不要假定本地存在便携 Python、模型、数据集或 Git 工具；不能据此验证生产推理，也不能重新初始化现有仓库。 |

便携成品不依赖固定盘符；应用代码始终从工作库根目录计算路径。源码工作树常常缺少被 Git 忽略的大文件，这是正常现象。

## 便携成品的日常入口

在完整便携工作库中双击 `START_HERE.cmd`。它会先自检，再打开工作台。

- `CHECK_ENVIRONMENT.cmd`：检查 Python、CUDA、依赖和可用生产模型。
- `VERIFY_WORKSPACE.cmd`：复制或异常断开后进行快速关键文件检查；不遍历数据集、不计算全盘哈希。
- `TRAIN_SAFE.cmd`：只向 `TRAINING_OUTPUTS` 写入训练候选。
- `SAVE_VERSION.cmd`、`SYNC_GITHUB.cmd`、`GIT_STATUS.cmd`：版本与同步操作，详见 [Git 使用说明](GIT_GUIDE_CN.md)。

完整硬件、离线与公共电脑条件见 [移动硬盘技术说明](PORTABLE_GUIDE_CN.md)，面向日常使用者的步骤见 [工作库使用说明](START_HERE_CN.md)。

## 生产模型与训练边界

`RESULTS` 是人工批准的生产模型区，也是 SENTINEL 唯一动态扫描的位置。训练输出只能进入 `TRAINING_OUTPUTS`；候选必须在工作台完成真实媒体测试与四项人工审核后，才可由用户明确批准复制到 `RESULTS`。新发现模型默认不选中，未知类别默认不告警。

## 分析模式

- **实验兼容基线**：保留历史逐帧跟踪、窗口计数和冷却语义，用于回退及受控对照。
- **生产时序防护**：可显式启用。它包含 PPE 冲突消解、受限 Alpha-Beta 运动预测、连续高置信公开 Track 门、时序证据、风险状态机与按对象事件冷却。

生产时序防护下，未发布 PPE 候选不会显示识别框或占用公开编号；缺失与预测位置只服务于下一次关联，不构成检测或告警证据。短暂噪声与已发布目标的重捕获边界见 [时序配置说明](docs/TEMPORAL_PIPELINE_CONFIG_CN.md)。

## 技术与实验文档

- [架构说明](docs/ARCHITECTURE_CN.md)
- [时序配置与兼容边界](docs/TEMPORAL_PIPELINE_CONFIG_CN.md)
- [Baseline / Proposed 实验协议](docs/EXPERIMENT_PROTOCOL_CN.md)
- [课程设计技术摘要](docs/MASTER_PROJECT_TECHNICAL_SUMMARY_CN.md)
- [Codex 工程交接说明](CODEX_HANDOFF_CN.md)

当前实现并不等同于工业安全认证。投产前仍需针对目标厂区完成正负样本标定、报警链路验收与持续运行验证；专用 Fire Temporal、Watchdog、Hard Sample 和自动实验报告仍是后续工作。
