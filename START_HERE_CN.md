# SentinelVision 工作库使用说明

## 你只需要记住一个文件

双击根目录中的 `START_HERE.cmd`。

- 工作库已自带 64 位 Python 3.11.9、CUDA 11.8 版 PyTorch、PySide6 和全部训练依赖。目标电脑不需要安装 Python、CUDA Toolkit，也不需要联网或管理员权限。
- 每次都会先完成自检，然后打开图形化工作台。首次换电脑不需要重新安装环境。
- 工作库不依赖固定盘符。移动硬盘从 `E:` 变为其他盘符时，项目路径会自动跟随。

适用范围是允许从 USB 运行程序的 Windows 10/11 x64 电脑。SENTINEL 与 GPU 训练要求兼容 CUDA 11.8 的 NVIDIA 驱动；RTX 4080 Laptop 和 RTX 4090 均在支持范围。若单位的 AppLocker/安全策略禁止从移动设备运行 EXE/DLL，需要管理员放行，软件不能绕过系统策略。

如果 `RUNTIME` 被误删或损坏，双击 `SETUP_ENVIRONMENT.cmd`。它只使用工作库自带的 Python 压缩包和离线 wheelhouse 重建，不读取本机 Python、不联网。

怀疑移动硬盘曾异常断开或复制不完整时，关闭训练和 SENTINEL 后双击 `VERIFY_WORKSPACE.cmd`。它只检查固定关键文件、10 个训练权重和 `RESULTS` 模型是否存在且大小正常，不遍历整个数据集，也不计算全库哈希，通常几秒内完成。

不要在训练、复制模型或安装环境时拔出移动硬盘。

## 工作台四个页面

### 1. 开始

- **启动 SENTINEL 检测**：打开实时检测界面。
- **运行完整环境自检**：检查 Python、CUDA、依赖和 `RESULTS` 中的部署模型。
- 自检允许单个损坏模型被隔离；只要还有正常模型，其他模型仍可使用。

### 2. 训练新模型

1. 选择数据集 YAML。
2. 在“预训练型号”中选择初始权重。默认推荐 `YOLOv5s`；`n/s/m/l/x` 是标准 640px 模型，`n6/s6/m6/l6/x6` 是高显存的 1280px 高级模型。
3. 填写实验名称、epochs、batch 和图像尺寸。
4. 先点“检查数据集”。
5. 检查通过后点“开始训练候选模型”。

所有训练产物只会进入 `TRAINING_OUTPUTS`。训练程序拒绝写入 `RESULTS`，因此未审核模型不会自动替换线上检测模型。

工作库自带的 10 个官方目标检测预训练权重位于 `PRETRAINED_WEIGHTS`。界面会显示速度、精度和显存建议，并在训练前校验官方文件指纹。分割、分类和没有官方 checkpoint 的实验 YAML 不会混入普通目标检测选择器，详细原因见 `PRETRAINED_WEIGHTS/README_WEIGHTS_CN.md`。

训练意外中断时，点击“安全续训 last.pt”，选择该候选实验的 `weights/last.pt`。程序会重新核对原数据集并保留原 `opt.yaml` 备份；不要用普通 `--resume` 绕过检查。

项目已经把历史数据整理成两个一致、互不泄漏的训练视图：

- `DATASETS/combined_legacy_v1/dataset.yaml`：`person=0, hat=1, fire=2`
- `DATASETS/fire_only_legacy_v1/dataset.yaml`：`fire=0`

原始历史数据没有被修改。生成规则在 `portable/prepare_training_datasets.py` 中，可追溯、可重建。

### 3. 审核与部署

1. 选择 `TRAINING_OUTPUTS` 中的候选 `best.pt`。
2. 点“读取类别与指纹”，确认 checkpoint 类别及顺序。
3. 设置模型 ID、显示名、每个类别的中文名、英文名、告警等级和颜色。
4. 人工检查指标曲线、类别顺序和失败样本；点击“用图片/视频测试候选”，检查真实媒体上的漏报、误报和类别名称。测试结果只写入 `TRAINING_OUTPUTS/_REVIEWS`，不会部署模型。
5. 只有四项审核均确认后，才可点“最终批准并部署到 RESULTS”。

部署时会再次复制校验 SHA-256。替换同 ID 模型时，旧版本移入 `MODEL_ARCHIVE`，不会直接丢失。

### 4. 已部署模型

这里显示 SENTINEL 能从 `RESULTS` 扫描到的全部模型及状态。训练候选区和历史 `runs` 永远不会被扫描为部署模型。

## SENTINEL 中怎样选择模型

SINGLE/DUAL 模式已经删除，改为直接勾选模型：

- 原 SINGLE 效果：只勾选 **COMBINED**。
- 原 DUAL 效果：取消 **COMBINED**，勾选 **FIRE ONLY** 和 **HELMET ONLY**。

如果两个模型会检测相同的统一类别，界面默认阻止同时勾选，避免重复框和重复告警。模型加载在后台进行；勾选越多，占用显存越多、速度越慢。

SENTINEL 每次启动都会重新扫描 `RESULTS`，也可以在界面点击“重新扫描 RESULTS”。新发现但未登记的模型默认不勾选，未知类别默认只画框、不告警。

## SENTINEL 中怎样选择分析模式

检测界面的“检测”标签内有一个简洁的“分析模式”选择器：

- **兼容基线**：保留原有按来源/类别的轻量跟踪、30 帧窗口和 10 秒防刷屏，适合回退和实验对照；
- **PPE 时序增强**：对 `hat=已戴安全帽`、`person=未戴安全帽` 先做冲突消解，再按逻辑头部 Track ID 做多帧稳定判断和独立风险状态管理。

旧设置默认使用兼容基线，不会在升级后静默改变生产报警逻辑。启用增强模式后，画面标签会显示 `Track #`；“显示 raw / stable 调试信息”只用于实验排查，普通值守无需开启。切换视频源或模型会清空全部旧轨迹和证据；切换分析模式或 PPE 类别只重置 PPE 状态，不会清除 Fire 的冷却。

增强告警详情和 CSV/JSON 导出会包含 Track ID、Event ID、稳定状态、风险状态与确认延迟。另一名新进入的未戴帽人员拥有独立 Track/Event，不会被第一名人员的冷却时间压制。当前 Fire 仍走原有路径；完整设计和参数说明见 `docs/ARCHITECTURE_CN.md` 与 `docs/TEMPORAL_PIPELINE_CONFIG_CN.md`。

## 手工放入外部模型

模型可以来自其他电脑或其他 YOLOv5 项目。人工检查完成后，可以放到：

```text
RESULTS\模型名称\weights\best.pt
```

SENTINEL 会把它显示为“新发现”。推荐随后使用工作台“审核与部署”登记类别语义、告警规则和指纹；裸 `.pt` 使用 Python pickle 格式，只能放入来源可信、由你自己训练或确认过的文件。

## 目录边界

- `DATASETS`：规范数据集。
- `TRAINING_OUTPUTS`：未审核候选实验。
- `PRETRAINED_WEIGHTS`：只读使用的官方 YOLOv5 v7.0 迁移学习初始权重。
- `RESULTS`：SENTINEL 唯一部署模型区。
- `MODEL_ARCHIVE`：被替换模型的可恢复归档。
- `RUNTIME`：项目自带的 Python、CUDA/PyTorch 依赖和离线修复包；不要改动。
- `.runtime`：按电脑隔离的缓存、临时文件和设置；可以由程序自动重建。

增强 PPE 的风险状态转换，以及所有模式的告警和人工复核，会追加到 `.runtime/events/ppe_events.jsonl`。该文件不含逐帧视频，不进入 Git；实验前可按 `docs/EXPERIMENT_PROTOCOL_CN.md` 归档所需记录。

## Git 版本保护

源码、脚本、配置和文档由本地 Git 及私有 GitHub 仓库保护；模型、数据集、运行环境和训练输出因为体积及 GitHub 限制而保留在移动硬盘，不会误上传。

- 修改完成后双击 `SAVE_VERSION.cmd`，自动保存一个版本；
- 双击 `SYNC_GITHUB.cmd`，同步私有 GitHub；
- 双击 `GIT_STATUS.cmd`，查看未保存修改和最近历史；
- 完整说明见 `GIT_GUIDE_CN.md`。

第一次使用会由 `INSTALL_GIT_TOOLS.cmd` 在工作库中安装便携 Git，不修改电脑系统。公共电脑上的 GitHub 登录凭据不会写入移动硬盘。
