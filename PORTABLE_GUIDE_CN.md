# SentinelVision 移动硬盘技术说明

## 日常入口

双击 `START_HERE.cmd`。它会自动：

1. 定位当前移动硬盘盘符和项目根目录。
2. 直接使用 `RUNTIME/python/python.exe` 中随盘携带的独立 Python 3.11.9。
3. 将 YOLOv5、Torch、CUDA 和 Matplotlib 的缓存留在移动硬盘并按电脑名隔离。
4. 检查 Python、CUDA、依赖和动态部署模型。
5. 打开图形化工作台。

`START_SENTINEL.cmd`、`CHECK_ENVIRONMENT.cmd`、`CHECK_DATASETS.cmd` 和 `TRAIN_SAFE.cmd` 仍保留给需要单独执行某一步的用户。

`VERIFY_WORKSPACE.cmd` 是可选的快速验收入口。移动盘异常断开或复制后，可用它检查固定关键文件是否存在、大小是否明显异常；它不会遍历整个数据集，也不会计算全库哈希。

## 目标电脑条件

- Windows 10/11 x64，并允许从 USB 存储设备执行 EXE 和 DLL。
- 已安装兼容 CUDA 11.8 的 NVIDIA 驱动。RTX 4080 Laptop、RTX 4090 均受支持；不需要另装 CUDA Toolkit。
- 不要求安装 Python、pip、Git、开发工具或管理员权限，不要求联网。
- 工作库建议至少保留 25 GB 空间，以便在运行时损坏时进行同盘离线重建。

`RUNTIME/python` 是官方 Windows embeddable Python 与 cp311/win_amd64 依赖组成的独立应用运行时；`python311._pth` 只使用相对路径，不读取宿主 Python、注册表环境或用户 site-packages。PyTorch wheel 自带 CUDA 11.8、cuDNN 等用户态库。

`RUNTIME/bootstrap`、`RUNTIME/wheelhouse` 和 SHA-256 清单组成离线恢复源。`SETUP_ENVIRONMENT.cmd` 会先校验 81 个恢复文件，再在临时目录重建并自检；完整成功后才替换运行时。旧的普通 venv 没有进入成品，因为它会绑定创建电脑上的基础 Python 和盘符。

## 训练安全边界

- `PRETRAINED_WEIGHTS` 随盘提供 10 个 YOLOv5 v7.0 官方目标检测 checkpoint：5 个 P5/640px 型号与 5 个 P6/1280px 型号。工作台按用途分组并在训练前校验所选官方权重的 SHA-256。
- 分割、分类、YOLOv3 以及没有官方 checkpoint 的实验结构不会进入普通检测训练选择器，避免任务类型或网络结构不兼容。
- `dataset_audit.py` 支持标准 YOLO YAML 的 `path:`，检查路径、类别、坐标、图片可解码性，以及不同文件名但内容相同的 train/val/test 泄漏。
- 历史数据已生成两个一致视图：`DATASETS/combined_legacy_v1` 与 `DATASETS/fire_only_legacy_v1`；原始文件没有被重写。生成器使用盘符无关的稳定命名、内容去重和审计后替换。
- `safe_train.py` 强制输出到 `TRAINING_OUTPUTS`，拒绝 `RESULTS`、普通/缩写 `--resume` 和 `--exist-ok` 绕过；工作台提供经过数据复核的安全续训，并在换盘符后改写旧绝对路径。
- 每个候选实验使用唯一时间戳目录，并记录候选清单。
- “停止训练”会结束本次训练的整个进程树，不会在后台遗留占用 GPU 的子进程。
- `last.pt`、`best.pt` 和周期 checkpoint 会先写入同盘临时文件，完整后再原子替换；意外断电或掉盘时尽量保留上一份完整权重。
- 候选模型可直接用真实图片/视频测试，结果保存在 `TRAINING_OUTPUTS/_REVIEWS`。只有工作台中的人工审核和最终批准动作会正式部署模型。

## 动态部署模型

SENTINEL 每次启动扫描 `RESULTS`，按复选框加载模型，不再存在 SINGLE/DUAL 模式。登记模型会校验 SHA-256；新手工模型默认不勾选，未知类别默认不告警。

现有历史模型来源仍标记为未知，不能根据当前项目中的旧 `runs` 推断其真实训练项目。详情见 `RESULTS/README_DEPLOYMENT_CN.md`。

## PPE 时序安全管线

检测台保留“兼容基线”，并提供可显式启用的“PPE 时序增强”。增强算法全部位于工作库源码中，不增加网络下载、外部服务或新的二进制依赖；便携 Python 已有的 PyYAML 足以读取版本化配置。

- Git 跟踪默认配置：`config/safety_pipeline.yaml`；
- 当前电脑的模式选择：`.runtime/config/sentinel_settings.json`；
- 低频事件审计：`.runtime/events/ppe_events.jsonl`。

`.runtime` 整体处于 Git 忽略边界，换电脑时可以重新生成。配置中的 0.60、0.20、8 帧/6 票等数值只用于初始实验，不是行业标准。完整字段见 `docs/TEMPORAL_PIPELINE_CONFIG_CN.md`。

事件日志只记录状态转换、事件字段和人工结论，不逐帧写入视频，因此不会因普通监控帧率快速膨胀。后续若启用完整实验 trace，必须继续写入 `.runtime/experiments` 并设置空间上限，不能放到源码或 `RESULTS`。

当前便携版本只完成 PPE 冲突、跟踪、时序与风险状态的首个纵切。Fire Temporal、Dynamic Risk、Hard Sample、Watchdog、自动实验报告、自动训练优化和 Dynamic ROI 尚未启用，也不会触发额外下载。

## 公共电脑使用注意

- 训练前关闭 SENTINEL，避免两个进程争抢显存。
- 训练或复制模型时不要拔盘。
- 安全弹出移动硬盘前，确认训练进程、SENTINEL 和工作台均已关闭。
- 公共电脑不需要长期保存项目文件；专用环境、缓存、数据和训练产物都位于工作库中。
- 网络摄像头或在线视频源仍然需要网络；本地摄像头、图片、视频、训练、审核和部署均可离线使用。

## 源码版本与 GitHub

工作库自带便携 Git / GitHub CLI 安装脚本，不依赖公共电脑预装开发工具。Git 只跟踪可审查的源码、配置和文档；便携运行时、数据集、训练输出与模型二进制由 `.gitignore` 强制排除，避免 GitHub 100 MB 单文件限制和仓库膨胀。

日常使用 `SAVE_VERSION.cmd` 保存本地版本，使用 `SYNC_GITHUB.cmd` 同步私有仓库。详情见 `GIT_GUIDE_CN.md`。
