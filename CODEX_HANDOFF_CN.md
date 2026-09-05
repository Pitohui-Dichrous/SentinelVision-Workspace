# SentinelVision 新 Codex 账号交接说明

## 给新账号的第一条指令

请把 `E:\SentinelVision_Workspace` 直接作为 Codex 工作目录，并先完整阅读根目录的 `AGENTS.md` 和本文。这个 E 盘目录是当前唯一权威工作库；不要继续编辑旧的 `D:\python\yolov5-7.0`，也不要把程序路径硬编码成 `E:`。应用代码必须继续从工作库根目录计算相对路径，才能在公共电脑盘符变化时即插即用。

不要重新初始化 Git，也不要把 GitHub 仓库重新克隆到 E 盘现有目录。GitHub 只保存源码和配置，不包含数据集、模型、训练输出、便携 Python 和工具；覆盖现有目录会丢失这些仅保存在移动硬盘中的大文件。

新账号第一次接手时建议先执行：

```powershell
Set-Location 'E:\SentinelVision_Workspace'
& '.\TOOLS\Git\cmd\git.exe' status --short
& '.\TOOLS\Git\cmd\git.exe' log -5 --oneline
& '.\RUNTIME\python\python.exe' -I -B -X utf8 '.\portable\verify_workspace.py'
& '.\RUNTIME\python\python.exe' -I -B -X utf8 '.\portable_check.py'
```

快速检查只核对固定关键文件是否存在且大小合理，不计算工作库哈希，也不遍历数据集或整个移动硬盘。用户已经明确要求不要恢复全文件哈希或全盘清单流程。

## 当前交付状态

- 当前权威工作库：`E:\SentinelVision_Workspace`。
- 私有 GitHub：`https://github.com/Pitohui-Dichrous/SentinelVision-Workspace`，默认分支为 `main`。
- 日常总入口：`START_HERE.cmd`。它先运行环境自检，再打开图形化工作台。
- 工作库自带 Windows x64 Python 3.11.9、PyTorch 2.2.1+cu118、PySide6、完整训练依赖、离线修复包、便携 Git 和 GitHub CLI。
- 已在 RTX 4080 Laptop 上通过 CUDA、自检、模型加载和多模型切换测试；目标公共电脑 RTX 4090 只需安装兼容 CUDA 11.8 的 NVIDIA 驱动，不需要另装 Python 或 CUDA Toolkit。
- UI 已按用户指定的 UI UX Pro Max 技能重做为浅色、简洁排版的原生工作台，包含“开始”“训练模型”“审核与部署”“模型库”四页。训练高级参数按需展开，类别审核使用摘要与独立编辑区。检测台保留明暗主题，突出视频画面，区域工具按需展开。原有视频、摄像头、告警、截图和检测功能均保留。设计规则和验证入口见 `design-system/sentinelvision/MASTER.md`。
- 检测台已加入可回退的“生产时序防护”：`hat/person` 冲突消解、高低置信双阈值、连续高置信命中/持续时间/EMA 公开 Track ID 门禁、受限 Alpha-Beta 运动预测重捕获、N-of-M/EMA/滞回、每 Track 风险状态机、Event ID 与结构化证据日志。未发布短噪声不显示识别框也不消耗公开编号；缺失框和预测框不构成告警证据。Fire 等非 PPE 告警类已接入通用按 Track 时序门禁，专用 FireTemporalVerifier 仍属后续阶段。
- 训练和检测代码均使用相对路径；交接前使用 Git 跟踪文件搜索确认，没有 `D:\python`、`D:/python` 或其他 D 盘路径硬编码。

## 模型训练与数据集

`PRETRAINED_WEIGHTS` 已放置 10 个 YOLOv5 v7.0 目标检测初始权重：

- P5 / 640px：`yolov5n.pt`、`yolov5s.pt`、`yolov5m.pt`、`yolov5l.pt`、`yolov5x.pt`；
- P6 / 1280px：`yolov5n6.pt`、`yolov5s6.pt`、`yolov5m6.pt`、`yolov5l6.pt`、`yolov5x6.pt`。

工作台会显示型号、速度/精度取向、推荐图像尺寸与显存提示。其他仅有 YAML、没有兼容官方 checkpoint 的结构不会冒充成可直接选择的 `.pt` 权重。

当前规范训练数据集：

- `DATASETS/combined_legacy_v1/dataset.yaml`：`person=0, hat=1, fire=2`；
- `DATASETS/fire_only_legacy_v1/dataset.yaml`：`fire=0`。

所有新训练只能写入 `TRAINING_OUTPUTS`。`safe_train.py` 会拒绝把训练输出写到 `RESULTS`，并阻止不安全的 `--resume`、`--exist-ok` 等绕过方式。训练中断后应从工作台使用“安全续训 last.pt”。

## 人工审核与生产部署边界

`RESULTS` 是 SENTINEL 唯一允许扫描的生产模型目录，必须视为用户人工批准区。任何训练候选都不得自动复制到这里。

正确流程：

1. 在 `TRAINING_OUTPUTS` 训练候选模型；
2. 在工作台“审核与部署”页面读取类别并用真实图片/视频测试；
3. 用户人工确认指标、类别、漏报/误报和失败样本；
4. 四项审核全部勾选后，用户手动执行最终批准；
5. 只有这一步才允许复制进 `RESULTS`。同 ID 旧模型会进入 `MODEL_ARCHIVE`。

当前 `RESULTS` 中有三个来源未知但已实际运行正常的历史模型：

- `COMBINED`：`person, hat, fire`；
- `FIRE ONLY`：`fire`；
- `HELMET ONLY`：`person, hat`。

这些历史模型不一定由当前仓库训练，禁止根据旧 `runs` 目录猜测或改写其来源。登记信息位于 `RESULTS/model_registry.yaml`。

## SENTINEL 动态模型选择

旧的 SINGLE/DUAL 用户界面已经移除。SENTINEL 每次启动都会动态扫描 `RESULTS`，用户通过复选框选择要加载的模型：

- 原 SINGLE 效果：只勾选 `COMBINED`；
- 原 DUAL 效果：勾选 `FIRE ONLY` 和 `HELMET ONLY`，不要勾选 `COMBINED`。

新发现的模型默认不勾选，未知类别默认只画框、不告警。类别重叠的模型默认不能同时选择，以免产生重复检测框和重复告警。未来用户添加新目标模型时，仍必须沿用动态扫描与“默认不选中”的规则，不能重新写死三个模型。

当前本机设置文件 `.runtime/config/sentinel_settings.json` 中选择的是 `HELMET ONLY`。这只是当前电脑的用户选择，不是代码默认值；用户可以在 SENTINEL 中自行更改，程序会记住选择。多模型推理目前在同一个检测循环内顺序执行，会增加显存占用和单帧耗时；它不是多 CUDA 流并发实现。

安全管线默认参数位于 `config/safety_pipeline.yaml`，其算法配置当前为 schema 3；schema 1/2 保留旧实验语义，只有 schema 3 启用新的证据与运动门禁。本机 `.runtime/config/sentinel_settings.json` 是另一套用户设置 schema，当前为 schema 3，显式选择 `ppe_temporal`（UI 显示“生产时序防护”）。低频风险转换、告警和人工复核追加到 `.runtime/events/ppe_events.jsonl`。算法配置无效时 UI 会 fail-closed 锁定兼容基线。架构、配置、实验方法和课程设计摘要位于 `docs/`。

不要把当前实验初值写成行业标准，也不要把代码回归通过等同于特定厂区的安全认证。Phase 6 专用 Fire Temporal 当前只完成通用告警门禁，空间聚类与形变验证仍待实现；Phase 7 Dynamic Risk、Phase 8 Hard Sample、Phase 9 Watchdog、Phase 11 自动优化和 Phase 12 Dynamic ROI 尚未实现；Phase 10 只完成指标/日志/实验协议基础，自动 trace、双重放与报告工具仍待后续。

## Git 与大文件边界

根目录 `.gitignore` 是强制保护边界。Git 只管理可审查的源码、脚本、配置、文档和 YAML；以下内容完整保留在移动硬盘，但不进入 Git/GitHub：

- `RUNTIME`、`.runtime`、`TOOLS`；
- `DATASETS`、历史数据目录；
- `PRETRAINED_WEIGHTS` 和所有 `.pt/.onnx/.engine` 模型二进制；
- `RESULTS`、`MODEL_ARCHIVE`、`TRAINING_OUTPUTS`、`runs`；
- 日志、缓存、用户设置和临时审核产物。

修改前必须使用便携 Git 检查状态：

```powershell
& '.\TOOLS\Git\cmd\git.exe' status --short
```

修改后先运行相应测试。只有用户明确要求提交或同步时，才分别使用：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File '.\portable\save_version.ps1' -Message '清晰的英文或中文提交说明' -NoPush
powershell.exe -NoProfile -ExecutionPolicy Bypass -File '.\portable\sync_github.ps1'
```

也可以让小白用户双击 `SAVE_VERSION.cmd`、`SYNC_GITHUB.cmd` 和 `GIT_STATUS.cmd`。GitHub 凭据由当前电脑的 Windows 凭据系统保存，不在移动硬盘上；新账号或公共电脑第一次同步时需要单独完成浏览器授权。软件运行、训练和本地 Git 保存不要求登录 GitHub。

## 已完成的关键验证

- E 盘可写，便携 Python 和隔离搜索路径正常；
- PyTorch、TorchVision、OpenCV、PySide6、IPython、GitPython 等依赖正常；
- RTX 4080 Laptop CUDA 可用，三个生产模型均能加载；
- `COMBINED` 与 `FIRE ONLY + HELMET ONLY` 两种选择方式均实际完成 GPU 模型切换；
- 10 个预训练权重均能作为 checkpoint 读取，P5/P6 配置映射正确；
- 训练安全边界、人工审核门禁、部署归档和动态 `RESULTS` 扫描通过测试；
- 快速关键文件检查与完整环境自检通过；
- PPE/通用告警纯算法、双 ID、连续发布门、运动预测重捕获、证据门禁、集成契约与离屏 UI 共 130 项 unittest 全部通过；20 个稀疏/密集目标与 40 个密集目标的便携 Python 跟踪基准 p95 分别约 1.9 / 2.3 / 7.4 ms（仅为本机回归参考，不代表目标厂区 SLA）。
- 用户已明确要求：从本阶段起，每次创建 Git 提交后同时推送 GitHub；若凭据或网络导致推送失败，必须明确报告，不得把本地提交误称为已同步。保护目录仍不得进入 Git。

## 新账号继续工作时的原则

1. 先阅读 `AGENTS.md`，再看 Git 状态，保留用户已有修改。
2. 只在 `E:\SentinelVision_Workspace` 工作；D 盘旧项目不再是来源。
3. 不在应用代码中硬编码 E 盘，因为公共电脑可能分配其他盘符。
4. 不自动部署训练模型，不替用户决定模型已通过人工审核。
5. 不恢复全库哈希、全盘遍历或大型 Git 提交。
6. 新模型继续通过 `RESULTS` 动态扫描和复选框接入，并默认不选中。
7. 修改完成后做相称测试；只有用户明确要求时才创建 Git 提交。当前用户已持续授权“每次提交同时推送 GitHub”，因此后续一旦按指示提交就应立即推送；若用户撤销该要求，以最新指示为准。
8. 当前没有已知阻塞性故障。接手后应先询问用户下一项具体研究或开发目标，不要无目的重构已经稳定运行的训练、部署和检测链。

更详细的小白使用说明见 `START_HERE_CN.md`，便携运行机制见 `PORTABLE_GUIDE_CN.md`，Git 操作见 `GIT_GUIDE_CN.md`。

用户在完成编写 CODEX_HANDOFF_CN.md 文件后，又对git做了少量修改，详细使用方式请查看 GIT_GUIDE_CN.md
