# SentinelVision Codex 工程交接

本文件面向维护源码、配置、脚本和文档的工程代理。开始工作前，先阅读根目录 `AGENTS.md`、本文件与 `GIT_GUIDE_CN.md`，再检查当前工作树状态。

## 1. 先识别工作环境

SentinelVision 有两种合法环境，不能混为一谈。

| 环境 | 识别条件 | 允许的工作 |
| --- | --- | --- |
| 完整便携工作库 | 存在 `RUNTIME`、`TOOLS`、`DATASETS`、`RESULTS` 与 Windows 启动脚本。 | 运行自检、桌面 UI、训练、人工审核与检测回归。 |
| 源码克隆或临时工作树 | 通常只有 Git 跟踪的源码；被忽略的运行时、数据、模型和工具可能不存在。 | 修改源码/文档、运行可用的纯 Python 测试；不可宣称已验证 GPU 推理或便携运行。 |

无论位置在哪个盘符，应用代码都必须从工作库根目录推导路径。不要把某个盘符、用户目录或旧项目目录写入应用代码。不要重新初始化、重新克隆覆盖或重置现有 Git 仓库。

## 2. 不可改变的边界

- `RESULTS` 是人工批准的生产模型区。训练候选只能写入 `TRAINING_OUTPUTS`，不得自动部署或替用户批准。
- 新模型仍通过 `RESULTS` 动态扫描与复选框接入；默认不选中，未知类别默认不告警。
- `.gitignore` 保护 `RUNTIME`、`TOOLS`、`.runtime`、数据集、模型二进制、训练输出、归档、运行日志与实验 trace。运行事件、健康日志、困难样本和 trace 必须写到 `.runtime`。
- 保留实验兼容基线。时序 PPE 必须显式启用，不能静默改变旧用户配置的行为。
- 保留既有训练 → 人工审核 → 批准部署流程，不以“改进模型”名义绕开审核门禁。

## 3. 当前产品状态

工作台包含“开始、训练新模型、审核与部署、已部署模型”四页；SENTINEL 使用动态模型扫描而非旧 SINGLE/DUAL 开关。

可选的“生产时序防护”已实现：

- `hat/person` 冲突消解；
- 高低置信双阈值与候选容量保护；
- 受限 Alpha-Beta 运动预测与歧义拒绝；
- 内部候选 ID 与公开 PPE Track ID 分离；
- 连续高置信命中、连续持续时间与连续段 EMA 的公开编号门；
- N-of-M / EMA 时序证据、风险状态机、独立 Event ID 和每 Track 冷却；
- Fire 等非 PPE 告警类的通用真实证据门。

未发布 PPE 候选、缺失框和预测框不会显示为当前检测，也不会产生告警证据。已发布目标可在短暂、唯一的运动缺失后继承原公开 ID；长期遮挡、多人完全交叉和外观 ReID 不在当前保证范围内。

当前默认算法配置是 `config/safety_pipeline.yaml` 的 schema 3。schema 1/2 保留旧实验语义；算法配置无效时 UI fail-closed 锁定兼容基线。当前实现不等于特定厂区的安全认证；专用 Fire Temporal、Dynamic Risk、Hard Sample、Watchdog、自动 trace/双重放报告、自动训练优化和 Dynamic ROI 均未完成。

## 4. 开始维护的最小检查

优先使用便携 Git；若当前是源码工作树且 `TOOLS/Git/cmd/git.exe` 不存在，可使用宿主 Git，但不得把宿主工具路径写入项目文件。

```powershell
git status --short --branch
git log -5 --oneline
```

完整便携工作库还应运行：

```powershell
& '.\RUNTIME\python\python.exe' -I -B -X utf8 '.\portable\verify_workspace.py'
& '.\RUNTIME\python\python.exe' -I -B -X utf8 '.\portable_check.py'
```

源码工作树若没有 `RUNTIME`、模型或数据集，只运行与改动相称的源码测试；不要为通过自检而复制、下载或生成受保护内容。

## 5. 验证与文档要求

- 涉及 `safety_pipeline`、模型选择、事件或警报的修改，至少运行相关单元测试；完整便携环境可运行 `python -m unittest discover -s tests -q` 和离屏 UI 冒烟。
- 变更默认参数、发布门、运行条件、脚本行为、模型审核边界或 Git 行为时，同步更新入口说明、便携说明、Git 指南与对应 `docs/` 文档。
- 公开说明必须区分“完整便携工作库”和“源码工作树”，不得把历史验证结果说成当前工作树已经实际执行。
- 当前回归基线为 130 项 unittest；该数字只能在完整测试重新执行后更新。

## 6. Git 协作

只暂存本次任务相关文件，先检查暂存差异与受保护目录。只有用户明确要求时才提交；只有用户明确要求同步时才推送。当前维护请求若只说“提交”，默认创建本地提交而不推送。

面向便携成品的脚本会优先使用盘内 Git/gh 工具，并在需要时请求 GitHub 浏览器登录。`SAVE_VERSION.cmd` 在检测到已授权 GitHub CLI 时会尝试推送；需要严格本地提交时，使用 `portable/save_version.ps1 -NoPush`。完整流程见 `GIT_GUIDE_CN.md`。

## 7. 文档地图

- `README.md`：项目入口、运行场景与产品边界。
- `START_HERE_CN.md`：日常操作者的完整便携成品指南。
- `PORTABLE_GUIDE_CN.md`：便携运行条件、离线恢复与公共电脑限制。
- `GIT_GUIDE_CN.md`：便携成品与源码工作树的 Git 操作边界。
- `docs/`：架构、配置、实验协议与课程设计摘要。

不要删除这些项目级说明；根目录 README 已移除不适用于 SentinelVision 的上游 YOLOv5 在线教程。第三方组件目录中的原始 README 仍可能用于对应集成，保留不动。
