# SentinelVision Git / GitHub 使用说明

Git 只保存可审查的源码、脚本、配置和文档。运行时、模型、数据集、训练输出、用户设置和运行日志保留在工作库或 `.runtime`，由根目录 `.gitignore` 保护，绝不应加入 Git。

## 先确认工作环境

| 环境 | Git 工具来源 | 推荐方式 |
| --- | --- | --- |
| 完整便携工作库 | 优先使用 `TOOLS/Git/cmd/git.exe` 和 `TOOLS/GitHubCLI`。 | 使用根目录 `*.cmd` 脚本。 |
| 源码克隆或临时工作树 | 便携工具可能不存在。 | 使用已安装的 Git，仅提交当前任务相关文件。 |

不要在已有克隆、临时工作树或已初始化的便携工作库中再次运行 `ENABLE_GIT.cmd`。它只用于一个复制出来、尚未初始化的完整便携工作库。

## 受保护内容

以下内容不属于 Git/GitHub，同步前后都不应删除、覆盖或手工加入暂存区：

- `RUNTIME`、`TOOLS`、`.runtime`；
- `DATASETS` 与其他原始数据目录；
- `PRETRAINED_WEIGHTS`、`RESULTS`、`MODEL_ARCHIVE`、`TRAINING_OUTPUTS`、`runs`；
- `.pt`、`.onnx`、`.engine` 等模型二进制；
- 日志、截图、实验 trace、用户设置、凭据和临时审核产物。

GitHub 的单文件限制与仓库体积限制只是附加原因；这些目录首先是运行与人工审核资产，不应被源码同步流程处理。

## 完整便携工作库：日常操作

1. 双击 `GIT_STATUS.cmd` 查看当前分支、未保存修改和最近历史。
2. 首次在一台没有盘内 Git 工具的电脑上使用时，双击 `INSTALL_GIT_TOOLS.cmd`。它只向工作库的 `TOOLS` 写入工具，不修改系统安装。
3. 若这是一个全新复制的完整工作库且 Git 尚未初始化，才双击 `ENABLE_GIT.cmd`。
4. 完成并核对修改后，双击 `SAVE_VERSION.cmd` 创建版本。
5. 需要与 GitHub 对齐或第一次授权时，双击 `SYNC_GITHUB.cmd`。

`SAVE_VERSION.cmd` 会暂存所有未被忽略的改动并创建提交；若本机已经有已授权的 GitHub CLI，它还会尝试推送。需要严格的仅本地提交时，请在 PowerShell 中运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File '.\portable\save_version.ps1' -Message '说明本次修改' -NoPush
```

`SYNC_GITHUB.cmd` 会先保存受保护边界之外的未提交修改，再检查远端、必要时合并或变基并推送。因此在运行前先查看状态，处理你不希望纳入提交的文件；它不是只读命令。

首次同步会要求在浏览器完成 GitHub 授权。令牌由当前 Windows 凭据系统保存，不写入移动硬盘；公共电脑使用后应退出 GitHub 并清除该电脑的凭据。

## 源码克隆或临时工作树：维护操作

源码工作树通常不包含 `TOOLS`、`RUNTIME`、模型和数据。这是正常的 Git 忽略结果，不需要安装或复制这些目录来修改文档与代码。

```powershell
git status --short --branch
git diff --check
git add -- <本次任务文件>
git diff --cached --name-only
git commit -m '清晰的修改说明'
```

只在用户明确要求同步时执行 `git push`。不要把临时工作树的绝对路径、宿主工具路径或本机凭据写入项目文件。若工作树处于 detached HEAD，应先按维护平台的工作流创建或切换到合适分支，而不是初始化新仓库。

## 撤销与恢复

- `REVERT_LAST_COMMIT.cmd`：推荐。它创建反向提交，保留历史，适合已经共享的版本。
- `DELETE_LAST_COMMIT_HARD.cmd`：危险。它只允许在工作区干净时删除最近一次提交；若远端正好指向该提交，可能以 `force-with-lease` 改写远端。

两者均需要双重确认。除非用户明确要求且目标提交已核对，不要使用 `reset --hard`、强制推送或覆盖工作库。

## 同步失败时

推送被拒绝通常表示远端已有新提交。先获取远端状态并检查差异；使用 rebase 或合并时保留双方改动，重新运行相称测试后再推送。不要用强制推送覆盖远端，也不要把“本地已提交”说成“已经同步”。

私有仓库地址：`https://github.com/Pitohui-Dichrous/SentinelVision-Workspace`。
