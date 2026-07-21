# SentinelVision Git / GitHub 使用说明

## 已采用的安全结构

Git 负责保存源码、界面、训练脚本、配置、文档和模型结构 YAML。GitHub 仓库为私有仓库：

`https://github.com/Pitohui-Dichrous/SentinelVision-Workspace`

下列内容仍完整保存在移动硬盘，但不会上传 GitHub：

- `RUNTIME`、`.runtime`、`TOOLS`：可移植 Python、CUDA 依赖和便携 Git 工具；
- `DATASETS`、`data/datasets`、`data/firedata`：训练数据集；
- `RESULTS`、`MODEL_ARCHIVE`、`TRAINING_OUTPUTS`、`runs`：模型权重、审核归档和训练输出；
- 所有 `.pt`、`.onnx`、`.engine` 等大模型文件；
- 运行日志、用户设置和临时检查报告。

这是必要限制，不是遗漏。GitHub 普通仓库拒绝单个超过 100 MB 的文件；把十几 GB 的便携 Python 和数据集塞进 Git 也会让仓库无法正常克隆。大文件继续由移动硬盘及用户现有备份保存。

## 第一次启用

只需依次双击：

1. `INSTALL_GIT_TOOLS.cmd`：把官方便携 Git 和 GitHub CLI 安装在本工作库的 `TOOLS` 中，不修改公共电脑；
2. `ENABLE_GIT.cmd`：创建本地 Git 仓库和第一次提交；
3. `SYNC_GITHUB.cmd`：第一次会打开浏览器进行 GitHub 授权，成功后自动推送到私有仓库。

工作库已经完成 Git 初始化并补齐 10 个官方 YOLOv5 v7.0 目标检测权重。日常快速检查不会遍历十几 GB 数据集，也不会生成全库 SHA-256 清单。

## 以后怎么用

- 完成一次代码修改后，双击 `SAVE_VERSION.cmd`。它会自动创建带时间的版本；已经登录 GitHub 时也会自动推送。
- 如果希望自己填写备注，双击 `SAVE_VERSION_WITH_MESSAGE.cmd`，输入本次修改内容后按回车。
- 需要主动上传或从 GitHub 对齐时，双击 `SYNC_GITHUB.cmd`。
- 想查看当前修改和最近版本时，双击 `GIT_STATUS.cmd`。
- 数据集、训练结果和模型权重不会因 Git 操作被删除或覆盖。

## 撤销或删除最近版本

- `REVERT_LAST_COMMIT.cmd`：推荐使用。它不会删除历史，而是创建一个新的反向提交。再次运行可以撤销这个反向提交，例如 `A → B → C → R1 → R2`，其中 `R2` 会恢复 `C` 的内容。
- `DELETE_LAST_COMMIT_HARD.cmd`：危险操作。它使用 `reset --hard HEAD~1` 删除最后一次提交；如果 GitHub `main` 正好指向同一提交，也会使用精确的 `force-with-lease` 将远端退回。脚本只允许在工作区完全干净时执行。

两个脚本都需要两次确认：先输入操作关键字，再输入窗口显示的当前 commit 短 ID。任何输入不匹配都会取消，并且不修改仓库。

在公共电脑首次同步 GitHub 时也需要浏览器授权。登录令牌只由该电脑的 Windows 凭据系统保存，不写入移动硬盘；离开公共电脑前应退出 GitHub 账户并清除该电脑凭据。即使不登录 GitHub，`SAVE_VERSION.cmd` 仍会把版本安全保存在 E 盘的本地 Git 历史中。
