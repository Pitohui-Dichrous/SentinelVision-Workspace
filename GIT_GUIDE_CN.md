# SentinelVision Git / GitHub 使用说明

## 已采用的安全结构

Git 负责保存源码、界面、训练脚本、配置、文档和模型结构 YAML。GitHub 仓库为私有仓库：

`https://github.com/Pitohui-Dichrous/SentinelVision-Workspace`

下列内容仍完整保存在移动硬盘，但不会上传 GitHub：

- `RUNTIME`、`.runtime`、`TOOLS`：可移植 Python、CUDA 依赖和便携 Git 工具；
- `DATASETS`、`data/datasets`、`data/firedata`：训练数据集；
- `RESULTS`、`MODEL_ARCHIVE`、`TRAINING_OUTPUTS`、`runs`：模型权重、审核归档和训练输出；
- 所有 `.pt`、`.onnx`、`.engine` 等大模型文件；
- 运行日志、用户设置和完整性清单。

这是必要限制，不是遗漏。GitHub 普通仓库拒绝单个超过 100 MB 的文件；把十几 GB 的便携 Python 和数据集塞进 Git 也会让仓库无法正常克隆。大文件由 E 盘备份和 `INTEGRITY` SHA-256 清单保护。

## 第一次启用

只需依次双击：

1. `INSTALL_GIT_TOOLS.cmd`：把官方便携 Git 和 GitHub CLI 安装在本工作库的 `TOOLS` 中，不修改公共电脑；
2. `ENABLE_GIT.cmd`：创建本地 Git 仓库和第一次提交；
3. `SYNC_GITHUB.cmd`：第一次会打开浏览器进行 GitHub 授权，成功后自动推送到私有仓库。

在 D 盘应用当前补丁到 E 盘时，可直接双击 `APPLY_GIT_TO_E_DRIVE.cmd`，它会顺序完成上述工作，并补齐 10 个官方 YOLOv5 v7.0 目标检测权重。最后会重建整个工作库的 SHA-256 完整性清单；由于需要读取十几 GB 和大量小文件，首次执行可能耗时几十分钟，中途不要拔盘。

## 以后怎么用

- 完成一次代码修改后，双击 `SAVE_VERSION.cmd`。它会自动创建带时间的版本；已经登录 GitHub 时也会自动推送。
- 需要主动上传或从 GitHub 对齐时，双击 `SYNC_GITHUB.cmd`。
- 想查看当前修改和最近版本时，双击 `GIT_STATUS.cmd`。
- 数据集、训练结果和模型权重不会因 Git 操作被删除或覆盖。

在公共电脑首次同步 GitHub 时也需要浏览器授权。登录令牌只由该电脑的 Windows 凭据系统保存，不写入移动硬盘；离开公共电脑前应退出 GitHub 账户并清除该电脑凭据。即使不登录 GitHub，`SAVE_VERSION.cmd` 仍会把版本安全保存在 E 盘的本地 Git 历史中。
