# SENTINEL 已部署模型区

`RESULTS` 是 SENTINEL 唯一会扫描的部署边界。`TRAINING_OUTPUTS` 和 `runs` 中的任何权重都不会被自动加载。

## 动态扫描规则

每次启动时，SENTINEL 会读取 `model_registry.yaml`，并扫描：

- `RESULTS/*.pt`
- `RESULTS/<模型>/best.pt`
- `RESULTS/<模型>/weights/best.pt`

`last.pt`、隐藏目录、`_STAGING`、备份和禁用目录会被忽略。登记模型会核对文件大小和 SHA-256；不一致的模型会被隔离，但不会阻止其他健康模型启动。

现有三个历史沿用模型为：

- `combined` / COMBINED：`person, hat, fire`
- `fire` / FIRE ONLY：`fire`
- `helmet_only` / HELMET ONLY：`person, hat`

SENTINEL 不再使用 SINGLE/DUAL。用户直接勾选一个或多个模型；FIRE ONLY + HELMET ONLY 等价于原来的 DUAL。

## 正式部署流程

1. 新训练统一输出到 `TRAINING_OUTPUTS`。
2. 检查指标、类别顺序、失败样本和真实视频。
3. 在工作台“审核与部署”页填写类别含义和告警策略。
4. 用户完成四项审核并点击最终批准。
5. 程序暂存复制、核对 SHA-256，再写入 `RESULTS` 和登记表。
6. 替换模型的旧版本自动移入 `MODEL_ARCHIVE`。

统一类别（例如 `fire`）的名称、颜色和告警等级是全局规则。新模型映射到已有统一类别时会沿用现有规则，不会静默关闭或改写其他已部署模型的告警。

也可以人工放入可信的 `best.pt`。这类文件会显示为“新发现”、默认不勾选，且未登记类别默认不告警。
