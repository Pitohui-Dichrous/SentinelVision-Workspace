# YOLOv5 v7.0 官方目标检测预训练权重

本目录中的 `.pt` 均来自 Ultralytics YOLOv5 官方 `v7.0` GitHub Release，并在加入工作库前核对了文件大小和 SHA-256。

这些文件是基于 COCO 80 类训练的“迁移学习起点”，不是 SentinelVision 的正式检测模型。新训练结果仍然只进入 `TRAINING_OUTPUTS`，完成人工审核后才可部署到 `RESULTS`。

## 标准模型（P5，推荐 640px）

| 型号 | 特点 | 适合用途 |
|---|---|---|
| `yolov5n.pt` | 最快、占用最低、精度基础 | 快速试验、小型数据集、低延迟 |
| `yolov5s.pt` | 速度与精度均衡 | 默认推荐，先验证数据集和参数 |
| `yolov5m.pt` | 更高精度、中等资源 | 正式进阶训练 |
| `yolov5l.pt` | 精度优先、资源较高 | 已确认数据质量后的精度实验 |
| `yolov5x.pt` | 标准系列最大、最慢 | 高成本精度实验 |

## 高分辨率模型（P6，高级，建议 1280px）

`yolov5n6.pt`、`yolov5s6.pt`、`yolov5m6.pt`、`yolov5l6.pt`、`yolov5x6.pt` 面向大图和小目标，但训练速度更慢、显存占用明显更高。RTX 4080 Laptop 不建议使用 `yolov5x6.pt`；在 RTX 4090 上也应保留 `batch=-1` 让程序自动估算。

## 为什么没有为 models 下每个 YAML 提供权重

- `models/segment/*.yaml` 是实例分割结构，必须使用分割数据集和 `segment/train.py`，不能用于当前目标检测训练页。
- `models/hub/yolov3*.yaml` 属于旧版 YOLOv3，不是 YOLOv5 权重。
- `yolov5-fpn`、`bifpn`、`panet`、`ghost`、`transformer` 等实验 YAML 没有官方同名预训练 checkpoint。
- `anchors.yaml` 只是锚框配置，不是模型。

YAML 只描述网络结构，不能凭空生成已经训练好的 `.pt`。工作库不会伪造或错误改名权重。

## SHA-256

```text
yolov5n.pt   4F180CF23BA0717ADA0BADD6C685026D73D48F184D00FC159C2641284B2AC0A3
yolov5s.pt   8B3B748C1E592DDD8868022E8732FDE20025197328490623CC16C6F24D0782EE
yolov5m.pt   61D933360BA5A7733A36764996C800287D973889D875227F5BEEDD2473A97A56
yolov5l.pt   2F603B7354C25454D1270663A14D8DDC1EEA98E5EEBC1D84CE0C6E3150FA155F
yolov5x.pt   9F27A794FA0308E2606F90565164571D6F0A0BA18A3CE2E5E5D323B71C157859
yolov5n6.pt  496C05A2B991DA15ACC6C4408C63DA287F2513A46C6756618776D4DD71170781
yolov5s6.pt  95BDA9019ADA63F37338308D0C81A66047A6FBABA061ABAFF271BA9334CF2A6F
yolov5m6.pt  7AFE7FA0F29A8351467200A2A5A33C7996D1155ED6636C7EDD8CA70B14951C64
yolov5l6.pt  B357B77F646B0190912985937C44E2923441369D5FDA69BA86DA2AA919212618
yolov5x6.pt  257C991B216D625427DC4D9C7C76D6C6D93CF3732797538DA34BDC02F97BFEF0
```
