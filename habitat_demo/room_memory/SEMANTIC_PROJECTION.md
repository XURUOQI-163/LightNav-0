# CLIP / Grounding DINO 房间语义证据图

CLIP 与 Grounding DINO 现在并列保留，主配置默认 CLIP。启动参数及代码结构见 [后端切换](BACKENDS.md)。二者共用长期几何投影和证据融合，识别及采样调度独立。

## CLIP 整图分类

CLIP 复用 omnivla 环境的常驻子进程，默认 CPU 2 线程。整图 letterbox 成正方形以避免预处理中心裁剪丢失两侧视野；保留左右子图模式 `left_right`，默认 `full`。每个视图输出完整类别分布。Top1 低于 0.35 **或**前三名分差小于 0.10 则跳过投影，不清空历史证据。参数均可在 `semantic_projection.backends.clip` 修改。

通过原视图对应的深度、拍摄位姿和推理前冻结的长期几何生成投影区域；融合完整类别分布及余弦平方视角权重。每格保存多类证据，以历史累计权重与当前权重更新，历史权重上限由 `max_weight` 控制。颜色取融合后的 Top1，不由 Room ID 决定。

## Grounding DINO 区域分类

使用官方 Swin-T OGC 配置和预训练权重，独立 groundingdino 环境的 CUDA 子进程推理。当前保留用户的六类提示：bedroom、living room、bathroom、kitchen、dining room、corridor。Places365 仍可独立选择。

## 输入输出和分数

`groundingdino_recognizer.py` 可单独测试一张 RGB 图片。官方预处理缩放后，网络输出 normalized cxcywh；适配器统一还原为原图像素 `bbox_xyxy`、`center_pixel`，并输出 `room_type`、`score` 和各类别 `class_scores`。每个 query 对提示短语各 token 的 sigmoid 分数求平均，选最高类别，按 `semantic_backends.groundingdino.box_threshold`（默认 0.25）筛选。不是类别 softmax，也不是校准正确率；保留所有达标候选框，不做 NMS。

Grounding DINO 是开放词汇区域检测器；房间词可能对应整幅画面、局部家具或没有输出，不保证得到真实房间分割。无达标框时不产生颜色，不强行选择房间。房间节点的多视图摘要使用每帧每类最高候选分数再跨帧平均；probabilities 为该摘要归一化结果，confidence 保留原始最高匹配分数。

## 同步、投影、融合

每个检测样本复制同一时刻 RGB、Depth、时间戳、相机内外参和 PoseSource 位姿。推理前冻结长期几何网格快照，推理后使用原始位姿和冻结几何计算投影，禁止取当前视角替代。队列只保留一个待处理最新帧；重置 session 后丢弃旧结果。

每个 bbox 掩蔽其外部的深度（横纵方向都限制），复用长期 StructuralMapBuilder 的高度范围和每列最远有效深度。覆盖对应水平角度的扇形区域，相邻有效采样列之间补射线以覆盖远处格子；不跨无效深度间隙、不穿已知墙，不给长期几何未覆盖的格子上色。没有独立最远半径，与短期局部导航地图半径无关。注意：这是检测框方向在可见几何上的证据投影，不是由 bbox 推断完整房间边界。

每个候选以其类别产生一票；票重为检测 score × 框中心方向的余弦平方角度权重。重叠候选先在同一帧聚合，单格每帧权重不超过 1，再与历史各类别证据加权融合，历史权重上限 `max_weight=20`。存储每格完整类别分布，不只保存 Top1。无新证据不清除旧颜色。没有 Portal 同样能工作，Portal/Room ID 不决定语义涂色。

## 页面、配置与输出

启动参数 `--room-memory --room-semantic-backend groundingdino`；默认 JSON 选择 CLIP，此参数显式切换至 DINO。第一张原始 RGB 图展示完成检测的原图与对应框、类别、分数、采集时间；等待下一次检测时保留这一帧，绝不把旧框叠到新的实时图。第三视角、模型输入与预测点保持原功能，模型输入未添加框。

`semantic_projection.interval=1.5` 控制检测采样间隔；`enabled=false` 仅停地图投影，图像检测仍运行。`regions.enabled=false` 同样保留检测。DINO 的 `view_mode=detections`，CLIP 默认 `view_mode=full`。旧 Top3 softmax 过滤不用于此后端，默认 `filter_low_confidence=false`，用 detector 的 `box_threshold` 控制。

`/api/room-memory/detection.jpg` 提供完整原始检测帧；`X-Frame-Id` 与 `X-Sim-Time` 标记采集来源，支持 after 返回 204。`/api/room-memory` 的 `semantic_map` 与 `live_semantic` 报告检测框、状态、源时间和耗时。session 目录保存 `room_detections.json`（最新帧、位姿、框）、`semantic_map.npz`（稀疏多类证据和权重）、`semantic_map.json`，保留原 graph/session/代表图功能。

官方模型来源：https://github.com/IDEA-Research/GroundingDINO
