# CLIP 与 Grounding DINO 并列后端

两个主配置默认选择 CLIP；每次运行只加载所选识别器，不会同时运行两套模型。Places365 仍是可选分类器。

## 启动与切换

保留现有仿真命令的速度、模型、场景和控制参数，仅替换以下选项：

```bash
--room-memory --room-semantic-backend clip
```

或：

```bash
--room-memory --room-semantic-backend groundingdino
```

修改后重启自己的仿真终端 B，网页 Ctrl+Shift+R；LightNav / OmniVLA 模型终端 A 无需重启。无需安装依赖或另启语义服务。

| 用途 | 默认配置文件 |
| --- | --- |
| Isaac | `isaac_demo/config/room_memory.json` |
| Habitat | `habitat_demo/room_memory/config/default.json` |

自定义 `--room-memory-config` 时修改该参数指定的文件。`semantic.backend` 是未传 CLI 时的默认后端。

## 并列代码与配置

| 职责 | CLIP | Grounding DINO |
| --- | --- | --- |
| 识别器及常驻子进程 | `clip_recognizer.py` | `groundingdino_recognizer.py` |
| 采样、推理及投影调度 | `clip_projection.py` | `groundingdino_projection.py` |
| 独立识别配置示例 | `config/clip.json` | `config/groundingdino.json` |
| 主配置里的识别参数 | `semantic_backends.clip` | `semantic_backends.groundingdino` |
| 主配置里的投影参数 | `semantic_projection.backends.clip` | `semantic_projection.backends.groundingdino` |
| 默认环境 | `/data/disk1/xrq/.conda/envs/omnivla` | `/data/disk1/xrq/.conda/envs/groundingdino` |
| 默认推理设备 | CPU，2 线程 | CUDA，`cuda:0` |

`projection_worker.py` 只负责选择后端；`semantic_projection.py` 共用几何投影和多类别证据融合；`service.py` 共用 session、节点语义及实时识别接口。没有修改 LightNav、MPC 或 Portal 检测核心逻辑。

独立识别示例 JSON 不包含完整的几何和门检测配置。正常仿真请继续使用上述主配置，只在参数中选后端。

## 自定义类别和提示词

CLIP：修改主配置 `semantic_backends.clip.labels` 和 `templates`。恢复的 8 类为 bedroom、living room、bathroom、kitchen、dining room、office room、gym、corridor。

DINO：保留用户在 `groundingdino_recognizer.py` 的 `DEFAULT_LABELS` 修改，目前为 bedroom、living room、bathroom、kitchen、dining room、corridor。也可以在主配置 `semantic_backends.groundingdino.labels` 中填写列表覆盖默认值，句点分隔的 prompt 由列表生成。进行公平类别对比时应自行将两套 labels 设置一致。

兼容旧 JSON 的 `semantic` 参数写法：相同后端时，`semantic` 中的显式参数优先于命名配置。CLI 切换到另一个后端时丢弃旧后端专属参数，选取对应 `semantic_backends` 配置；仅保留旧 `live_enabled/live_interval` 公共设置，避免环境、模型路径、阈值串用。

## 投影与页面

CLIP 默认 `view_mode=full`，整图保持完整视野后分类。保留 `left_right` 左右子图实现，但默认不运行。投影前对每个视图过滤：Top1 小于 0.35 **或**前三名最大最小分差小于 0.10 时跳过；不擦除已有可信颜色。参数在 `semantic_projection.backends.clip`。

DINO 固定 `view_mode=detections`，按 bbox 范围及其深度投影，候选使用 `semantic_backends.groundingdino.box_threshold`（0.25）过滤，不使用 CLIP Top3 过滤。第一张画面显示检测源帧及框、类别、分数；CLIP 时恢复实时原始画面及实时分类结果。两者都显示房间语义地图，页面标明当前后端。

两种投影都绑定采集时刻 RGB、Depth、位姿、时间及长期几何快照，投到长期房间范围图，不取当前位姿、不借短期导航地图半径裁剪，也不依赖 Portal。保留 session、轨迹、代表图和完整多类别证据。详见 [投影与融合](SEMANTIC_PROJECTION.md)。

## 单图对比

在 Habitat 目录运行（输入 JPEG；多个图片参数表示多视图）：

```bash
cd /data/disk1/xrq/projects/LightNav-0/habitat_demo
/data/disk1/xrq/.conda/envs/omnivla-habitat/bin/python -m room_memory.compare_semantics \
  /path/to/image.jpg \
  --config room_memory/config/default.json \
  --backends clip groundingdino \
  --output runs/semantic-comparison/result.json
```

也可指定 `--backends clip places365`。宿主进程自动调用各自环境；两种模型分数定义不同，不能把分数高低直接当成准确率高低。
