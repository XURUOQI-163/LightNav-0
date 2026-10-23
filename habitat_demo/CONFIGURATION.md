# 参数与接口

## 环境/页面：`python -m habitat_web`

| 参数 | 默认 | 意义 |
| --- | --- | --- |
| `--scene` | 必填 | 本地 .glb，需配套可通行 navmesh |
| `--scene-dataset-config` | 不指定 | 数据集 `.scene_dataset_config.json`；HM3D 需提供配套配置，加载正确的坐标方向和材质。省略时保留原测试场景行为 |
| `--host` | 127.0.0.1 | 网页绑定地址；远程直接访问用 0.0.0.0 |
| `--port` | 8090 | 网页端口 |
| `--lightnav-url` | ws://127.0.0.1:8050 | 原 LightNav 服务 |
| `--omnivla-url` | http://127.0.0.1:8891 | 本目录纯 OmniVLA 服务；不要指向旧复杂 dashboard |
| `--seed` | 0 | 启动时的可通行起点采样种子 |
| `--dt` | 1/3 | 每轮动作的仿真秒数，范围 (0,1] |
| `--max-linear` | 0.15 | 线速度上限 m/s |
| `--max-angular` | 0.5 | 角速度上限 rad/s |
| `--max-steps` | 0（不限） | 每次开始后的最大动作次数；0 表示不限步数，正整数才启用上限 |
| `--timeout` | 30 | 模型请求墙钟超时秒数 |
| `--width`, `--height` | 640,480 | 相机分辨率 |
| `--camera-height` | 0.45 | 相机距 agent 原点高度 m |
| `--hfov` | 90 | 水平视场角，度 |
| `--room-memory` | 关闭 | 开启同步前向 RGB-D + GT pose 的被动房间记忆与 2D 页面 |
| `--room-memory-config` | 内置默认 | JSON 文件，示例 `room_memory/config/default.json`，配置 memory/geometry/semantic/doorway/regions |
| `--room-memory-output` | runs/room-memory | 每个仿真回合的 graph.json、structure.npz、采样 trajectory.jsonl 和每房代表图 |
| `--room-semantic-backend` | 按 JSON，default.json 为 clip | `clip` / `groundingdino` / `places365`，选择房间识别器；需要同时启用 `--room-memory` |

GET `/api/state` 返回状态和最近预测；`latency_ms` 是图像请求往返，`model_latency_ms` 是模型服务报告的推理时间，页面展示后者。GET `/api/camera.jpg` 只读取最新 JPEG。POST `/api/start` 接收 `{backend:"lightnav"或"omnivla", instruction:"..."}`；`/api/stop`、`/api/reset` 接收 `{}`。显示和轮询不会驱动环境。`stop_reason` 区分 `model_stop`（模型停止）、`empty_path`（无路径）、`zero_command`（零速度）、`step_limit`（显式步数上限）、`user_stop`（手动停止）和 `error`；开始/重置时清空。停止不等于目标已找到。

开启房间记忆后 GET `/api/room-memory` 返回近期轨迹、房间节点、门户、语义状态、候选和丢帧计数；`/api/room-memory/export` 下载当前 JSON 快照；`/api/room-memory/frames/F0.jpg` 返回该回合的代表视角。关闭时状态接口返回 `enabled:false`。快照不含原始 depth/点云。

房间默认参数：轨迹位移 0.05 m / 旋转 5° / 最长间隔 2 s；默认视觉模式第一次可信 RGB+Depth 观测即建立蓝色门口，实际跨平面超过两侧 0.20 m 后转红。视觉开口宽度 0.75–2.80 m，连续可通行宽度至少机器人宽 0.60 m 加裕度 0.10 m。`memory.portal_confirm_frames=2` 只控制显式关闭视觉检测后的旧纯深度模式，不要求视觉门口等待两帧。

### YOLOE-26s 门口配置

`doorway.enabled` 默认 `true`；改为 `false` 使用原深度检测模式。关闭整个房间记忆仍然只需去掉 `--room-memory`。`doorway.queue_size=2` 限制等待视觉检测的 RGB-D 帧数，过载丢旧帧，导航不等待 YOLOE。

| 字段 | 默认 | 意义 |
| --- | --- | --- |
| `doorway.detector.python_executable` | `/home/xrq/.conda/envs/navigation/bin/python` | 已有 Ultralytics 推理环境，不合并到 Habitat |
| `model_path` | `.models/yoloe/yoloe-26s-seg.pt` 的本地绝对路径 | 官方 YOLOE-26s 权重 |
| `prompt_cache_path` | `.models/yoloe/opening-prompts.pt` 的本地绝对路径 | 与权重、labels、Ultralytics 版本匹配的文本嵌入缓存 |
| `labels` | doorway / open doorway / door opening / door frame / open door / door | RGB 提议词；修改后必须重新 `--prepare` |
| `device` / `cpu_threads` | cpu / 2 | 默认不占导航 GPU；GPU 编号遵循 CUDA_VISIBLE_DEVICES |
| `confidence` / `imgsz` | 0.25 / 640 | 检测置信分数门槛和输入尺寸，不代表校准后的概率 |
| `timeout_seconds` / `startup_timeout_seconds` | 45 / 240 | 单帧和首次加载超时；失败显示在网页 |
| `max_detections` | 12 | 每帧 RGB 提议上限 |
| `doorway.depth.depth_min` / `depth_max` | 0.2 / 15 m | 当前 Habitat 有效轴向深度范围；实车须改为真实传感器量程 |
| `portal_min_width` / `portal_max_width` | 0.75 / 2.8 m | 深度估计开口宽度范围 |
| `clearance_depth` | 0.35 m | 观察射线须延伸至门后多远 |
| `valid_fraction` / `free_fraction` | 0.65 / 0.65 | 局部采样列有效深度及自由证据比例 |
| `memory.portal_match_distance` / `portal_match_angle_deg` | 0.55 m / 30° | 世界坐标去重位置及法向阈值 |
| `portal_update_alpha` / `portal_update_freeze_distance` | 0.2 / 0.6 m | 位姿测量融合权重及接近门口时的冻结距离 |

其余视觉深度参数在 `doorway.depth` 中，包括 ROI、采样数、墙条拟合误差与角度。`geometry` 的旧双侧门框/profile 参数仅影响辅助证据，不能否决已经通过视觉深度验证的开口。门槛被画面底边截断或未观察到足够可通行宽度时暂不建立节点。

JSON `schema_version=3`：保留 v2 节点/门户字段，并增加房间区域与 `structure` 诊断。`rooms` 是已进入节点，`prospective_rooms` 是未进入候选；门户 `status` 为 `unexplored`（蓝）或 `explored`（红），另有观测次数、首次/末次看到时间、穿越次数、辅助几何支持。`doorway_status/error/diagnostics/dropped_frames` 反映检测状态、最近源时间戳和拒绝原因。红色不代表搜索完成；候选语义带问号，Places365 不决定拓扑。

### 房间范围配置

`regions.enabled=true` 默认随房间记忆启用。设为 `false` 后重启 Habitat，只停止区域构建，节点/门口/轨迹/Places365 保持。页面“显示房间范围”只控制绘制，不停止后台计算。

| `regions` 字段 | 默认 | 意义 |
| --- | --- | --- |
| `cell_size` / `grid_size` | 0.05 m / 1024 | 初始位姿周围约 51.2 m 方形；尺寸 32–2048，分辨率 0.02–0.5 m |
| `update_interval` | 0.75 s | 结构更新最短仿真时间间隔；Portal 改变可能额外重分区 |
| `depth_min` / `depth_max` | 0.2 / 6 m | 结构模块有效轴向深度，独立于门口验证的 15 m 上限 |
| `sample_columns` / `sample_rows` | 160 / 64 | 最多采样列/行；上限 320 / 160 |
| `height_min` / `height_max` | 0.1 / 2.4 m | 机器人局部高度带 |
| `wall_min_span` / `wall_min_top` | 0.65 / 1.2 m | 墙候选垂向跨度和顶部高度 |
| `wall_bin_size` / `wall_min_length` | 0.12 / 0.35 m | 水平分组和邻近连续支持尺度 |
| `wall_confirm_frames` | 2 | 同一栅格重复墙体支持次数，按结构更新帧计数 |
| `wall_padding` / `portal_padding` | 0.1 / 0.12 m | 分割时墙体留边/门口两端延伸，不影响导航碰撞 |
| `seed_radius` | 0.3 m | 入口/轨迹种子寻找附近观测格的最大坐标偏移 |
| `floor_tolerance` | 0.35 m | 超出初始高度时暂停区域更新并提示；不控制机器人上楼 |
| `min_region_area` | 0.02 m² | 导出片区面积下限 |

邻域循环还有计算量限制：墙分组半径最多 8 个 bin，墙体留边最多 8 格，种子寻找最多 32 格，portal_padding ≤ 0.5 m；超限启动时明确拒绝。详细算法、字段和局限见 [ROOM_REGIONS.md](room_memory/ROOM_REGIONS.md)。

`structure.status` 包括 `waiting`、`ready`、`disabled`、`insufficient_depth`、`floor_mismatch`、`capacity_reached`。`structure.error` 是错误文本；`build_ms` 为最近一次范围更新耗时，`update_count` 为更新次数，不是导航模型延迟。房间增加 `region_polygons`、`boundary_segments`、`region_area_m2`、`boundary_state`、`geometry_confidence`。面积是已记录范围；边界证据比例不是完整度。

已有本地权重和缓存可直接启动，无需新终端或额外端口。单图检查：

```bash
cd ~/projects/LightNav-0/habitat_demo
/home/xrq/.conda/envs/navigation/bin/python -m room_memory.doorway_detector \
  --config room_memory/config/default.json --image /实际路径/原始RGB.jpg
```

该命令只输出 RGB 检测框，不代表已经通过深度通行验证。权重/缓存缺失不会自动下载、装包或替换模型。准备过程与环境说明见 [room_memory/VISUAL_PORTALS.md](room_memory/VISUAL_PORTALS.md)。

语义默认本地 CLIP，也可选择 Grounding DINO-T 或 Places365，无需 API key。房间节点保留多视图归纳：最多 5 个不同视角，至少 3 帧时分类，离开或停止时允许不足 3 帧，每次访问最多一次。另有当前画面的实时单帧识别，在首次观测和后续导航新帧到达时运行；相机没有新帧时不重复推理。两者共用同一个常驻模型，导航不等待分类结果。停止导航保留图，重置场景新建 session，旧图保留在输出目录。

### Places365 语义配置

默认 JSON 使用 CLIP；`--room-semantic-backend groundingdino` 或 `places365` 切换后端。模型路径、类别和阈值在 `semantic_backends.<后端>` 下分开配置，投影在 `semantic_projection.backends.<后端>` 下配置。兼容旧 semantic 参数写法，详见 [并列后端](room_memory/BACKENDS.md)。

`room_memory/config/default.json` 的 `semantic` 字段：

| 字段 | 默认 | 意义 |
| --- | --- | --- |
| `backend` | `places365` | 使用本地场景分类器 |
| `repo_path` | `/home/xrq/projects/places365` | 包含 `resnet18_places365.pth.tar` 和 `categories_places365.txt` 的目录 |
| `python_executable` | `/home/xrq/.conda/envs/places365/bin/python` | 常驻推理子进程使用的解释器 |
| `mode` | `subprocess` | 自动启动另一个环境的推理子进程；`in_process` 使用当前解释器中的工作线程 |
| `device` | `cpu` | 支持 `cpu`、`cuda`、`cuda:N`；GPU 编号遵循可见设备环境变量 |
| `num_threads` | `2` | PyTorch CPU 线程数，1–32 |
| `timeout` | `60.0` | 单次请求超时秒数，首次请求包含加载权重；子进程超时/取消时退出 |
| `mapping_path` | `null` | 仅保留给 CLI 的旧映射诊断，不再影响预测类别或分数 |
| `live_enabled` | `true` | 在相机下方显示当前图像识别；设 false 只关闭实时识别，保留房间多视图分类 |
| `live_interval` | `1.0` | 实时识别最小启动间隔，单位为实际时间秒；推理较慢时只保留最新一帧待处理 |

成功识别时直接保留原始 365 类 Top-1，不做房间类型映射，也不按分数过滤。当前画面使用单帧概率；房间节点先对多视图的原始 365 类概率取均值，再取 Top-1。两者可能不同。`room_type_probabilities` 现在使用原始 Places365 类别名，`semantic_confidence` 是该 Top-1 的实际分数；代表图选择对最终类别支持最高的一帧。分数未校准，不能当作房间分割正确率。旧配置 `memory.semantic_min_confidence` 仍可读取但已不再生效，默认 JSON 已去掉它。类别只更新语义，不创建或切换 Room ID。

网页相机下方的“当前画面 · Places365”显示 Top-1、分数、源图像仿真时间、图像年龄、识别耗时（含共享分类器等待/首次加载）、完成次数、处理状态或错误；节点详情显示多视图结果。等待/失败时不伪造类别或显示 0 分。`GET /api/room-memory` 的 `live_semantic` 字段包含这些实时信息和 `top5`，不保存额外原始图像；`graph.json` 继续保存房间级结果。实时状态的 `source_timestamp` 是结果对应的观测时间，不是网页最新图像的保证。默认每秒最多启动一次，实际频率还受新图像和推理耗时限制。

LightNav 显示 APOS/OPOS 时，网页保留该预测的输入图像；实时识别也跟随这张显示图像，不会提前改用运动后的下一帧。几何记忆仍正常接收最新 RGB-D 和位姿。重复显示同一张图不会重复提交识别。

单独运行识别器时也直接输出 Top-1，同时附带 `places365_top5`；`unmapped_probability` 仅用于对照旧映射，不参与分类。修改代码或配置后，重启 Habitat 并刷新网页；旧 session 的 `unknown` 记录不自动重写。

当前 Habitat 环境没有 torch/torchvision，不能直接选择 `in_process`。默认子进程模式使用已有 Places365 环境，不升级 Habitat 依赖；不下载权重，不开放额外 HTTP 端口。若以后使用同时具备兼容 Habitat、torch 和 torchvision 的环境，可选择 `in_process`，此时忽略 `python_executable`。线程内推理超时后底层计算可能仍在运行，关闭时会等待它完成。

旧千问适配器仍可选：将整个 `semantic` 对象替换为 `{"backend":"qwen","model":"qwen-vl-plus","base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","timeout":20.0}`，并在启动终端设置 `DASHSCOPE_API_KEY`。Places365 默认配置不读取或要求该密钥。

### CLIP 配置

`--room-semantic-backend clip` 为主配置默认选项。`semantic_backends.clip` 下设置 `labels`、`templates`、`model_path`、`python_executable`、`device`（cpu）和 `num_threads`（2）。`semantic_projection.backends.clip` 默认 `view_mode=full`，保留可选 `left_right`；Top1 < 0.35 或 Top3 分差 < 0.10 跳过投影，已有证据保留。参数名为 `filter_low_confidence`、`low_top1_threshold`、`top3_spread_threshold`。界面恢复原始实时图及当前画面的 CLIP 分类。

### Grounding DINO-T 配置

`--room-semantic-backend groundingdino` 选择独立环境中的官方 Swin-T OGC；默认 JSON 为 CLIP，传此参数切换到 DINO。第一张原始图显示完成检测的源帧及其框/分数，地图投影用同一帧 RGB-D 和位姿。未检测到候选时不强制分类。

| semantic_backends.groundingdino 字段 | 默认 | 说明 |
| --- | --- | --- |
| `python_executable` | `~/.conda/envs/groundingdino/bin/python` | 独立 GPU 子进程，主仿真环境不安装 torch |
| `model_path` | 同级 GroundingDINO/weights/groundingdino_swint_ogc.pth | 官方 Swin-T 权重 |
| `config_path` | 同级 GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py | 官方配置 |
| `bert_path` | 同级 GroundingDINO/weights/bert-base-uncased | 离线文本编码器 |
| `device` / `num_threads` | `cuda:0` / `2` | CUDA 编号遵守当前进程 CUDA_VISIBLE_DEVICES |
| `box_threshold` | `0.25` | 候选短语 token sigmoid 均分阈值 |
| `timeout` | `120` | 子进程请求秒数，首次含加载 |
| `labels` | 用户保留的 6 类 | bedroom、living room、bathroom、kitchen、dining room、corridor |

检测间隔使用 `semantic_projection.interval=1.5`；`semantic_backends.<后端>.live_enabled/live_interval` 用于 CLIP / Places365 的单图分类通道。DINO 的检测与地图共用一个同步 RGB-D 通道。`semantic_projection.enabled=false` 可关闭投影但保留框；关闭整个功能需移除 `--room-memory`。更多输出接口、坐标及融合说明见 [语义投影](room_memory/SEMANTIC_PROJECTION.md)。

独立测试（无需启动仿真）：

```bash
cd ~/projects/LightNav-0/habitat_demo
~/.conda/envs/groundingdino/bin/python -m room_memory.groundingdino_recognizer /path/to/image.jpg --output runs/detection.json --annotated runs/detection.jpg
```

`compare_semantics --backends clip groundingdino` 可对比两种后端（默认组合）；也支持 `--backends clip places365`。两者分数不代表可比较的正确率。CLIP 与 DINO 代码及配置并列保留。重启仿真终端并强制刷新网页，导航模型服务无需重启。

## 纯 OmniVLA：`python -m habitat_web.omnivla_service`

| 参数 | 默认 | 意义 |
| --- | --- | --- |
| `--repo` | 必填 | 原 OmniVLA 项目根目录 |
| `--checkpoint` | 必填 | 完整模型本地权重目录 |
| `--resume-step` | 120000 | 动作头/位姿投影器的 checkpoint step |
| `--host` | 127.0.0.1 | HTTP 绑定地址 |
| `--port` | 8891 | HTTP 端口 |

GET `/health` 返回模型就绪；POST `/predict` 接收 JPEG base64、prompt、session_id，返回原始 Nx4 waypoints、latency_ms、stop:false。POST `/reset` 确认会话边界（当前纯模型无会话历史）。单个模型串行推理，忙时返回 503。不会启动 ROS、检测器或拓扑模块。
