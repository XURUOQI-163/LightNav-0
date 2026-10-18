# CLIP 多类别房间语义证据图

本层独立于 Room ID 和 Portal，在现有 StructuralMapBuilder 的固定世界网格上融合 CLIP 房间类别证据。它是视图级场景分数的空间投影，不是像素语义分割，也不自动切房间、建 Portal 或改变导航指令。Places365、房间多视图代表图和实时全图分类继续保留。

## 参考与区别

参考 ApexNav 的 [value_map2d.cpp](https://github.com/Robotics-STAR-Lab/ApexNav/blob/main/src/planner/plan_env/src/value_map2d.cpp)（视场余弦平方权重、分数融合）、[sdf_map2d.cpp](https://github.com/Robotics-STAR-Lab/ApexNav/blob/main/src/planner/plan_env/src/sdf_map2d.cpp)（深度射线支持的格子）和 VLFM 的 [value_map.py](https://github.com/rai-opensource/vlfm/blob/main/vlfm/mapping/value_map.py)（深度可见范围、坐标变换、多通道价值融合）。本实现自行实现，不移植 ROS、ESDF 或规划模块。

区别：这里保存固定房间类别的完整分数向量，使用累计且有上限的证据权重；不使用目标物体相似度。VLFM 使用每列最远深度构造视野轮廓，本层共用长期几何建图的 project_columns，采用每列最远有效水平距离，由原始帧深度和长期几何地图决定范围，并在已知墙体处终止，以降低穿墙、大面积涂色风险。不是逐行复现这些项目的融合公式。

## 数据链与同步

`RoomMemoryService.offer` 收到同步 RGB、depth、sim_time、pose、depth_camera 时复制数据，几何观察完成后交给独立 `SemanticProjectionWorker`。显示用 `observe_camera(jpeg, timestamp)` 没有深度/位姿，因此不进入投影。

每个样本保存原始时间戳、PoseSource 输出、相机内外参和深度。默认 view_mode=full：每个投影样本使用整张 RGB 图，只调用一次现有持久化 CLIP 子进程，语义作用于整幅视野的长期几何可见范围。整图按比例缩放并补灰边至正方形，避免 CLIP 隐式中心裁剪丢失边缘。原左右两幅完整高度子图设计保留为 view_mode=left_right，只有显式选择时才运行；该模式分别识别、投影对应半幅。类别固定为 bedroom、living room、bathroom、kitchen、dining room、office room、gym、corridor；不再包含 lounge 和 laundry room。

投影先生成该采集帧对应的世界格子索引及权重，然后等待分类结果。返回后直接使用这些固定索引，绝不查询当前机器人位姿重投影。队列只有一个可替换待处理样本，另有一个在途样本；旧 session 的结果、同一/更旧时间戳的重复结果丢弃。延迟结果可以更新其原始位置的历史证据，但不能覆盖当前视角。reset 清空网格并使旧请求失效，stop 保留已融合地图。

## 投影与融合

相机光学坐标使用轴向深度 Z，按 K 反投影，经 camera_to_robot 和采集时刻二维位姿转到固定世界 XY。深度有效范围、高度带、行列采样完全使用长期结构图的 regions 配置，与结构建图共用每列最远观测计算。每列有有效观测即可，不再使用语义层的最近深度，也不在终点前退缩 0.10 米。射线以半格步长行进，只对结构图已有 coverage 的非墙格子投票，在已知墙体处停止；无效深度、视野后方、超出量程和其他楼层不参与。Portal 不参与这一计算。

每个格子保留 `P[class]` 和 `weight`，不是仅保留 top1。角度权重为相对当前分类图像视场中心的 cos²（full 为整图中心，left_right 为各子图中心），不设额外投影半径或 3 米距离权重。同帧同格射线取最大权重，多个裁剪在同帧内先合并并将总权重限制在 1，避免像素密度造成重复投票。时间融合为：

`P_new = (W_old * P_old + w * p_view) / (W_old + w)`

`W_new = min(W_old + w, max_weight)`

该上限避免历史观测永久压住后续证据。网页显示 P 的 top1 固定颜色，以分数调节透明度，融合前先过滤低分或模糊的子图。CLIP softmax 是固定候选中的相对分数，不是校准过的正确概率。

每个分类视图分别判断：Top1 < `low_top1_threshold=0.35` 或 Top1 − Top3 < `top3_spread_threshold=0.10`，任一条件成立就丢弃整个视图的本次投影。只有分数不低且前三名并不接近时才可投影。新区域保持无色，已有可信证据保留。阈值作用于归一化候选类别分数，不是原始余弦相似度，仅用于过滤地图投影，不改变节点语义和实时分类展示。

邻近有效列之间按较短深度补充射线，避免远处稀疏条纹；不跨越采样列中的无效深度。投影域是长期结构图已有 coverage 的非墙格子，不依赖 R0/R1 分区或短期局部避障图，也不会把整个房间中当前不可见的区域直接刷成同一类。

## 配置与输出

`room_memory/config/default.json` 的 `semantic_projection` 段控制本层；只在 CLIP backend 和二维 regions 启用时运行，无需新增模型启动参数。

| 配置 | 默认 | 含义 |
|---|---:|---|
| enabled | true | 开关；Places365 不启动此层 |
| view_mode | full | 整图分类和投影；left_right 恢复保留的左右子图模式 |
| interval | 1.5 s | 两轮开始间隔（墙钟时间），不阻塞导航等待推理 |
| filter_low_confidence | true | 融合前启用低分或接近过滤 |
| low_top1_threshold | 0.35 | Top1 低分界限 |
| top3_spread_threshold | 0.10 | Top1−Top3 接近界限 |
| regions.depth_min / depth_max | 0.2 / 6 m | 共用长期几何图的深度有效范围 |
| regions.height_min / height_max | 0.1 / 2.4 m | 共用长期几何图高度带 |
| regions.sample_columns / sample_rows | 160 / 64 | 共用长期几何图采样 |
| max_weight | 20 | 每格历史融合权重上限 |

网页房间记忆面板增加“CLIP 语义投影”、固定颜色图例、状态、源时间戳和有证据格子数。启用时原房间范围为灰色底图，彩色像素来自语义证据；节点、轨迹、Portal 保留。关闭显示开关仅隐藏图层，不停止计算。配置 enabled=false 才停止计算。

`/api/room-memory` 和 JSON 导出增加 `semantic_map`：状态、源位姿/时间、view_mode、对应视图的完整类别分数、采纳/过滤原因和 filter_timestamp、固定调色板、PNG 图层及世界边界。PNG 翻转 Y 以与页面 +Y 向上对齐。完整格子证据保存于当前 session 的 `semantic_map.npz`，包含扁平网格 indices、probabilities、weights、labels、origin、cell_size、grid_size、timestamp、revision；`semantic_map.json` 保存最近图层与源帧信息。不长期保存 RGB-D 或点云。索引恢复：`iy, ix = divmod(index, grid_size)`，格心为 `origin + ([ix, iy] + .5) * cell_size`。

## 验证与边界

CPU 测试覆盖墙体遮挡、无效深度、视野后方、位姿移动/转向后迟到结果不漂移、重复时间戳、完整分数融合、不同方向标签、无 Portal 工作、session reset 丢弃在途结果。浏览器测试覆盖固定配色、世界尺度、显示开关和 reset。另用运行中的 Isaac 同步只读观测包做了真实 CLIP 单帧方向子图推理，没有推进小车。

这不是房间精确分割：整图或子图仍可能同时包含门两侧的内容，错误分类也会融合进去；深度射线保证几何可见性，不保证场景标签绝对正确。近处低矮家具不再单独截断投影；若门口同时能看到两个房间，最远观测不保证属于同一房间，因此仍可能混入跨房间语义。GT 位姿下验证坐标稳定，不代表实车定位漂移已经解决。尚未进行跨房间完整真实导航的分类质量评测。

兼容旧 semantic_projection 配置中的 max_range、depth_min/max、height_min/max、endpoint_margin、sample_columns/rows，但这些几何键不再控制投影；默认 JSON 已移除。几何域统一由 regions 配置管理，不依赖短期局部避障图。regions.depth_max 是有效测量量程，不是语义层额外的圆形窗口。
