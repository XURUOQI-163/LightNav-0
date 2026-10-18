# 局部占据地图与路径修正

核心只接收同步 RGB-D 观测字典、GT 位姿、相机标定和机器人 footprint，不依赖 Isaac API、模型权重或 Room Graph。
当前接入 Isaac 网页入口；Habitat/MuJoCo 未自动启用此功能。依赖 numpy、Pillow、scipy，现有 omninavbench 环境已具备。

## 数据流与文件

| 文件 | 职责 |
|---|---|
| `models.py` | 参数验证、米/弧度坐标转换；局部 +X 前、+Y 左 |
| `rolling_grid.py` | 世界对齐栅格，机器人中心 5 m 圆窗；移出即丢弃，不做时间衰减 |
| `depth_mapping.py` | 光轴 Z 深度投影、地面支持、分高度障碍证据；临时点集不持久保存 |
| `blindspot.py` | 相机内外参判断近处下方盲区，只对其中未知格子初始化独立空闲假设 |
| `collision.py` | 朝向相关矩形 footprint 膨胀、未知检查、平移/转向扫掠检查 |
| `path_corrector.py` | 原始路径检查、带航向的有界局部搜索、默认仅建议修正；起点重叠允许沿原预测离开，修正失败保留原预测 |
| `command_guard.py` | 保留的旧 guarded 模式使用；默认不运行。对跟踪器实际 v/w 圆弧、反应距离和制动延伸再检查；必要时降速或原地转向，均重新检查 |
| `recovery.py` | 比较左右 RGB 对应区域的有效深度，提供原地旋转解困方向；IsaacRuntime 执行恢复循环 |
| `service.py` | 串行状态、仅轨迹纠正/手动采图模式、地图/两条路径快照 |

`IsaacRuntime` 在每次完整 RGB-D 返回后更新地图；模型输出先按输入帧位姿转入世界坐标，修正结果再转回当前车体坐标交给原跟踪器。
模型原始 waypoint 和输入图片上的编号不被替换。已有 STOP、任务取消、Reset 仍生效；取消中的规划结果不能下发动作。
网页轮询 `/api/local-navigation` 只读快照，既不推进物理，也不重复计入深度证据。

## 默认参数（`../config/local_navigation.json`）

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `window_radius_m` / `resolution_m` | 5 / 0.05 m | 圆窗半径/栅格边长；窗口直径 10 m |
| `initial_clear_radius_m` | 当前 JSON 为 1.5 m | 一次性出生点空闲假设半径，允许 0–3 m，截在地图窗口内；0 关闭。省略此项时类默认值为 2 m |
| `near_blind_clear_distance_m` | 1.5 m | 近处下方盲区初始化的距离上限，允许 0–3 m，0 关闭；仍必须满足下方盲区且未知，并非填满半径内区域 |
| `safety_margin_m` | 0.05 m | 车体每侧安全余量，另加栅格和航向离散误差的保守补偿 |
| `soft_clearance_m` | 0.25 m | 路径搜索的近障碍代价范围 |
| `max_goal_adjustment_m` | 0.75 m | 在模型目标附近尝试可行终点 |
| `max_detour_extra_m` | 2 m | 相对模型路径的额外路程上限 |
| `max_search_expansions` / `search_timeout_s` | 20000 / 1 s | 搜索预算，超出后保留前缀；没有修正结果时沿原预测继续 |
| `heuristic_weight` | 2 | 加权启发式，优先缩短搜索耗时，不保证最短路径 |
| `heading_bins` / `primitive_length_m` | 32 / 0.15 m | 航向离散数/前进运动原语长度，包含左右弧线及原地转向 |
| `depth_min_m` / `depth_max_m` | 0.15 / 8 m | 有效深度范围；地图仍只保存圆窗内证据 |
| `sample_columns` / `sample_rows` | 160 / 96 | 深度抽样上限 |
| `ground_tolerance_m` | 0.04 m | 地面高度容差；地面支持只连接相邻且间距不超过 0.25 m 的测量 |
| `height_bin_m` / `height_margin_m` | 0.1 / 0.05 m | 障碍高度层/高于车体顶部的余量 |
| `clear_confirm_frames` | 2 | 新的有效射线连续穿过同一高度层才清掉旧障碍；无效深度不会清障 |
| `max_pose_jump_m` / `floor_tolerance_m` | 1 / 0.15 m | 不连续位姿/地面高度变化时重置，不跨跳变拼图 |
| `braking_deceleration` | 0.5 m/s² | 制动延伸估计的线减速度 |
| `braking_angular_deceleration` | 1 rad/s² | 制动延伸估计的角减速度 |
| `reaction_time_s` | 0.1 s | 执行扫掠的额外反应裕量 |
| `recovery_max_turn_deg` | 0°（默认关闭） | 未取得平移进展时的累计旋转指令预算；0 关闭自动解困 |
| `recovery_angular_speed` | 0.35 rad/s | 解困角速度，另受页面启动参数 `--max-angular` 限制 |
| `recovery_forward_distance_m` | 0.5 m | 退出解困前必须通过的前方完整 footprint 扫掠长度 |
| `recovery_reset_translation_m` | 0.2 m | 相对本轮解困起点的平移达到此值，下次解困重置预算 |
| `recovery_min_valid_fraction` | 0.2 | 左/右 ROI 至少 20% 有效深度才参与选向 |
| `recovery_depth_percentile` | 30 | 有效深度百分位，较大的一侧更空旷；不取最远单点 |
| `recovery_roi_top` / `recovery_roi_bottom` | 0.25 / 0.65 | 图像高度归一化范围；横向固定为左 5–45%、右 55–95% |
| `path_replan_interval_s` | 0.5 仿真秒 | 整段模式随地图更新周期性重算原预测剩余路径；前进段结束或受阻立即触发 |
| `path_failure_confirmations` | 3 | 新观测下连续局部失败次数，达到后决定重预测；仅显式启用解困时先尝试一次解困 |
| `path_no_progress_s` / `path_progress_translation_m` | 8 仿真秒 / 0.05 m | 无进展判据；纯转向预测可用朝向误差减小作为进展 |
| `model_replan_limit` | 3 | 无有效位移时最多额外重预测次数；达到上限明确停止 |

`--local-navigation --no-realtime-update` 使用 `isaac_web/incremental_execution.py`：
保存拍摄时刻的完整世界坐标预测，从原预测的剩余部分持续修正，不用截短后的绿色路径替代原始目标。
`plan_remaining` 保留全部蓝色点和原始拍摄位姿，`remaining_start_index` 标记已通过部分；
`goal_index` 使用原预测编号，使安全前进段不会被误判为完整完成。
走完修正前进段后继续用更新后的地图延伸。默认不做地图执行拦截；仍保留无进展/超时触发模型重预测的逻辑。旧 guarded 模式才使用碰撞拦截与可选旋转解困。
一次搜索超时不代表全局不可达。`--path-timeout` 在此模式中是旧预测总执行预算，超时触发重预测。
额外请求次数在实际位移达到 `recovery_reset_translation_m` 或整段完成后重置；重复失败到上限则停止。
默认 `planning_only` 仅在规划阶段检查 footprint；实际 v/w 不做占据扫掠拦截，观测中的 collided 标志也不触发停止或解困。

footprint、车高和相机参数来自选中的 `original` / `half_xy` 配置，不复制一套车体尺寸。
障碍优先于地面空闲及出生点假设；未知与窗口外不触发轨迹纠正，也不拦截实际运动。旧障碍只有新的有效深度反证或离开圆窗才消失。
地图 PNG 中橙色是当前航向的硬膨胀区；路径和指令检查会使用各自航向，不把车体近似成固定圆。
原始蓝点最多按实际模型输出显示，绿色修正路径点数不固定。路径的 `map_revision` 落后时页面变淡，避免把历史结果当成当前可执行结果。

## 未知区域中的预测路径（2026-10-17）

轨迹纠正使用 `CollisionChecker.for_planning()` 的独立查询视图：**未知允许作为预测路径，已知膨胀障碍才触发修正**。
未碰到障碍的原始点完整保留，包括超出局部窗口的预测后缀；`status=unchanged`，涉及未知/窗口外时
`reason=no_known_obstacle`。这表示没有检测到已知障碍，不表示整段都已实测可行。
不仅检查 waypoint 本身，也检查相邻点之间的扫掠，防止未知端点掩盖中间的墙。
远端点的检查只采样与有限地图相交的部分，不因距离很大而分配大量样本。

有已知障碍时，原有有界局部搜索继续使用朝向相关 footprint 膨胀，允许未知格子作为候选，
软障碍距离代价也只以已知障碍计算；修正搜索仍限制在局部窗口内。
真实栅格标签不改变，`point_status` / `segment_status` 保留 unknown / outside_window 信息，
`planning_policy=known_obstacles_only` 标记本策略。新深度揭示障碍后，下次规划按新证据修正。

默认模式为 `planning_only`：取消执行前速度扫掠拦截。`guarded` 旧实现保留为显式 API 比较选项，页面默认按钮只启用轨迹纠正。
起点落入膨胀区时，不抹除障碍，保留原始路径中到第一个可行点的离开段，再对其后的路径进行原有搜索。
这个离开段未经验证，以黄色显示；`start_overlap_ignored=true`。找不到修正路径时，`status=fallback`、
`correction_failure` 记录原因，沿剩余原预测继续，下次更新再尝试修正，不宣称已找到安全路径。
原始路径/地图证据不变；模型 STOP、手动停止、异常、步数及超时/无进展处理仍保留。
API `execution_collision_check=false` 表示当前不做地图碰撞拦截；Reset 回到此默认模式。

## 启动盲区与使用范围

按用户指定的出生环境假设，启动/显式 Reset 后，在首份标定深度观测的位置建立一次配置半径的空闲先验。
`grid.initial_free` 单独记录假设，既不写入实测 `ground/free`，也不删除障碍；首次扫描即有障碍的格子不接受假设。
后续地面/障碍证据替换对应假设格；未知深度不恢复假设。剩余先验固定于出生世界坐标，随滚动窗口移出而清除，返回不补种。
位姿跳变或地面高度跳变造成的内部重置不允许重新播种；新建服务或用户显式 Reset 才允许再次初始化。
页面用浅青色区分假设和实测空闲，API 的 `initial_clearance` 提供出生中心、配置半径和剩余假设格数。
这并非首帧后立刻丢掉整片假设（那会重新卡住盲区），而是只初始化一次，后续依靠真实证据逐格替换，不给新位置添加假设。
设置 `initial_clear_radius_m=0` 可关闭。保留显式手动采图模式作为可选操作，期间不拦截手动碰撞且不允许自动导航。

## 行驶中的近处下方盲区

`near_blind_clear_distance_m` 独立于出生点假设。每份新的同步观测先更新真实深度证据，
然后把世界栅格地面中心转换到当前机器人坐标、再按 `camera_to_robot` 的逆变换投影到相机。
只有**当前未知 AND 地面投影位于图像下沿以外**的格子才初始化；还必须位于机器人前方、相机水平视场内，
并处于配置的近处距离上限内。画面内因深度无效而未知的区域、车后方、水平视场外区域不会补空闲。
相机位置和俯仰变化会改变盲区，因此不是固定前方圆盘，也不是把灰色孔洞一律填平。

`grid.blind_free` 与实测 `ground/free` 分开保存；`blind_initialized` 防止同一个保留中的格子被反复初始化。
新位置遇到符合条件的未知格子也可以初始化，但已初始化区域固定在世界坐标，不随车体旋转漂移。
后续实测地面/障碍覆盖假设；已有障碍、膨胀区和执行前检查不会因这项假设被清除。
随滚动窗口移出的格子及初始化记录一起丢弃；Reset 同样清空。
网页用浅青色显示假设，`near_blind_clearance` 提供上限和剩余假设格数。设为 0 关闭这项功能，重启仿真生效。

本模块是基于采样深度和运动学的局部避碰层，无法保证未观测物体、动态物体或轮胎打滑下绝对无碰撞。
默认地面高度沿用现有单层配置，不支持自动识别楼层/台阶，也不改成 SLAM、全局地图或 MPC。
制动参数目前是保守运动学估计，没有标定真实 Carter 制动动力学。实时更新模式下推理与已有路径跟踪并行，逐步更新地图并检查速度。
局部搜索优先到达模型晚期目标附近；如果不可达、没有已知目标或预算耗尽，会保留搜索中已通过完整 footprint 扫掠检查且沿原路径取得进展的前进段。返回 `safe_prefix / reachable_progress`，`search_reason` 保留远端搜索失败原因。跟踪器生成的速度还要通过独立执行检查。

Isaac 中没有可行修正路径、速度被拦截为零、或步进的前进受阻标记触发时，会自动选择深度较大一侧原地旋转，每小步更新完整观测。选向忽略无效深度和底部近地面，深度相同优先左转；单次旋转过程锁定方向，重新开始一轮才重新选向。按用户明确的仿真假设，`scene.step(0, signed_speed, duration)` 不经过旋转碰撞检查；该豁免不传播到带线速度的指令或手动控制，也不删改物理碰撞体。
恢复的判据是前方配置距离的车体扫掠可通行，而非左右深度分数大就立即前进。恢复后重新推理，不执行旋转前缓存的 waypoints；如果新路径仍受阻，会在剩余旋转预算内重试。预算按下发角速度×时长累计，原地旋转不重置预算；Stop 可取消，任务步数上限也生效。双侧均无可靠深度或旋转预算耗尽时停止。前进仍可能被地面盲区或真实障碍限制，恢复循环不会伪造空闲区域。Habitat 默认恢复钩子为空，行为保持原样。

相机标定回归：旧适配器的焦距 `.02` 与对应光圈在本机 Isaac 5.0 RTX 下产生实际渲染/K 不一致；平面地板反投影误差约 1.49 m。现在 `configure_pinhole_camera` 使用实测一致的焦距尺度 `2.0`，同时按原 HFOV 设置光圈。独立 GPU 平面测试的误差约 5.5 mm，移动/转向后实测地面格增长，不依赖出生先验。复查命令（约半分钟，需空闲 GPU）：

```bash
cd /data/disk1/xrq/projects/LightNav-0/isaac_demo
PYTHONPATH=.:../habitat_demo /data/disk1/xrq/.conda/envs/omninavbench/bin/python scripts/check_camera_calibration.py --gpu 0
```

CPU 合成障碍、HTTP、取消流程和上述实际渲染检查不替代 GRScenes 连续行驶验收；当前用户仿真未被自动重启或控制。
