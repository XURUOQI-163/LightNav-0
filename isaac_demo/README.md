# Isaac Sim / Carter 网页适配

复用 `habitat_demo/habitat_web` 的网页、模型客户端与导航循环，模拟器替换为独立 Isaac worker。
完整操作与原理记录见 [`安装手册2.md`](../../安装手册2.md)。

**验证状态（2026-10-11 更新）**：当前 `nvidia-smi` 已正常。新增实时第一人称、第三人称跟随和模型输入视图；自动测试及简单场景 GPU 测试通过；完整 GRScenes 首帧加载超过独立测试的 900 秒上限，未完成该场景验收。启动瓶颈已定位到大量 MDL 路径别名与材质处理，详见《安装手册2》的 2026-10-12 诊断。历史驱动问题保留在该手册中供排查。**2026-10-12 追加**：已实现运行时 MDL 路径归一化和网页提前显示加载阶段；按用户要求，本次未启动仿真、未运行新增回归测试，因此以上历史结果不代表本次优化已通过完整验收。

## 启动

机器人尺寸可用 `--robot-profile original`（默认原尺寸）或 `--robot-profile half_xy` 切换。
小尺寸档长宽约为原来的 0.5 倍，总高度不变；轮子保持圆形，半径 0.12 m、轮距 0.27 m。
前向相机偏移为 `[0.2,0,0.6]` m（原尺寸为 `[0.4,0,0.6]`），高度/FOV/分辨率保持。
两档配置位于 `config/robots/`，统一提供轮距、轮径、相机偏移及 footprint；优先于场景 JSON 的同名旧字段。
footprint 使用车体根坐标系，保留前后不对称：原尺寸 x=[-0.431,0.279]、y=[-0.315,0.315] m，小尺寸减半。
门检测宽度随档位更新，物理安全余量不缩放；启用 `--local-navigation` 时，同一 footprint 用于局部路径修正和执行前碰撞检查。
原 USD 不改写；小尺寸在匿名运行时图层中重塑车体、下移缩小后的圆轮、修正关节锚点和质量，并交由 PhysX 重算惯量。
动态刚体保留标准 translate/orient/scale，其初始位姿写入运行时 root layer，确保 PhysX 的运动更新能传递到可见模型；静态几何和关节使用 session layer。
2026-10-13：CPU/USD 检查覆盖几何比例、总高度、圆轮、关节锚点、资产不变、配置及动态位姿写入后可见模型跟随更新；尚未重新运行完整 GRScenes 的动态行驶验证。
改变档位需重启终端 B，模型终端无需重启；当前运行中的进程不会因修改文件而自动缩小。

```bash
cd /data/disk1/xrq/projects/LightNav-0/isaac_demo
# 终端一；首次运行等待 READY
MODEL_GPU=1 ./start_model.sh lightnav
# 或 MODEL_GPU=1 ./start_model.sh omnivla
```

```bash
# 终端二；省略 --room-memory 可单独检查基础导航
cd /data/disk1/xrq/projects/LightNav-0/isaac_demo
./run.sh --gpu 0 --host 0.0.0.0 --port 8091 \
  --room-memory --room-semantic-backend clip
```

浏览器访问服务器的 8091，或使用 VS Code 转发给出的地址。运行图形引擎不需要服务器桌面。
网页现在先启动；场景与材质在独立进程中加载。页面显示阶段和等待时间，直到 `scene_ready=true` 才开放控制。能打开网页不代表仿真已就绪；加载失败时页面保留错误，需要修复后重启网页服务。
原 Habitat/MuJoCo 入口仍保留。

## 运行契约

- GT 世界：X/Y 水平，Z 向上；车体 +X 前、+Y 左；yaw 逆时针，米/弧度/秒。
- `pose_frame=ground_projected_base`：pose.z 是 floor_z，相机外参以地面投影坐标系为原点；真实车体 XYZ/四元数另见 chassis_pose。
- RGB：HWC uint8，JPEG 传给现有模型；深度：float32 H×W，光轴 Z 深度（米），无效值 0。
- RGB/深度同一相机，`depth_camera` 提供实际 K 和 optical→robot 外参。
- 物理步 `1/60s`，每条模型路径执行默认 `1/3s`，等待推理/浏览器刷新期间不推进物理时间。
- 使用两轮速度目标驱动 PhysX，不把路径直接写成机器人瞬移；仅 reset 设置出生位姿。
- Room Memory 是被动观察者，不决定运动。定位用 IsaacGTPoseSource，保留 CLIP/Grounding DINO/Places365 选择，默认配置为 `config/room_memory.json`，详见 [后端配置](../habitat_demo/room_memory/BACKENDS.md)。
- 此版本使用配置中的单层 `floor_z`；跨楼层时需要另行处理楼层归属。不是 SLAM 或 D455 噪声仿真。

## HTTP

| 接口 | 内容 |
|---|---|
| `GET /api/state` | 运行状态、速度上限、当前 GT、步数、模型延迟、停止原因 |
| `GET /api/camera.jpg` | 模型输入视图；LightNav 指向点时保留预测对应帧 |
| `GET /api/display.jpg?view=first` | 最新运动显示帧；`view=third` 为第三人称；`after=编号` 未变化返回 204 |
| `POST /api/start` | `{"backend":"lightnav","instruction":"go forward"}`，也支持 omnivla |
| `POST /api/stop` | `{}`，等待已开始的有限物理步完成 |
| `POST /api/reset` | `{}`，回出生点，重置模型任务和 Room Memory session |
| `POST /api/manual` | `{"linear":0.1,"angular":0,"duration":0.33}`，仅空闲时允许，最多 1s |
| `GET /api/sensors` | 最新物理观测的 GT、内外参、时间、观测帧编号、深度定义 |
| `GET /api/rgb.jpg` | 最新物理观测 RGB，与显示帧可能不同 |
| `GET /api/depth.npy` | 最新 float32 深度；用 `np.load(...,allow_pickle=False)` |
| `GET /api/observation.npz` | 原子同步 RGB-D 包，推荐后续适配使用 |
| `GET /api/room-memory` | 现有 Room Graph、轨迹、区域、多视图与实时分类 |
| `GET /api/room-memory/export` | 下载房间记忆 JSON |
| `GET /api/local-navigation` | 局部地图快照、原始/修正路径、当前轨迹纠正模式；未启用时 `enabled=false` |
| `POST /api/local-navigation-mode` | 空闲时切换 `{"mode":"planning_only"}` / `{"mode":"observe_only"}`；后者仅允许手动采图，不拦截碰撞 |

NPZ 内字段：`rgb`、`depth`、`K`、`camera_to_robot`、`pose=[x,y,z,yaw]`、`sim_time`、`frame_id`、`depth_definition`。
Isaac 还提供 `pose_frame`、`chassis_position`、`chassis_orientation_wxyz`，避免混淆地图原点与车体根节点高度。
不含 Python object/pickle。分开下载 RGB、depth 时比较 `X-Frame-Id` 与 `X-Sim-Time`；运行中可能跨越两次观测，因此推荐 NPZ。
观测帧编号独立于网页指向点显示的 `state.frame_id`。

```bash
curl --noproxy '*' http://127.0.0.1:8091/api/observation.npz -o /tmp/carter-observation.npz
```

```python
import numpy as np
with np.load('/tmp/carter-observation.npz', allow_pickle=False) as obs:
    print(obs['rgb'].shape, obs['depth'].shape, obs['pose'], obs['K'])
```

## 参数

`./run.sh --help` 可查看完整列表。配置文件内出生角度单位是**弧度**，CLI `--spawn X Y Z YAW_DEG` 的角度单位是**度**。

| 参数 | 默认 / 作用 |
|---|---|
| `--config` | `config/grscenes_carter.json`，默认已下载的第一套家居场景 |
| `--scene` / `--robot` | USD 覆盖；换场景必须同时给 `--spawn` |
| `--spawn X Y Z YAW_DEG` | 出生点，米/度；先验证是无障碍、可落地位置 |
| `--gpu` | 0；只开一个渲染 GPU；不叠加 CUDA_VISIBLE_DEVICES |
| `--width` / `--height` / `--hfov` | 640 / 480 / 90° |
| `--host` / `--port` | 127.0.0.1 / 8091 |
| `--lightnav-url` | ws://127.0.0.1:8050，服务器端访问地址 |
| `--omnivla-url` | http://127.0.0.1:8891，服务器端访问地址 |
| `--no-normalize-mdl-paths` | 关闭默认的运行时 MDL 路径归一化，仅用于恢复旧加载方式做对照 |
| `--display-fps` | 配置默认 15：每仿真秒最多输出多少中间帧，按物理步取整；0 禁用中间帧；不是实际网页 FPS 保证 |
| `--dt` | 简单跟踪器默认 1/3 仿真秒，最多 1 秒；MPC 固定 0.1 秒 |
| `--mpc` / `--no-mpc` | 默认关闭；启用仓库 MuJoCo/实车同源的 CasADi MPC，仅作用于 LightNav |
| `--realtime-update` / `--no-realtime-update` | 默认开启，单个异步模型请求与当前路径跟踪并行，新预测在控制步之间切换；关闭后整段执行；启用局部地图时允许持续修正、失败后重预测，仅作用于 LightNav |
| `--path-position-tolerance` / `--path-yaw-tolerance-deg` | 整段终点到达容差：0.10 m / 10° |
| `--path-timeout` | 单段执行时限：60 仿真秒；局部地图整段模式超时触发重预测，其他模式保持停止 |
| `--max-linear` / `--max-angular` | 0.15m/s / 0.5rad/s |
| `--max-steps` | 0（无限）；不屏蔽模型 STOP |
| `--timeout` | 模型请求超时 60 秒 |
| `--startup-timeout` | 场景启动超时 1800 秒 |
| `--command-timeout` | 模拟器单条调用超时 120 秒；超时终止本任务 worker，需重启 |
| `--room-memory` | 启用被动房间记忆；默认不启用 |
| `--local-navigation` | 启用独立局部占据地图、路径修正、实际速度指令检查；默认关闭 |
| `--local-navigation-config` | `config/local_navigation.json` |
| `--local-map-radius` | 覆盖圆形窗口半径，默认 5 m；需同时启用 `--local-navigation` |
| `--room-memory-config` | `config/room_memory.json` |
| `--room-memory-output` | `runs/room-memory`，各 session 独立目录 |
| `--room-semantic-backend` | clip / groundingdino / places365；不传时使用配置（clip） |

JSON 还提供 `floor_z`、`camera_offset`、`wheel_radius`、`wheel_base`、`physics_dt`、`settle_steps`、`render_frames`、`ambient_light`、`materials`。

`render_frames` 仅用于启动时预热（默认 4 次）。正常 RGB-D 和中间预览各触发一次显式传感器采集，只有缓冲区未就绪才重试；采图不推进物理时间。
实时更新最多保留一个在途模型请求，推理期间继续跟踪当前可行路径，仍逐步更新局部地图并检查运动指令。
新路径使用模型输入图像采集时的位姿转换；旧路径走完会停车等结果，不外推运动。
停止、重置、模型错误均收尾在途请求。`--no-realtime-update` 保留原整段执行调度。

## 局部路径修正（2026-10-14）

```bash
./run.sh --gpu 0 --host 0.0.0.0 --port 8091 \
  --lightnav-url ws://127.0.0.1:8050 \
  --robot-profile half_xy --max-linear 0.30 --max-angular 0.5 --display-fps 5 \
  --local-navigation --local-map-radius 5
```

可继续追加 `--room-memory --room-semantic-backend clip`。局部地图与房间记忆独立；不需要启用房间检测 才能避碰。
模型输入、原始预测和现有跟踪器保留；可选 `--mpc` 切换官方 MPC，`--no-realtime-update` 等待整段完成再请求。页面蓝色编号是模型原始点，绿色线是送往跟踪器的修正路径，可能包含更多点。
默认使用 `planning_only`：地图仅用于提前纠正预测路径，不再拦截跟踪器速度或根据 collided 标志停止。
起点已在膨胀区时，保留原预测到第一个可行点的离开段，继续纠正后续路径；找不到修正路径时沿原预测继续，并显示 `fallback`。
未经验证的离开段/回退路径显示黄色，不宣称可通行。原始障碍证据、实体碰撞几何不删除。
自动旋转解困不启用；模型 STOP、手动停止、异常、步数限制以及原有超时/无进展处理仍有效。
新预测不会在后台排队执行旧路径。

**左右深度解困（仅显式旧 guarded 模式，以下说明保留实现）**：比较图像左右各一块中央高度 ROI 的有效深度第 30 百分位，朝分数较大的一侧原地转；每次转动最多一个控制周期，并重新获取 RGB-D/位姿、更新地图。一次解困固定方向，避免左右振荡。按用户的仿真假设，只有这条解困指令（线速度严格为 0）跳过车体旋转碰撞检查；没有修改物理碰撞几何，正常跟踪和手动控制仍走原有保护。
当前朝向前方 0.5 m 的完整车体扫掠在局部图上可通行后，结束解困。默认更新模式用新视角重新请求模型 waypoints；LightNav 的 `--no-realtime-update` 模式保留剩余世界坐标路径，继续本地修正和跟踪；反复修正/解困仍不可执行时用最新画面重新预测。模型 STOP 仍直接结束任务。
配置 `recovery_angular_speed=0.35` rad/s（另受 `--max-angular` 限制）、`recovery_max_turn_deg=0`（默认关闭，显式设为正值才启用）；没有至少 0.2 m 平移进展时，多次解困共用转角预算，避免反复触发而无限旋转。设 `recovery_max_turn_deg=0` 关闭。深度不足或用尽预算仍无法恢复时停止并显示原因。Stop/Reset 可中断，`--max-steps` 非零时旋转小步也计入步数。
页面状态栏显示选向深度、旋转方向和已用转角；`/api/state.recovery` 提供详情。此更新只需重启自己的终端 B 并 Ctrl+Shift+R；模型终端 A 无需重启。

模型连接在渲染和旋转期间由独立接收协程持续处理 WebSocket 心跳，预测回复进入有界队列，不发送额外推理帧、不重新登录。旧客户端仅在请求预测时读取消息，长时间解困后可能因心跳超时产生 `Cannot write to closing transport`。应用修复需重启终端 B；若模型服务确实退出，页面会明确提示模型连接中断，服务恢复后手动重新开始任务。

**前向相机盲区**：当前 JSON 配置仅在出生点建立一次半径 1 m 的空闲假设；范围固定在出生位置，不跟着小车清空，也不伪装成实测地面。相机高度约 0.84 m、水平视场角 90° 时，最近可见地面约在车前 1.3 m；1 m 先验可能留下未知间隙。只有确认初始附近空旷时才将此项设为 2 m。
首次或后续深度发现的障碍立即优先；观测到的地面替换假设。未观测部分作为出生点先验保留，移出 5 m 地图窗口后丢弃，返回不会重建。
配置 `initial_clear_radius_m` 可调整这份假设，设为 `0` 关闭该空闲先验。页面浅青色显示尚未被观测替换的假设区域。
若仍遇到未知区域，可停止任务后主动选择“手动采图（不拦截碰撞）”，采集后切回轨迹纠正。采图模式不允许自动 VLN。
显式 Reset 会清空地图、路径并重新建立出生点假设；位姿异常跳变仅清图，不重新建立假设。停止任务保留地图。

窗口仅按空间移出清理，不按时间清理。相机每次完整同步观测后更新地图；显示预览帧不用于更新。
CPU 工作在独立单线程中，每次导航等地图/修正完成后再下发动作，不修改物理步长。
配置与模块说明见 [`local_navigation/README.md`](local_navigation/README.md)。相机配置已用独立 RTX 平面场景验证，实际 GRScenes 连续行驶仍需验收。相机更新必须重启终端 B，重新建立地图；模型终端 A 无需重启。
第三人称 JSON 参数：`chase_distance=1.8`（车后距离）、`chase_height=1.3`（相对车体根的高度）、`chase_target_height=0.3`（相对车体根的注视点高度），单位米；跟随 yaw，忽略车体轻微侧倾。跟随相机没有墙体避让，在狭窄位置可能被墙遮挡，可看左侧第一人称或减小后退距离。

原资产只读；运行层应用单位转换并停用场景自带 physicsScene。
运行层同时把资产空分组节点上的碰撞声明补到实际静态网格，保留门和室内空间的三角面结构。

## 视角与性能

网页横向并列展示实时前向 RGB、第三人称跟随、模型输入与 APOS/OPOS，不需要切换；窄屏纵向排列。
后者保留预测对应图，避免把旧预测点画到新视角；另两个视图不叠加预测点。
中间帧经私有 Pipe 到达显示缓存，不修改模型输入缓存、完整 RGB-D 观测或 Room Memory 采样。
一条动作仍执行原来的物理步数。运动段按实际物理姿态出图并限制过快突发；等待模型时仍暂停推进。
新增渲染会占用 GPU、增加墙钟耗时；`--display-fps 0` 可关闭中间出图做性能对照。

页面显示“模型推理”（服务报告）、“推理往返”（客户端请求）、“运动段耗时”（物理+出图+等待）、“渲染与编码”（该段所有出图累计）和“显示新帧率”（最近两秒实际展示的新帧数/2）。
渲染耗时是运动段耗时的一部分，不能重复相加。帧率在等待推理或停止时下降到零是正常现象。
`/api/state` 还提供 `physics_ms`、`render_ms`、`step_wall_ms`、`motion_phase`；前三项是上一段运动值，非持续平均。

## 验证

真实测试需要 GPU，执行前确认无另一实例占用相同资源：

```bash
cd /data/disk1/xrq/projects/LightNav-0/isaac_demo
PYTHONPATH=.:../habitat_demo "$HOME/.conda/envs/omninavbench/bin/python" tests/smoke_sim.py
```

输出 `runs/smoke/start.jpg`、`forward.jpg`、`left.jpg`、`reset.jpg`，相应深度 `.npy`、第三人称 `*-third.jpg`、中间帧 `preview-*.jpg`、`preview-times.json` 和 `report.json`。
测试断言前进位移、左转正角度、深度有效比例和重置位置。

若网页已经启动，可复用正在运行的 Isaac，避免重复加载材质（测试会停止当前任务、移动车辆并重置）：

```bash
"$HOME/.conda/envs/omninavbench/bin/python" tests/smoke_http.py
```

输出位于 `runs/smoke/http/`，包括同步 NPZ、RGB 图片和运动报告。

CPU 检查（测试解释器需 pytest、pytest-asyncio、numpy、pillow、aiohttp）：

```bash
cd /data/disk1/xrq/projects/LightNav-0
PYTHONPATH=isaac_demo:habitat_demo python -m pytest habitat_demo/tests isaac_demo/tests
```

本服务器有系统 ROS pytest 插件，实施时使用了 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 和显式 `-p pytest_asyncio.plugin`。
浏览器测试需要 Playwright/Chromium，未安装时自动跳过。

## 启动阶段与材质路径（2026-10-12）

`normalize_mdl_paths` JSON 布尔参数默认为 `true`，CLI `--no-normalize-mdl-paths` 可覆盖为 `false`。
对环境中 `info:mdl:sourceAsset` 已解析的本地 `.mdl` 路径做 realpath 归一化，写入匿名 USD session layer。
只统一路径，不合并 shader、不修改参数、贴图属性和绑定、不保存源 USD；远程、未解析、包内、时变路径不处理。
归一化发生在环境引用建立后、物理初始化及相机首次渲染之前。同步资产加载仍开启。

`/api/state` 新增 `scene_ready` 和 `startup`：
- `phase` / `message`：worker 报告的阶段；包括 engine、scene、materials、collision、physics、camera、first_frame，以及父进程的 ready/error。
- `elapsed_s` / `phase_elapsed_s`：启动总耗时 / 当前阶段耗时，网页进程计算，加载阻塞时仍更新；完成或失败后停止计时。
- `mdl_paths`：`rewritten`（更新的 sourceAsset 数）、`aliases`（已处理的不同解析路径数）、`canonical_files`（真实文件数）、`skipped`（跳过项）。统计只覆盖当前环境遍历到的 sourceAsset，不是 GPU 编译进度。

页面不显示虚构的百分比或预计完成时间。未就绪时 start/stop/reset/manual 返回 400，并禁用对应按钮。
HTTP 先启动仅在 Isaac CLI 启用；普通 Habitat 初始化方式保持原样。
退出正在加载的网页服务会取消等待并清理该服务创建的 worker，不操作模型服务或其他用户进程。
本次未测 GRScenes 优化后的首帧时间及材质外观，不能承诺从十五分钟降到某个数值。


### 2026-10-12 用户复跑补充

用户运行日志表明 MDL 归一化生效，材质创建阶段约 210 秒（旧日志约 1099 秒），但该次 worker 在约 294 秒关闭，HTTP 为 `error/EOFError`；这不是成功出图验收，也不是受控性能对照。
构造异常现已在 Kit 关闭前打印并发送到父进程，避免 fast shutdown 掩盖 traceback。
首次预热逐次报告 render 返回，随后报告 `readback`（读取和检查 RGB-D/第三人称）；渲染次数保持不变。
`Timeout (0:03:00)!` 是 faulthandler 的诊断输出，真正超时仍由 startup-timeout 控制。
本次错误上报修正尚未经过实际重跑；完整步骤及记录见《安装手册2》和《命令行总结》。


### 首帧相机缓冲区就绪（2026-10-12 后续修正）

用户的下一次完整 traceback 已确认：四次预热 render 返回后，`Camera.get_rgb()` 返回 `None`，直接转为 uint8 数组导致 TypeError。
已将“固定四次渲染后直接读图”改成“预热后检查实际 RGB-D/第三视角缓冲区”。
缺失/空数组时显式触发 Replicator 同步采集，使用 `delta_time=0.0`、`pause_timeline=False`、`wait_for_render=True`；采集期间关闭 playSimulations，结束时恢复原值。正常物理步数、速度及模型接口不变。
每轮重新读取前向 RGB、第三视角 RGB、深度，全部就绪后才复制并发布；不沿用某次重试遗留的单个通道，不使用空图/零填充代替。
中间显示帧仍只读两个 RGB，不给 Room Memory 添加额外深度帧。

新增 JSON 参数（原启动命令无需增加选项）：
- `camera_ready_max_renders: 60`：一次读取最多触发的额外显式采集次数；整数 1–600。
- `camera_ready_timeout_s: 30.0`：预热完成后，额外等待的墙钟预算；大于 0、最多 300 秒。

上述时间预算在显式采集调用之间检查，不能强制打断 C++ 阻塞调用；完整 worker 仍由 startup-timeout / command-timeout 限制。
非空但分辨率不符时立即报错，不偷偷缩放或忽略问题。若持续为空，错误包含缺失通道、次数和耗时，保留进程退出前原始异常上报。
启动页面新增 `sensor_wait` 阶段显示正在等待的通道。本次仅完成静态检查与代码审查，未另开仿真；真实 GRScenes 出图仍需后续运行确认。

补充采集状态处理：本机 Replicator 显式 step 后会停在 STEPPED 状态。因此一旦首次进入显式采集模式，后续每个完整观测和中间预览都显式调度新采集，不能仅因旧缓冲区非空就把它当新帧。原路径从未遇到缺失数据时仍沿用自动采集。新增跨帧更新、软时间预算的假相机用例，未执行。


三路视图各自独立更新；预测图及 APOS/OPOS 保持对应，不叠加到实时相机图。显示帧率只统计第一人称实时新帧。
2026-10-12 三图并列改动仅涉及静态网页，强制刷新浏览器即可生效，不必重启 Isaac。历史“切换视角”的操作说明由本节替代。
