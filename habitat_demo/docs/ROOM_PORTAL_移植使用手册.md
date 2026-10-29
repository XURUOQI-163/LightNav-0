# Room–Portal 地图移植使用手册

更新日期：2026-10-29。面向接手另一个项目的 Codex。本文依据当前代码整理，不要求移植 LightNav、MPC、Isaac Sim 或高层 Agent。

## 1. 先复制哪些文件

源项目根目录：`/data/disk1/xrq/projects/LightNav-0`。下文路径均相对此目录。

**建议整体复制 `habitat_demo/room_memory/`，保留包名与目录结构，再复制当前 Isaac 配置 `isaac_demo/config/room_memory.json`。** 不复制 `__pycache__`、历史 runs、虚拟环境。模型权重与对应推理环境需要单独准备，配置里的本机绝对路径不能直接照搬。

核心算法实际在 `habitat_demo/room_memory`，不是 `isaac_demo/local_navigation`。后者是短期避障/路径修正地图，不是本手册要移植的长期房间图。

需要原网页效果时，还需参考或复制：

- `habitat_demo/habitat_web/web/room_map.js`：地图绘制及轮询。
- 同目录 `index.html` 中的房间记忆面板、`styles.css` 中相关样式。不能只复制 JS，因为它依赖已有 DOM ID。
- `habitat_demo/habitat_web/server.py`：房间记忆 HTTP 接口实现示例。
- `robot_navigation_agent/web/room_map.js` 位于另一个项目，是 8092 页面的展示副本；8092 代理 8091 的数据，不是第二个建图后端。

## 2. 当前实际上有两套 R/P，不要混用

| 数据 | 来源 | 用途 |
|---|---|---|
| `room_partition.rooms / portals` | `partition.py` | 长期几何范围按开口分块，分块后汇总语义，形成当前几何分区 Rn/Pn |
| 顶层 `rooms / portals` | `graph.py`、`frontiers.py` | 旧的观测/跨门事件图：候选房间、真实进入、代表图、重访等 |

两者并存，并非同一套 ID。**不要拿分区 R6 去索引旧 graph 的 R6。** 如果另一个项目主要复用当前“房间轮廓＋语义＋开口”效果，主要读取 `room_partition`、`semantic_map` 和 `trajectory`；如果还需要门检测、跨门状态和代表图，再保留并展示旧图对应数据。长期几何分区不依赖必须先检测出视觉门口。

## 3. 文件职责

### 核心与服务

| 文件（位于 `habitat_demo/room_memory/`） | 做什么 |
|---|---|
| `__init__.py` | Python 包入口，复制时保留。 |
| `models.py` | `Pose2D`、`PoseSource`、`RGBDFrame`、轨迹点、RoomNode、ProspectiveRoom、PortalEdge 等数据结构；包含 Habitat 位姿适配。 |
| `config.py` | 参数加载、后端选择与校验；分别加载 memory、geometry、regions、doorway、semantic_projection、partition。 |
| `service.py` | 推荐接入入口 `RoomMemoryService`。管理异步队列、建图线程、语义与门检测任务、缓存、session、reset/close；不发运动指令。 |
| `structure.py` | RGB-D 反投影、连续垂直高度支持过滤、长期覆盖范围和重复高表面墙体证据。 |
| `partition.py` | 范围清理、距离变换、几何分区与瓶颈开口、区域 ID 跟踪、几何中心、区域语义汇总。 |
| `regions.py` | 栅格轮廓/孔洞提取与旧图的 Portal 约束区域归属；分区代码也复用它，不可随意删除。 |
| `graph.py` | 旧房间图管理器；处理轨迹、房间切换、代表帧、语义请求和持久化，同时持有 `structure`。 |
| `frontiers.py` | 视觉 Portal 的世界坐标匹配去重、候选房间、跨门晋升和相关记忆更新。 |
| `geometry.py` | 相机模型与深度几何门/瓶颈检测；区别于 `structure.py` 的房间覆盖投影。 |

### 语义与门检测

| 文件 | 做什么 |
|---|---|
| `semantics.py` | 语义分类抽象、输入帧/输出结果类型、类别及旧 Qwen 实现。其他识别器使用其中的公共类型。 |
| `clip_recognizer.py` | CLIP 图文场景分类及常驻子进程协议；类别和文本模板来自配置。 |
| `groundingdino_recognizer.py` | Grounding DINO 房间提示词检测、原图坐标框及子进程封装。 |
| `places365_recognizer.py` | Places365 分类与类别映射、独立环境子进程封装。 |
| `projection_worker.py` | 按后端选择 CLIP 或 Grounding DINO 投影 worker。 |
| `clip_projection.py` | 采集完整视图或左右视图，冻结采集时几何，异步分类后融合；最新待处理帧优先。 |
| `groundingdino_projection.py` | 根据检测框及同帧深度产生语义投影；输出检测框图像。 |
| `semantic_projection.py` | 公用语义栅格、视角权重、多类别证据融合、低置信度过滤、颜色和输出。 |
| `live_semantics.py` | 当前画面分类状态展示；不能把它的最新分类直接用于别的时刻的几何投影。 |
| `doorway_detector.py` | YOLOE 开放门口 RGB 检测及子进程封装。 |
| `doorway_depth.py` | 对视觉门框用同帧深度验证开口，并估计位置、宽度和朝向。 |
| `compare_semantics.py` | 单图后端比较工具，不是在线建图必经路径。 |
| `config/` | 默认配置与类别等资源；整体保留。 |

`README.md`、`ROOM_REGIONS.md`、`ROOM_PARTITION.md`、`SEMANTIC_PROJECTION.md`、`VISUAL_PORTALS.md`、`BACKENDS.md` 是深入阅读材料。部分旧文档描述历史版本，遇到冲突以源码、当前选用的配置和本手册的两套图说明为准。

即使只使用 CLIP，也建议先完整复制包：`service.py` 等文件仍导入公共类型及旧后端适配器，删文件容易造成导入失败。后续再按依赖拆分。

## 4. 基本原理与运行顺序

```text
同步 RGB + 轴向深度 + 相机内外参 + 采集时位姿
                   ↓ service.offer
           RGBDFrame / RoomGraphManager
                   ↓
     structure：长期 coverage + 墙体证据
           ┌───────┴────────┐
           ↓                ↓
  几何清理、开口分区     冻结当前投影范围 → CLIP/DINO
           ↓                ↓
       区域 Rn/Pn       多类别语义栅格融合
           └───────┬────────┘
                   ↓
   按分区汇总语义、中心点、连通关系和访问状态
                   ↓
    service.cache → HTTP JSON → room_map.js

并行分支：RGB 门检测 + 深度辅助 → 候选门/候选房间 → 跨门更新旧图
```

1. **长期范围**：在有效高度带内，为每个采样图像列选择“具有连续垂直可见高度支持”的最远实测位置。相邻列按较短距离铺扇形，累计到固定世界坐标栅格。当前默认垂直支持为 0.60 m、至少 3 个采样点，抑制家具窄凹槽；不是无条件取单个最远像素。
2. **墙体证据**：高表面需要高度跨度、邻域支持和多帧重复；高柜也可能被误认为墙，不是建筑墙真值。
3. **几何分房间**：对覆盖做小洞填补、平滑及内部碎墙等处理；用距离变换找较宽内部区域，优先扩张，再依据窄连接判定/合并分区，抑制细长伪区域。语义不参与切割。区域变化时通过重叠跟踪 ID，不能保证所有合并/分裂下编号永久不变。
4. **语义投影**：分类绑定原 RGB-D、位姿和冻结的投影区域，异步结果不能套用机器人后来位置。视场中心权重高、边缘低（余弦平方），每格保留各类别分数并做有界加权融合。CLIP 当前默认整图分类；左右视图实现仍保留。前三名分数低或彼此接近时可跳过该视图，不擦除旧可信证据。
5. **区域命名**：对分区内有语义证据的格子按证据权重汇总各类别分数，取最高类；没有语义证据时不能编造标签。输出分区轮廓、中心点、类型和 Portal 连通关系。

这不是 SLAM，也不是完整探索；不会保存每个物体的位置或稠密点云。范围可能跨过家具上方，所以 **coverage 不代表地面可通行**，不能直接拿来替代避障地图。

## 5. 新项目必须提供的输入

推荐实现新 `PoseSource` 并调用 `RoomMemoryService.offer(observation)`。默认 PoseSource 是 Habitat，其他环境必须显式替换。

| observation 字段 | 约定 |
|---|---|
| `sim_time` | 同步帧时间，秒、单调递增；字段名沿用现有接口，实机可填统一传感器时间。 |
| `jpeg` | 原始 RGB 图编码后的 JPEG bytes，不要输入网页预测点/文字叠加图。 |
| `depth` | 与 RGB 对齐的 H×W NumPy 数组，单位米，光学 Z 轴深度；不是灰度图、毫米或射线欧氏距离。无效值可用 0/NaN。 |
| `pose` | 由自定义 PoseSource 读取，转换成同一地图坐标系的 x、y、yaw。 |
| `depth_camera` | `fx/fy/cx/cy`（像素）和 4×4 `camera_to_robot`。 |
| `floor_elevation` | 非 Habitat 适配器的可选楼层/基座高度参考，米；用于跨层隔离。 |

相机反投影使用光学坐标：右、下、前；`camera_to_robot` 必须把它转换到机器人 **前、左、上** 坐标系，包含真实相机安装位置。地图 XY 为水平面，yaw 为弧度、绕上轴逆时针为正。机器人位姿不要误填成相机位姿，否则会重复计算安装偏移。

参考适配：`isaac_demo/isaac_web/geometry.py::IsaacGTPoseSource`；参考输入构造：`isaac_demo/isaac_web/scene.py`；参考服务初始化：`isaac_demo/isaac_web/__main__.py`。只借鉴这几段接口，不需要复制整个仿真器。

## 6. 最小接入骨架

以下放进新项目的异步应用生命周期；项目中要能 `import room_memory`。`frames()` 代表新项目自己的同步传感器流，**不是本包提供的函数**。

```python
import math
from pathlib import Path
from room_memory.models import Pose2D
from room_memory.service import RoomMemoryService
from room_memory.config import (
    load_config, load_region_config, load_doorway_config,
    load_projection_config, load_partition_config,
)

class ProjectPoseSource:
    def pose_at(self, timestamp, observation):
        p = observation["pose"]  # 此例假设已转换为地图 XY / yaw
        result = Pose2D(float(p["x"]), float(p["y"]), float(p["yaw"]))
        if not all(math.isfinite(v) for v in (result.x, result.y, result.yaw)):
            raise ValueError("invalid synchronized pose")
        return result

async def run_memory(frames):
    cfg = Path("config/room_memory.json")
    backend = "clip"
    memory_cfg, geometry_cfg, semantic_cfg = load_config(cfg, semantic_backend=backend)
    service = RoomMemoryService(
        memory_cfg, geometry_cfg, semantic_cfg, Path("runs/room-memory"),
        pose_source=ProjectPoseSource(),
        region_config=load_region_config(cfg),
        doorway_config=load_doorway_config(cfg),
        projection_config=load_projection_config(cfg, semantic_backend=backend),
        partition_config=load_partition_config(cfg),
    )
    await service.start()  # 创建队列、会话与后台 worker
    try:
        async for observation in frames():
            service.offer(observation)  # 非 await；不要等待建图完成再控制机器人
            # HTTP 路由可返回 service.cache；它是最近已完成的快照。
    finally:
        await service.close()
```

新场景/位姿原点重置时调用 `await service.reset()`，不要每帧 reset。不要在 `offer()` 后立刻复用并改写同一深度数组；调用方应提供稳定的一帧数据。观察、语义与门检测有界排队，积压时会丢帧；新项目应在自己的事件循环中创建和调用该服务。

## 7. 配置与依赖

- 主建图进程：Python（当前 Habitat 使用 3.9，Isaac 使用 3.11）、NumPy、Pillow、SciPy；若沿用原 Web 后端还需 aiohttp。
- CLIP 环境：PyTorch、OpenAI CLIP 及模型权重；通过 `semantic_backends.clip.python_executable` 指定常驻子进程解释器。
- YOLOE 门检测环境及权重：由 `doorway.detector` 指定。只迁移几何分区时可设置 `doorway.enabled=false`，几何分区继续工作。
- DINO/Places365 是可选识别后端，但并非都提供相同的稠密语义投影功能：当前投影 worker 主要支持 CLIP 和 Grounding DINO。
- `semantic_backends` 中模型路径、解释器、设备，以及 `doorway.detector` 中模型/提示缓存路径必须改为新机器可用路径。不要整份复制旧 Conda 环境，也不要把所有模型依赖强行装进主应用环境。
- 当前 Isaac 配置优先用 `isaac_demo/config/room_memory.json`。`regions.cell_size` 默认 0.05 m，语义与分区共享该格网；`grid_size=1024` 是固定容量，不是无限地图。
- `regions.depth_max` 当前默认 6 m 是传感器接纳量程；“不额外限制语义投影半径”不表示取消深度有效量程。
- 调整新机器人尺寸：检查 `geometry` 和 `doorway.depth` 中宽度、margin、开口宽度。原 Isaac 使用 `robot_profiles.py::room_geometry_for_robot` 注入车体尺寸，新项目需要自己做对应适配。
- 重点调参块：`regions`（几何及垂直窄缝过滤）、`partition`（清理与分割）、`semantic_projection`（视图方式、过滤与融合）、`memory`（轨迹/节点）、`doorway`（视觉门）。

## 8. 输出、网页与接手顺序

每个会话目录通常包含 `graph.json`、`trajectory.jsonl`、代表图、`structure.npz`；启用相关功能后还有 `semantic_map.json/.npz`、`room_partition.json/.npz`。NPZ 保存格网或证据，JSON 供查询/展示。不要把这些文件误当作可直接恢复整个服务的 checkpoint；目前没有完整的跨 session 恢复接口。

网页最小接口：`GET /api/room-memory` 返回 `service.cache`；导出接口可返回相同 JSON；代表图接口参考 `await service.image(frame_id)`。DINO 检测图另有 `/api/room-memory/detection.jpg`。前端需要配套 `room-memory-panel`、`room-map`、`room-map-note`、详情和代表图等 DOM，直接参考原 HTML 整块迁移。不要为第二个页面再启动一套建图 worker。

建议另一个 Codex 按以下顺序操作：

1. 完整复制包和配置，修正本机路径，先接入同步 RGB-D/位姿和 `service`。
2. 先确认尺度、坐标、相机外参正确，再看 coverage 是否随运动稳定累计。
3. 接 CLIP 多类别语义，再接几何分区显示；视觉门检测可以独立启用。
4. 只在新项目适配层改传感器和 API，保持地图模块不依赖导航控制器。
5. 回归用例参考 `habitat_demo/tests/test_structural_vertical_support.py`、`test_room_regions.py`、`test_semantic_projection.py`、`test_room_partition.py`、`test_partition_cleanup.py`、`test_visible_portals.py`。

当前边界：不支持完整多楼层/SLAM 回环纠正；单帧可见范围及家具遮挡会影响分区；高柜与墙、真实门与家具开口仍有歧义；旧错误覆盖不会因调新阈值自动清除，比较参数应使用新会话。保留原始数据和坐标约定，比直接调分类阈值更重要。
