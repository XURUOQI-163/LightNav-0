# Habitat 简洁交互页面：OmniVLA / LightNav

服务器在无显示器模式下运行 Habitat，浏览器实时显示 RGB；页面输入语言指令并选择模型。视觉风格复用 MuJoCo 深色页面，功能仅包括相机、语言输入、模型选择、开始、停止、重置和延迟/步数显示。

默认保持简洁页面；可用 `--room-memory` 额外启用基于前向 RGB-D 的房间记忆图，包含独立 YOLOE-26s 门口检测与 CLIP / Grounding DINO-T 可选房间语义。没有雷达、ROS、小车部署控制台或复杂暂停工作流。现有 MuJoCo、OmniVLA 和 LightNav 模型源码未改动。页面选择模型只是选择连接的服务，不会自动加载权重。

## 启动

### 1. Habitat 与网页（一个终端）

已有 `omnivla-habitat` 环境和本机场景可用。网页依赖已安装；换机器时先执行 `python -m pip install -e .`，勿将 Habitat 安装到模型环境。

```bash
conda activate omnivla-habitat
cd ~/projects/LightNav-0/habitat_demo
CUDA_VISIBLE_DEVICES=1 ./run.sh \
  --scene /home/xrq/datasets/habitat/scene_datasets/habitat-test-scenes/skokloster-castle.glb \
  --host 0.0.0.0 --port 8090
```

浏览器访问 `http://服务器可访问IP:8090/`。只有浏览器和服务在同一台机器，或已有端口转发时，才能使用 `http://127.0.0.1:8090/`。服务器必须允许浏览器访问 8090；不强制额外建立 SSH 隧道。此简易控制台面向可信局域网，没有用户认证。

场景需有对应 navmesh；也可通过 `--scene` 使用已有其他场景。不会下载新数据集。加载完成后 aiohttp 输出监听地址。

### 切换到已下载的 HM3D 示例场景

先在现有 Habitat 网页服务终端按 `Ctrl+C`，避免重复占用 8090。模型服务可以继续运行。以下命令从普通住宅的走廊开始（固定 `--seed 0`）：

```bash
conda activate omnivla-habitat
cd ~/projects/LightNav-0/habitat_demo
CUDA_VISIBLE_DEVICES=1 ./run.sh \
  --scene /home/xrq/datasets/habitat/scene_datasets/hm3d/example/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb \
  --scene-dataset-config /home/xrq/datasets/habitat/scene_datasets/hm3d/example/hm3d_example_basis.scene_dataset_config.json \
  --host 0.0.0.0 --port 8090
```

刷新原网页，选择已启动的模型并输入指令即可。两个模型和 APOS/OPOS 显示继续使用原有接口。场景配置会交给 Habitat 加载，运动仍使用 Habitat 的 Y-up 世界坐标，不对相机和导航网格额外旋转。

三个场景共用上面的配置文件；切换时只替换 `--scene` 路径中对应的目录和文件名，重启 Habitat 网页服务：

| 场景目录 | 文件名 | seed=0 初始画面 |
| --- | --- | --- |
| `00337-CFVBbU9Rsyb` | `CFVBbU9Rsyb.basis.glb` | 住宅楼梯旁的走廊 |
| `00770-NBg5UqG3di3` | `NBg5UqG3di3.basis.glb` | 带壁炉的装饰房间 |
| `00861-GLAQ4DNUx5U` | `GLAQ4DNUx5U.basis.glb` | 房门及室外方向 |

`--seed` 可改变起点采样；三个扫描场景仍可能有缺面，提升分辨率不能修复扫描缺失。当前下载不含语义标注，Habitat 0.3.3 可能输出 `SemanticScene ... SSD Load Failure`；已实测不阻止本页面的 RGB 渲染和 navmesh 导航，本页面不使用语义传感器。

### 2A. 运行完整 OmniVLA（另一个终端）

使用纯模型服务，不启动 `limo_adapter.server_omnivla_full` 或其复杂 dashboard：

```bash
conda activate omnivla
cd ~/projects/LightNav-0/habitat_demo
CUDA_VISIBLE_DEVICES=1 python -m habitat_web.omnivla_service \
  --repo /home/xrq/projects/OmniVLA \
  --checkpoint /home/xrq/projects/OmniVLA/omnivla-original \
  --host 127.0.0.1 --port 8891
```

服务直接复用 `OmniVLA/inference/run_omnivla.py` 的模型加载、语言模态预处理和 L1 动作头，未导入 `limo_adapter` 或 `semantic_topology`。默认 `--resume-step 120000`。需要本地完整 checkpoint；缺权重时报错，不自动下载。当前完整模型接口是单帧推理，不在服务里增加新的历史策略。

等待 `http://127.0.0.1:8891/health` 返回 `ok: true`，页面选择 **OmniVLA**，输入英文目标描述（如 `find a doorway`），点击开始。

### 2B. 运行 LightNav（选用已有服务或另开终端）

```bash
conda activate lightnav
cd ~/projects/LightNav-0
PORT=8050 CUDA_VISIBLE_DEVICES=1 lightnav-serve \
  --task vln --model_path checkpoints/LightNav-0 \
  --backend vllm_local --gpu_memory_utilization 0.75
```

等待模型加载和预热完成、8050 开始监听，页面选择 **LightNav**。已有 8050 服务时直接复用，不要重复启动。GPU 编号和显存比例应按空闲资源选择；两套模型不必同时启动。

## 页面操作和运行方式

- **开始**：从当前位姿开始新任务、创建新模型会话。
- **停止**：取消待返回的结果；若一个环境步进已经开始，等它结束并更新最终画面后才确认停止。
- **重置**：停止任务，回到此次启动时的相同起点和朝向，清空本轮步数。
- 切换模型前停止；做相同起点对比时先重置，再选择另一模型。
- 关闭网页不等同于停止任务；任务在服务端继续，默认不限步数。需要停止时点击停止按钮。
- 推理过程中环境保持静止，收到结果才推进一段仿真时间。因此画面更新频率受推理频率影响，仿真时间不是墙钟时间。页面刷新频率不会改变运动。
- **LightNav 预测坐标**：第一人称画面叠加橙色 APOS、绿色 OPOS 十字标记及 `(x, y)` 像素坐标，原点在图像左上角。仅显示模型实际返回的点；目标不可见、转向或停止等状态显示文字，不生成替代坐标。OmniVLA 不显示这两个点。
- 有预测坐标时，页面显示与坐标对应的**模型观测帧**，也就是该次动作执行前的 RGB，避免将旧坐标画在移动后的图像上。下一次预测返回时一起更新画面和坐标；手动停止、重置或开始新任务会清除旧坐标并显示当前画面。送给模型的始终是环境最新的原始 RGB，叠加标记不参与控制，不改变速度和运动逻辑。

两种模型使用同一个简易局部轨迹速度控制器，默认线速度上限 0.15 m/s、角速度上限 0.5 rad/s，每次预测执行 1/3 仿真秒，默认不限制动作次数（`--max-steps 0`）；仅在显式指定正整数时限制步数。这是 navmesh 上的 agent 导航，不是完整 Limo 动力学，也不是原 MuJoCo MPC或官方导航评测。不要仅凭显示延迟比较两种模型的导航能力。

OmniVLA 的 `[x,y,cosθ,sinθ]` 按现有部署约定将位置乘 0.1 转成米；LightNav 的 `[forward_m,left_m,yaw_rad]` 已是米和弧度。均取第 5 个点（不足则末点），限速保持前进弧线曲率。Habitat 朝向默认 -Z，左转为正，使用 +Y 旋转。OmniVLA 的这个动作头没有可靠的独立到达信号，本服务不伪造 `goal_reached`；可手动停止；显式设置正整数 `--max-steps` 时也可在上限停止。页面会显示停止原因，模型返回 STOP 也不代表系统已验证找到目标。

## 文件

| 文件 | 用途 |
| --- | --- |
| `habitat_web/simulator.py` | 唯一 Habitat 调用点，无窗口渲染、navmesh 运动和相机 |
| `runtime.py` | 单线程 EGL 所有权、异步推理闭环、停止与重置 |
| `clients.py` | LightNav WebSocket / OmniVLA HTTP 协议适配 |
| `control.py` | 单位和朝向转换、共同的简易速度控制 |
| `server.py` | 页面、最新图像、状态和控制 HTTP 接口 |
| `omnivla_service.py` | 独立纯 OmniVLA 模型服务 |
| `web/` | MuJoCo 风格的简洁页面 |
| `tests/` | 协议、单位、生命周期与 HTTP 测试 |

具体参数见 `CONFIGURATION.md`，验证证据见 `docs/VERIFICATION.md`。

## 可选：房间语义拓扑记忆

在上述 Habitat 启动命令后增加：

```bash
  --room-memory --room-memory-config room_memory/config/default.json
```

网页将增加 2D 轨迹、房间节点、已穿越的门户、当前箭头和代表图。默认使用本地 **CLIP ViT-B/32**（omnivla 环境，CPU）；`--room-semantic-backend groundingdino` 可切换至独立 groundingdino 环境的 GPU 区域检测。两者均支持语义投影，无需 API key，详见 [并列后端配置](room_memory/BACKENDS.md)。使用 `--room-semantic-backend places365` 可选择原 CPU 分类器。无需另外开终端启动识别服务。几何处理与语义识别在独立队列执行，不给导航下发动作。

选择 Places365 时读取 `/data/disk1/xrq/projects/places365` 中的权重和类别表。路径、Python 解释器、线程数及运行模式在 `room_memory/config/default.json` 的 `semantic` 中配置。365 类结果按映射表汇总到房间类别；未映射类别和低置信度结果保留 unknown。

开口只是几何候选，实际跨越后才连边；不保证得到完整建筑房间划分。设计取舍、开源参考、接口、存储方式和启动命令见 [room_memory/README.md](room_memory/README.md)，增量验证见 [docs/ROOM_MEMORY_VERIFICATION.md](docs/ROOM_MEMORY_VERIFICATION.md)。
