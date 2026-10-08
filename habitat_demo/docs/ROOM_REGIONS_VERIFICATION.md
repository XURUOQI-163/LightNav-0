# 房间范围验证（2026-10-07）

基于已有 room_memory 增量增加区域；输入实际前向 RGB-D 与 Habitat GT pose。不读取场景完整几何、物体坐标或真实房间标签进行建图。回放工具用保存轨迹/导航网格安排测试相机位姿，**不是 LightNav/OmniVLA 自动导航成功率测试**，也没有运行实体机器人。

## 自动测试与真实启动

Habitat demo **166 项测试通过**，包含 Chromium 测试；原有基线为 144 项。

- 新增 22 项覆盖：有效/无效深度、遮挡后方、前向扇形范围、旋转坐标、重复墙体证据、低矮家具、不完整轮廓、未知孔洞、Portal 分区、候选晋升、晚到门户重分区、跨门迟滞种子过滤。
- 配置越界、无限循环风险、单层高度保护、session 重置、JSON/NPZ 内容、错误在真正恢复前保持可见。
- 原被动性测试启用区域：开启/关闭整个记忆时，固定策略输出的导航命令序列仍完全一致。
- 网页测试实际执行区域绘制、透明孔洞、点击详情、隐藏图层、地图缩放、错误提示和 session 清空；原节点/门户/代表图/APOS/OPOS 测试保持通过。

使用真实 Habitat CLI 在临时端口 18090 启动 00770 和默认配置，接口返回 `structure.enabled=true/status=ready`，初始范围约 10.645 m²，`doorway_status=ready` 且两个错误字段为空。测试结束已关闭该服务并确认端口释放。

## 两个真实场景回放

| 数据 | 结果 |
| --- | --- |
| `00337` 已有穿门返回轨迹，171 帧 | 两个正式节点 R0/R1、一条 P0，P0 穿越两次，最后返回 R0；区域分别约 14.335 / 20.865 m²，均为 partial |
| `00770` 同层往返轨迹，184 帧 | 两个正式节点、两个候选节点、三个 Portal；累计水平观测覆盖约 97.388 m²；所有区域保持 partial |

候选房间可在未进入时拥有实际看到的区域；尚未看到的部分没有生成完整矩形。Places365 在这两条回放中未达到命名条件，保留 unknown；没有修改类别阈值制造房间名称。

截图：[00337 穿门返回](room-regions-return.png)、[00770 范围与候选空间](room-regions-00770.png)。截图使用实际回放 graph.json 驱动现有网页，浏览器未报告 JS 异常。代表图来自真实回放采集。

数据：[00337 快照](fixtures/room-regions-return-20261007/graph.json)、[00770 快照](fixtures/room-regions-00770-final-20261007/graph.json)。同目录 summary.json 记录帧数、检测统计、结构诊断和各节点面积。

## 已确认限制：00770 的返程漏切换

00770 这条测试路线最后回到初始 XY，但当前 Room ID 仍为 R1，未切回 R0。已用修改前源代码、相同相机轨迹及相同 YOLOE 配置完整重放对照，**Portal 列表、所有轨迹房间标签和最终 current_room_id 完全一致**。这是既有门口检测/穿越关联链路的限制，不是本次区域层改变了运动或节点逻辑。

因此该场景图能展示区域构建与候选空间，但不能当作正确房间分割的标注结果。后续仍需单独改善门检测召回、门平面定位及回访关联。此次按授权范围保留原门户和模型行为，没有通过强制改 Room ID 或语义标签掩盖问题。

高柜仍可能形成假墙，墙面缺失会造成片区泄漏或人工分界；虚线与 partial 必须保留。上楼超过高度阈值只暂停区域投影，不提供多楼层拓扑，也不禁止环境运动。

## 开销与优化

对 00770 相同轨迹优化了区域扩展循环，改用 Python 原生标量缓冲代替逐格 NumPy 索引；优化前后正式与候选节点的完整区域输出逐项一致。

最终回放共执行 84 次范围更新。工具按每帧记录最新一次更新耗时，去重后采得 76 条记录：中位约 **51 ms**，P95 约 **71 ms**，采样最大约 **99 ms**。这不包含同一帧中被后一条计时覆盖的更新，不是所有更新的严格最坏耗时，也不包含 VLN 推理、渲染、语义、落盘和网络开销。

默认每 0.75 仿真秒更新结构；门户变化会额外分割。异步与队列上限避免积压，但不保证无 CPU 争用。需要降低频率时可把 `regions.update_interval` 调到 1.5。

独立代码审查发现的两个问题已加入先失败后通过的回归测试：邻域参数可能造成超量计算、异步门户回调错误清除结构错误提示。均已修正。

## 复验命令

在仓库根目录运行完整 demo 测试：

```bash
PYTHONPATH=habitat_demo PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
TEST_CHROMIUM=/home/xrq/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome \
  mujoco_demo/.venv/bin/python -m pytest -p pytest_asyncio.plugin habitat_demo/tests -q
```

Habitat Python 使用独立环境，测试使用已有 `.venv` 的 pytest-asyncio/Playwright；没有升级 Habitat 运行依赖。Python 3.9 编译检查、Habitat demo Ruff 检查通过。未修改 src/，本次不声称重新验证 GPU 模型本身。

实际门口返回路线复验（输出必须为新目录）：

```bash
conda activate omnivla-habitat
cd ~/projects/LightNav-0/habitat_demo
HM3D_DIR=/home/xrq/datasets/habitat/scene_datasets/hm3d/example
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tools/replay_visible_portals.py \
  --trajectory docs/fixtures/visible-portal-return.jsonl \
  --scene "$HM3D_DIR/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb" \
  --scene-dataset-config "$HM3D_DIR/hm3d_example_basis.scene_dataset_config.json" \
  --output /tmp/room-regions-return-check --semantics
```

00770 的测试轨迹保存于 [room-regions-00770-route.jsonl](fixtures/room-regions-00770-route.jsonl)，替换上述 trajectory 和 scene 即可；该轨迹仅为离线验证，不是新增导航策略。
