# YOLOE 可见门户验证（2026-10-06）

本次基于已有 room_memory 增量修改。导航/控制器与模型代码未改；测试没有运行实体车，也没有连接真实 LightNav/OmniVLA 推理服务。

## 实际模型与场景

使用本地官方 `yoloe-26s-seg.pt`、Ultralytics 8.4.78，CPU 两线程；真实 Places365 ResNet18 子进程。Habitat-Sim 0.3.3、HM3D example `00337-CFVBbU9Rsyb`、640×480 同步 RGB-D、GT pose、相机距 agent 原点 0.45 m。不读取语义传感器、真实房间 ID 或物体坐标。

实测发现旧估计将框底边直接投到 robot z=0 平面，扫描地面与 navmesh 高度偏差导致门距离和宽度低估。已改为优先使用框底部实际深度与局部墙条；新增高度偏差回归测试。Habitat 中门后的有效深度可达约 13 m，因此默认视觉验证量程上限设为 15 m；这不代表 D455 量程，也不把无效深度当自由空间。

## 两条真实 RGB-D 回放

| 回放 | 观测数 | 结果 |
| --- | --- | --- |
| 原 `Find a toilet.` 保存轨迹 | 134 | 29 次有效 RGB+Depth 观测合并为同一 P0；第 30 个零起始样本提前创建蓝色 P0 和候选 R1；路线未穿过这个门口，因此保持未进入 |
| 穿过已看到门口并原路返回 | 171 | 第 30 个样本建立 P0，第 99 个样本正式进入 R1；最终只有 P0 一条边、R0/R1 两个正式节点，P0 红色、穿越计数 2，返回 R0，visit_count=2 |

第二条是离线验证路线：保留原轨迹前 51 点，再沿 Habitat navmesh 中的一条可通行折线到门后并返回。不是 VLN 自动导航结果，不是强制探索功能，也没有把预设门框/房间标签输入算法。回放核心收到的仍是实际渲染 RGB-D+pose，门框来自真实 YOLOE 推理。

暖机后 YOLOE 单帧中位数分别约 90 ms 和 97 ms，首次加载另计；这是本机短回放观测，不能当作模型端到端导航延迟基准。最终 Places365 分数未满足分类条件，节点保留 unknown；没有为了展示名称而强改阈值。

**限制：原路线真正穿过的侧门仍被 RGB 模型漏检，因此那一段没有新增正式房间。** 这次验证证明提前建点、世界去重和穿越生命周期已接通，不证明全部侧门召回，也不是人工标注的房间分割准确率评测。不同门通向同一房间的合并仍需更多证据。

结果与权重 SHA256：[visible-portals-replay.json](visible-portals-replay.json)。
原始图快照：[蓝色未进入](fixtures/visible-portals-blue.json)、[穿越返回](fixtures/visible-portals-return.json)。

## 网页和自动验证

Habitat demo 全套 **144 项测试通过**，覆盖原页面、模型客户端、控制、Places365 与新增视觉功能：

- 可见门口即时创建候选，重复观测合并；附近两扇门保持独立。
- 开放/关闭/无效深度/过窄开口、斜视及单侧墙条、navmesh 高度偏差。
- 真实穿越后同编号晋升、同一门户返回；旋转、位姿跳跃、旧会话结果不制造穿越。
- 异步检测使用源 pose/timestamp，队列有界；慢检测不阻塞观察，关闭取消并回收子进程。
- 候选语义不影响拓扑，迟到 ROI 不覆盖正式语义；导航开启/关闭记忆时命令序列相同。
- Chromium 蓝/红状态、候选问号、点击详情、晋升后选择保持、代表图、错误显示、重置，以及原 APOS/OPOS 页面。

真实回放 graph.json 接入临时 HTTP 页面，验证蓝/红门户可点击、实际代表图加载、无 JS 异常。截图：[未进入](visible-portals-blue.png)、[穿越后返回](visible-portals-red.png)。另修正候选文字覆盖门户点击区域的 SVG 层级问题。临时服务测试结束即关闭。另用真实 Habitat CLI 在 18090 启动默认配置，初始观测正常导出 R0，YOLOE 子进程自动启动并返回 `doorway_status=ready`、`doorway_error=null`；随后 Ctrl+C 关闭并确认端口释放。

Ruff、Python 3.9 编译与 `git diff --check` 通过。本次未修改 `src/`，没有用不兼容的 Habitat/demo Python 去声称验证整个 GPU 模型仓库。

## 复现

仓库根目录运行测试：

```bash
PYTHONPATH=habitat_demo PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
TEST_CHROMIUM=/home/xrq/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome \
  mujoco_demo/.venv/bin/python -m pytest -p pytest_asyncio.plugin habitat_demo/tests -q
```

真实返回路线回放，输出必须使用新目录，避免覆盖已有 session：

```bash
conda activate omnivla-habitat
cd ~/projects/LightNav-0/habitat_demo
HM3D_DIR=/home/xrq/datasets/habitat/scene_datasets/hm3d/example
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/replay_visible_portals.py \
  --trajectory docs/fixtures/visible-portal-return.jsonl \
  --scene "$HM3D_DIR/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb" \
  --scene-dataset-config "$HM3D_DIR/hm3d_example_basis.scene_dataset_config.json" \
  --output /tmp/visible-portal-return-check --semantics
```

脚本默认仅保存标量检测诊断、图、轨迹和每房代表图；只有显式 `--save-rgb` 才另存有检测的诊断 JPEG，不存原始深度或点云。原用户路线可将 `--trajectory` 换为 `runs/room-memory/bfe951ecd52c499dbf94b5bd5ab0f020/trajectory.jsonl`。
