# Room Memory 验证记录（2026-10-02）

## 当前视觉门户版本

2026-10-06 已增加 YOLOE-26s 提前发现门户与候选房间。当前回放、限制与网页截图见 [VISUAL_PORTALS_VERIFICATION.md](VISUAL_PORTALS_VERIFICATION.md)。下文保留此前纯深度/语义接入的历史验证，不代表新版对所有侧门的召回保证。

## 200 步自动停止与侧门漏检排查

- 真实服务状态确认 `Find a toilet.` 回合在 `step=200/max_steps=200` 停止，无异常，最后返回非零路径；属于旧默认步数限制，不是已确认找到目标。
- 默认上限改为 0（不限），保留正整数作为可选限制；新增停止原因供页面展示。回归测试实际执行 205 步后响应模型停止，并检查手动停止、空路径、零速度、显式限制、重置清空原因。
- 右侧详情明确标注空间节点编号；unknown 且语义推理完成时显示“分类已完成，暂未定类”，暂定拓扑节点标注未确认门户穿越。
- 本次 **105 项测试通过（20.79 秒）**，含 Chromium 停止原因/节点显示验证；Ruff、Python 3.9 compileall、git diff --check 通过。
- 门户漏检仍未修复：在同场景按保存的 134 个轨迹位姿重渲染，130 帧无满足条件的双侧门框，4 帧未通过开口自由空间验证；没有门户候选。相机沿走廊右转进侧门，超出当前近似正面双侧门框算法的适用范围。没有改变阈值、没有按语义标签变化新增房间，没有修改用户已保存图。此重渲染不是原始全帧深度录像。诊断见 [room-memory-side-door-diagnosis.json](room-memory-side-door-diagnosis.json)。
- 用户运行中的 Habitat 与模型服务未重启；需重新启动 Habitat 才会采用新步数默认值。

## 未映射概率与识别置信度修正

- 排查用户的 `cam_example.jpg` 输出：图片本身带彩色 CAM 热力图；真实 ResNet18 Top-3 为 candy_store 0.162055、food_court 0.087395、ice_cream_parlor 0.064184，均未映射到当前房间类型。unknown 0.965029 是未映射概率质量，不是识别置信度。
- unknown 获胜时 confidence 改为 0，保留原始概率；命令行新增 `places365_top5` 和 `unmapped_probability`，不把原始场景预测隐藏在房间映射之后。增加真实 CLI 回归测试。
- **101 项测试通过（19.85 秒）**；Ruff、Python 3.9 compileall 与 git diff --check 通过。
- 再次完成真实 Habitat RGB-D + 固定假策略 + 真实 Places365 CPU 推理网页联调：30 步、20 个轨迹点、1 个节点、0 丢帧、无脚本异常，类别输出 corridor，分数约 0.653。此为该路径多视图分类输出，没有人工标签，不能视为准确率结论。重置 session 成功，测试服务及子进程已回收；用户自己的 8090 服务未停止。
- 最新记录：[places365-web-smoke.json](places365-web-smoke.json)，截图：[places365-web.png](places365-web.png)。下文保留早先不同测试节奏的记录。

## Places365 替换与收尾验证

当前默认语义后端已由千问改为本地 Places365。推理封装在 `room_memory/places365_recognizer.py`，读取用户已下载的 ResNet18 权重和类别表；默认 CPU、2 线程、常驻子进程，无需 API key 或额外服务端口。网页显示 Places365，启动说明与配置文档已同步。

- 环境：Habitat 为 Python 3.9.23 / NumPy 1.20.3，未安装 torch/torchvision；Places365 为 Python 3.10.22 / torch 2.2.0+cu121 / torchvision 0.17.0+cu121 / NumPy 1.26.4，未安装 Habitat。当前环境不能直接共用一个 Python 进程；实际采用 Habitat 自动管理另一个环境的推理进程，未合并或升级依赖。`in_process` 是已有兼容依赖时的可选模式，不代表本机 Habitat 已具备这些依赖。
- **100 项测试通过**，包含 Chromium 页面测试、真实 Places365 权重推理、进程复用与取消清理、缺失权重时几何继续工作、多视图概率聚合和未知类别保留。`ruff check habitat_demo` 通过。
- 真实集成：在现有 Python 3.9 Habitat 环境中创建 HM3D `00337-CFVBbU9Rsyb` 的 640×480 RGB-D 仿真，使用固定测试策略执行 30 步。未启动真实 LightNav/OmniVLA 权重，未调用云端。自动启动的 Places365 进程返回有效概率，Habitat 进程中未导入 torch；再次推理复用同一进程，单帧热推理约 37.81 ms（单次观测，不是性能基准）。
- 本次共保留 14 个采样轨迹点，快速假策略压力下有界观察队列跳过 17 帧；该指标已显示在网页。获得 1 个 provisional 节点和 5 张代表视角。房间标签为 unknown，聚合分数中 unknown 约 0.406、corridor 约 0.255、staircase 约 0.230，没有达到默认 0.55 的已知标签阈值。未调整阈值来强行命名房间。
- Chromium 连接真实临时 HTTP 服务，确认 Places365 状态、轨迹/节点、代表图加载和导航按钮可用，无 JavaScript 异常；JSON 导出和代表图接口可用。重置建立新 session，退出后推理子进程已回收，临时服务已关闭。

记录：[places365-smoke.json](places365-smoke.json)；页面截图：[places365-web.png](places365-web.png)。以上验证本地分类接入、生命周期与网页链路；没有房间人工真值，不能据此声称识别准确率已达标。

复现自动测试（仓库根目录）：

```bash
PYTHONPATH=habitat_demo PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
TEST_CHROMIUM=/home/xrq/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome \
  mujoco_demo/.venv/bin/python -m pytest -p pytest_asyncio.plugin habitat_demo/tests -q
```

## 初版几何与千问适配器验证（替换前）

- 90 项 Habitat demo 自动测试通过（含 Chromium）；新增覆盖前向轴向深度投影、无效深度不可算自由空间、地板/阻塞拒绝、门宽阈值、门户稳定匹配、真实跨越迟滞、已确认候选不被后续开口覆盖、同门户返回复用 R0、语义变更不换房、有限轨迹窗口与完整采样归档。
- 慢语义 API 测试：几何继续更新；导航请求不等 API；重置后旧结果不污染新回合；关机取消未完成请求。
- 固定局部策略输出下，启用/关闭房间记忆执行的速度命令序列完全相同。
- Qwen 协议用本地假服务验证多图 payload、JSON 输出、概率校验、代表图 ID、超时和失败。**环境未配置真实 API key，因此未验证真实房间标签质量或实际云端请求成功。**
- 网页测试：独立轮询、失败/超时恢复、2D Y 轴与朝向、长语义字符串按文字显示、选中房间与代表图、桌面/手机布局、原有导航按钮与 APOS/OPOS 兼容。
- 真实网页联调：在独立临时 18090 端口启动 HM3D + Depth + Room Memory，接本地固定路径假模型，完成 30 次导航动作、20 个采样轨迹点、R0 代表图、APOS 叠加与新 session 重置；无 JavaScript 异常，丢帧为 0。截图 `room-memory-web.png`、记录 `room-memory-web-smoke.json`。修复了 Python 3.9 中启动前创建 asyncio.Queue 导致与 aiohttp 事件循环不一致的问题；队列和控制锁现在在异步启动阶段初始化。测试服务已关闭，未启动真实模型或调用云端。

## HM3D 实际 RGB-D 回放

场景：`00337-CFVBbU9Rsyb`，官方 example scene dataset config；640×480 RGB+Depth，0.45 m 相机高度，seed=0。验证脚本使用 GT navmesh 找约 25.38 m 路径并原路返回，以 0.04 m 采样回放，共 1212 次 RGB-D 观测。

初始 3 帧门口确认阈值漏掉了该路线的短暂开口；调试还发现候选覆盖导致穿越证据丢失，已增加对应回归测试并保留已确认候选。当前默认至少 2 帧高质量几何观测，加实际跨面超过 0.20 m；该阈值需要在更多带人工标注数据上评估误检/漏检权衡。

最终结果：2 个几何节点、1 条门户边，门户穿越计数 2，返回原 R0，R0 visit_count=2，R1 visit_count=1。两房均保留 partial/unknown，未调用云端、未虚构语义标签。记忆处理（不含 Habitat 渲染及模型推理）平均约 10.7 ms，95 分位约 15.7 ms。结果见 `room-memory-replay.json`。

**这是无人工房间真值的链路验证，不是建筑房间分割准确率评测。** 不能据此推断所有门口都被找到。第一版仍会漏掉斜门、视野外门、遮挡和开放式区域分界；几何窄口也可能位于同一真实房间内。

复现命令（在仓库根目录）：

```bash
conda activate omnivla-habitat
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=habitat_demo \
  python habitat_demo/tools/smoke_room_memory.py \
  --scene /home/xrq/datasets/habitat/scene_datasets/hm3d/example/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb \
  --scene-dataset-config /home/xrq/datasets/habitat/scene_datasets/hm3d/example/hm3d_example_basis.scene_dataset_config.json \
  --output /tmp/room-memory-replay --samples 1500
```

## 配置和性能边界

关闭 `--room-memory` 时不创建 Depth sensor 或后台记忆 worker。开启时增加深度渲染、有限 CPU 计算和页面轮询，不能宣称零开销；排队过载记录 dropped_frames 并保持有界，不阻塞策略控制循环。长时间运行还需独立评估 GPU 渲染开销和磁盘归档增长。

原有 `src/lightnav`、MuJoCo 控制器及 OmniVLA 模型源码均未修改。验证没有启动 ROS 或真实机器人。
