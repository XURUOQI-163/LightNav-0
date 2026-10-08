# Habitat 简洁交互页面

已确认范围：复用 MuJoCo 页面风格，服务器无窗口 Habitat，语言输入与 RGB，选择完整 OmniVLA 或 LightNav，开始/停止/重置；不接入地图、雷达、复杂暂停状态机或 ROS。所有新代码放此独立目录，保留现有 MuJoCo 和模型代码。

实现与验证：
1. 单线程环境执行器管理 Habitat/EGL 创建、步进、渲染、关闭，保证上下文线程一致。
2. 异步模型客户端分别适配 LightNav WebSocket 和纯 OmniVLA HTTP，将输出统一为米/左正/弧度轨迹。
3. 每轮观测→推理→有限时间动作，不在推理等待时继续行驶；停止/重置取消旧结果，页面轮询不驱动仿真。
4. 独立纯 OmniVLA 服务只调用原始 full 模型接口，不导入 limo_adapter/semantic_topology。
5. 页面复用视觉风格，支持显示连接错误、运行状态和推理延迟；HTTP 控制与最新 JPEG。
6. CPU 协议和生命周期测试、真实浏览器测试、现有场景的 headless Habitat 实测；模型可用时分别做真实推理联调。

不将这套简易速度控制称为官方评测或 Limo 动力学复现。比较模型应保持场景、起点、控制周期、速度限制相同。无需新增数据集下载。

完成：上述 1–6 项均已实现。纯 OmniVLA 与原 LightNav 分别通过真实短程闭环；浏览器和 16 项自动测试通过。状态与限制见 docs/VERIFICATION.md。

## 后续任务状态（2026-10-02）

房间级语义拓扑记忆已接入独立 `room_memory/` 与网页。最新要求是将千问替换为用户已下载的 Places365，封装 `room_memory/places365_recognizer.py` 并检查是否可共用环境。

已完成：Places365 作为默认后端，复用已有 `places365` 环境的常驻 CPU 子进程；本机 Habitat 没有 torch/torchvision，因此保持两套环境，启动时由程序自动管理。100 项测试与 Ruff 通过，真实 HM3D + Places365 + Chromium 联调、重置与退出清理通过，配置和启动文档已更新。验证数据及识别质量边界见 `docs/ROOM_MEMORY_VERIFICATION.md`。临时验证进程已关闭；正常服务需按 README 启动。
