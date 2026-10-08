# 验证记录（2026-10-01）

- 自动测试：22 项 Python 测试和 1 项 Chromium 浏览器测试通过（23 passed）。包含场景配置参数及非法配置拒绝、坐标/单位转换、限速、非法返回、LightNav 会话/序号、纯 OmniVLA HTTP 协议、重复启动、推理中停止、环境步进中停止、重置和 HTTP 页面接口。没有调用 ROS 或真实小车。
- 新目录 Python 文件 Ruff 检查通过。
- Habitat 实测：使用现有 `omnivla-habitat` Python 3.9 环境、EGL 和 `skokloster-castle.glb`/navmesh，真实 RGB 渲染、动作执行、同起点重置通过。已有 Habitat/模型依赖未升级，仅在 Habitat 环境安装了 aiohttp 及其轻量依赖和此独立包。
- LightNav：2026-09-30 通过原有 8050 服务完成真实 3 步闭环，收到 10 个路径点，仿真时间 1 秒，最后一次服务端延迟约 276 ms。此后用户服务已停止，没有为后续测试重启它；协议回归使用假服务验证。
- 完整 OmniVLA：加载现有本地 `omnivla-original` 权重，调用原始 RGB+语言前向，真实 3 步闭环通过，返回 8 个路径点，仿真时间 1 秒、位移约 0.15 m，最后一次模型耗时约 159 ms。完整状态见 `omnivla-smoke.json`。
- 浏览器 Chromium：真实页面相机画面、OmniVLA 开始/结束、LightNav 服务缺失后的错误恢复、模型切换、重置、1360px 桌面和 390px 手机布局通过；无 JavaScript 异常。
- 审阅发现并修复：停止恰逢仿真步进开始时，等待 C++/EGL 完成并发布最终图像、位姿及步数，避免下一次导航使用旧观测。对应回归测试已覆盖。
- APOS/OPOS 增量验证：使用模拟模型返回值和图像，验证两个点与推理输入帧绑定；Chromium 实际运行页面，验证 16:9 图像在 4:3 容器内的留边对齐、1280px/390px 布局、目标不可见时只显示有效点、重置和切换 OmniVLA 后清除标记，无 JavaScript 异常。本次未重启真实模型，未增加 GPU 推理调用。
- HM3D 配置增量验证：已下载的 `00337-CFVBbU9Rsyb`、`00770-NBg5UqG3di3`、`00861-GLAQ4DNUx5U` 均携带官方 example 配置在真实 Habitat/EGL 下完成 640×480 RGB 渲染、三次固定速度步进（合计 1 仿真秒）和精确起点重置；起终点位于 navmesh。随后从 `run.sh --scene-dataset-config ...` 启动临时 18090 页面，在 Chromium 验证住宅走廊图像、输入框、可用的开始按钮与重置，无 JavaScript 异常，截图见 `hm3d-web.png`。此项没有调用模型或测量导航成功率。临时服务验证后关闭，用户 8090 和 8050 服务未重启。

图片：`desktop.png` / `mobile.png`。短程测试仅证明链路与页面可用，不代表任务成功率，也不证明达到了原模型官方导航指标。两个模型共用此演示的简易控制器，不使用原 MuJoCo MPC。

测试服务端口使用 18090 和 18991，交付前关闭；用户正常启动端口为 8090 和 8891。没有替用户启动或改动原 8088/8050 服务。

## 自动测试命令

在仓库根目录，使用已有 MuJoCo 测试环境（只作为测试解释器，不加载 MuJoCo）：

```bash
PYTHONPATH=habitat_demo PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  mujoco_demo/.venv/bin/python -m pytest -p pytest_asyncio.plugin habitat_demo/tests -q
```

浏览器测试需额外设置 `TEST_CHROMIUM` 为已安装的 Chromium 可执行文件路径，并在测试解释器中安装 Playwright；未设置时只跳过此项。本机使用 `/home/xrq/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome`。
