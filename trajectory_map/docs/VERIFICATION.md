# 验证记录

日期：2026-09-24。本次只验证 MuJoCo 和独立轨迹模块，未启动实车、ROS EKF、Habitat 或 GPU 推理服务。

## 实际浏览器验证

在测试端口 8098 启动 MuJoCo，使用已有 Chromium 和 Playwright 执行：

- 相机图像实际加载成功。
- 通过网页 Take control + WASD 完成前进、转弯和再次前进。
- 路线从 1 点增长到 16 点，当前相对位置约 (1.355m, 0.485m)，yaw 约 0.765rad。
- STOP 保留路线；刷新网页保留同一 session 和路线。
- RESET 切换 session，路线回到 1 个新起点。
- 390px 手机视口没有横向溢出。
- 浏览器未报告 JavaScript 异常。

证据：[桌面截图](mujoco_trajectory.png)、[手机截图](mujoco_trajectory_mobile.png)。截图使用仿真真实位姿，不是 EKF 输出；没有借用雷达或特殊路口节点。

## 自动化与打包

核心、适配器、HTTP/RESET 集成及 Chromium 绘制测试：21 项。
原 MuJoCo demo 测试：27 项。
最终合跑结果：**48 passed in 33.49s**。命令：

```bash
cd /home/xrq/projects/LightNav-0
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  TRAJECTORY_BROWSER=/home/xrq/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome \
  mujoco_demo/.venv/bin/python -m pytest -p pytest_asyncio.plugin \
  --import-mode=importlib -c mujoco_demo/pyproject.toml \
  trajectory_map/tests mujoco_demo/tests -q
```

独立 wheel 构建成功，检查包含 trajectory_map.js 和 trajectory_map.css；核心通过 Python 3.8 语法解析检查，执行测试使用 Python 3.11，未声明在真实车载 Python/ROS 环境完成运行验证。

对新模块、测试和修改的 server.py 运行 Ruff 通过；git diff --check 通过。
全仓 Ruff 检查发现以下既有 I001 导入排序问题，对 git show HEAD 的原文件执行同样检查可复现，未修改这些无关文件：

- mujoco_demo/tests/test_control.py
- mujoco_demo/tests/test_vln_client.py
- mujoco_demo/tests/test_vln_protocol.py
- tests/test_fp8_quant.py

## 独立审查

只读审查未发现阻塞问题。指出的浏览器 bfcache 恢复问题已增加失败用例并修复；设计文档的待确认状态已更新。
