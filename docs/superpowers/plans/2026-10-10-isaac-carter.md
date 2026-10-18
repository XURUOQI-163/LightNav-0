# Isaac Carter Integration Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. User authorized direct implementation.

**Goal:** 用真实 Carter 物理模型在 GRScenes 中复用导航网页、模型服务与房间记忆。
**Architecture:** 独立 Isaac spawn worker + 同步 Scene 接口，复用 Habitat Runtime 和 HTTP UI。
**Tech Stack:** Isaac Sim 5.0 / Python 3.11 / aiohttp / numpy / existing Room Memory.
**Spec:** `isaac_demo/docs/design.md`

## Global Constraints
- 不修改 LightNav/OmniVLA 模型；不改 Habitat 控制逻辑。
- 不修改共享驱动、其他用户进程、源 USD，不引入雷达。
- 用户要求直接完成；在当前干净工作区实施，保留可审阅 diff，不额外提交或创建工作树。

## Review Focus
- 深度光轴与欧氏距离混用：几何单元测试和实际 depth 统计。
- world/up/unit、轮关节符号错误：纯坐标测试和实物理前进/转向验证。
- stop/reset 遇到延迟 worker：有限步进、串行锁和超时清理测试。
- 开始自动导航与手动操作竞争：HTTP 测试必须拒绝并发驱动。
- 迁移后路径、加载超时与黑画面：实际启动、截图与明确诊断。

### Task 1: Scene 协议和坐标
- [x] 写 geometry/PoseSource/worker timeout 测试并看到失败。
- [x] 新建 isaac_web/geometry.py、bridge.py、worker.py、scene.py。
- [ ] 实际加载 Carter 与默认家居场景，确认出生点、RGB-D 和轮控方向。

### Task 2: 网页与 Room Memory
- [x] 写手动/自动互斥、HTTP RGB-D 和停止测试并看到失败。
- [x] 新建 isaac_web/runtime.py、__main__.py、run.sh、配置文件。
- [x] 复用 server/static，只新增可选手动入口和动态仿真名称。
- [ ] 运行 habitat_demo/tests + isaac_demo/tests；真实网页与语义子进程验证。

### Task 3: 交付和审阅
- [ ] 真实模型连接与有限步试验；保留截图和 JSON 验证记录。
- [x] 写安装手册2.md；更新命令行总结及 CLI 配置文档。
- [ ] 独立审阅、修复重要问题，最终汇报证据和限制。

## 实施记录
- 初始 git diff 为空；当前两张 GPU 有其他工作，所有试验仅操作本任务进程。
- 基线测试遇到系统 ROS pytest 自动插件污染（缺 lark）；使用 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 显式加载 pytest_asyncio，不安装 ROS 依赖。
- 独立审阅完成：修复手动期间开始请求排队、初始化取消 Scene 丢失、地面外参与车身 pose 原点不一致、源 root 已有缩放冲突；加严格 GPU/像素整数验证。
- 2026-10-10 CPU/Chromium 回归 211 passed、1 skipped；跳过的 USD 模块单独加载 Isaac 自带 pxr 后 2 passed。Ruff 对全部新增 Python 和修改 Python 通过，JS 语法与 git diff --check 通过。
- 真正落地验证曾发现空父 prim 碰撞不继承：补齐 4418 个实际静态网格后，新出生点 Carter z≈0.240m。仍在等待最终 RGB-D/物理运动闭环检查。
- 2026-10-11 恢复检查：后台测试已停止且未留下 RGB-D 报告。系统内核 NVIDIA 580.173.02、NVML/libcuda 580.178.04；实际 Isaac 启动出现 NVML_ERROR_LIB_RM_VERSION_MISMATCH、GPU Foundation is not initialized，真实渲染被系统状态阻塞。未改驱动、未启动其他用户操作；手册2与命令行总结已补充确切验证状态和管理员处理后的继续步骤。最终物理运动/导航验收仍未完成，不标完成。
