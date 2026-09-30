# 独立实时轨迹图：设计方案

日期：2026-09-24。状态：用户已确认，按 MuJoCo 仿真位姿完成首版实现；实车与 Habitat 通过统一接口后续接入。

## 1. 目标与范围

在现有 MuJoCo 浏览器页面增加实时二维轨迹图，只显示：

- 起点：本轮第一条有效位姿，显示为固定标记。
- 路线：实际位置观测形成的历史折线。
- 当前机器人：位于最新位置、指向最新航向的箭头。

本功能不接雷达，不读取 scan，不创建路口/拐角/死胡同/Frontier，不设计特殊节点触发、语义识别或选路。保留现有相机画面和导航操作，轨迹记录独立于模型推理成功与否，手动移动也要显示。

OmniVLA 的原始文件保持不变；本次不导入其整个 semantic_topology 包。“删掉特殊节点”落实为新模块和新页面完全不包含这些节点，而非破坏 OmniVLA 的既有工程。

## 2. 现有 EKF 的角色

已阅读：

- `../../OmniVLA/localization_v1/odom_covariance_adapter.py`
- `../../OmniVLA/localization_v1/config/odom_covariance_adapter.yaml`
- `../../OmniVLA/localization_v1/config/ekf_limo_v1.yaml`

它们建立以下链路：

```text
/odom → 修补平面位姿/速度协方差 → /odom_ekf_input ─┐
/imu ────────────────────────────────────────┤
                                    robot_localization EKF
                                              ↓
                                    /odometry/filtered
                                              ↓
                                      位姿输入适配 → 轨迹图
```

这里使用 ROS 2 robot_localization 的 ekf_node，不另行引入一个名为 EKF2 的滤波器。现有配置 two_d_mode=true，融合轮式 x/y/yaw、vx/vy 与 IMU 绕 Z 轴角速度，publish_tf=false。轨迹模块不修改协方差、不积分控制速度、不重复做传感器融合。

首版已确认使用 MuJoCo 已有真实位姿验证轨迹功能，界面标注“MuJoCo 仿真位姿”，不能把它标为 EKF 输出。仿真轮式 odom/IMU 并运行 ROS EKF 不属于本次范围。

## 3. 推荐结构与取舍

选用独立轻量 Python 包和独立浏览器绘图组件，代码全部归入根目录 `trajectory_map/`。现有 demo 只增加接入代码与页面容器。

不把轨迹历史只放浏览器：否则刷新页面会丢路线，多个浏览器状态不同。
不复制整个 OmniVLA 拓扑模块：其依赖与控制状态机超出当前三个显示要素的需求。

计划目录：

```text
trajectory_map/
  DESIGN.md
  README.md
  pyproject.toml
  src/lightnav_trajectory/
    __init__.py
    model.py                 位姿、轨迹状态、序列化与重置
    adapters.py              MuJoCo/ROS Odometry 数据转换，不直接依赖 ROS
    web/
      trajectory_map.js      可复用 Canvas 绘制组件
      trajectory_map.css     独立样式
  tests/
    test_model.py
    test_adapters.py
    test_browser.py          Playwright 驱动真实 Chromium 检查绘制
    test_mujoco_integration.py
```

模块不导入 MuJoCo、ROS、torch、OmniVLA 或 LightNav 推理代码。模型层使用 Python 标准库，兼顾未来车载较旧 Python 环境。MuJoCo demo 通过本地包依赖安装并打包读取组件资源，不依赖当前工作目录或 sys.path 临时修改。

需修改的现有文件：

- `mujoco_demo/pyproject.toml`、`uv.lock`：本地模块依赖。
- `mujoco_demo/vln_mujoco/server.py`：采样位姿、提供轨迹快照、接入 RESET、托管绘图资源。
- `mujoco_demo/vln_mujoco/web/index.html`：相机下方增加轨迹图区域。
- `mujoco_demo/vln_mujoco/web/app.js`：将轨迹快照交给独立绘图组件，处理断线状态。
- 必要的页面布局样式和集成测试。

## 4. 统一位姿接口

基本输入：

```python
Pose2D(stamp_ns, x_m, y_m, yaw_rad, frame_id, source)
TrajectoryMap.update(pose)
TrajectoryMap.snapshot()
TrajectoryMap.reset()
```

约定：

- x/y 单位米，yaw 单位弧度；yaw=0 指向 +X，正方向逆时针。
- stamp_ns 是该来源采样时刻的整数纳秒，不得在不同来源间混用时钟。
- 一个记录会话内 frame_id 和 source 固定；切换来源或坐标系先 reset。
- 第一条有效位姿建立平移原点，显示坐标为 x-x0、y-y0；保持来源坐标轴方向，不把初始朝向强行旋转为北向。
- 起点的原始位置保存在快照中，便于还原坐标。
- 最新箭头始终使用当前有效位姿；小于路线采样距离的移动或原地转向也更新箭头。

适配方式：

- MuJoCo：读取 `Simulation.snapshot()` 的 pose 和 sim_time，作为同一次快照输入。
- ROS：将 `/odometry/filtered` 的 position、四元数、header.stamp/frame_id 转成 Pose2D；提供不依赖 rclpy 的转换函数与接入示例，后续 ROS 订阅由车载桥接层负责。
- Habitat：预留同一 Pose2D 输入契约。调用方先将 Habitat 坐标、朝向和仿真时间转换到该约定；本次不声称已经实现 Habitat 运行桥接。

模块不向底盘或仿真发布运动指令。

## 5. 路线记录与异常行为

- 后端记录历史，浏览器断开、刷新不清空，重连获得完整快照。
- 路线默认每移动 0.05m 追加点；当前末端连接到最新位置，使低速显示连续。
- 起点永久保留到 reset。停车、停止 VLN、切换手动/自动不会清空地图。
- RESET 与仿真 reset 同步，删除旧轨迹，从复位后的有效位姿建立新起点，绝不连线回旧位置。
- 重复和倒序时间戳不追加、不覆盖新位姿；时间倒退需要显式 reset，防止乱序消息破坏路线。
- 拒绝 NaN/Inf、错误 frame/source 和无效时间输入，保留最后有效状态。
- 明显位姿跳变分段显示，不画穿过场景的虚假连接线；它只是路径数据断点，不是特殊拓扑节点。
- 后端设置总点数上限 4096；超过时对旧路线降采样，保留起点、各段边界和当前点，并记录降采样状态。段数同样受整体存储上限约束，过多异常分段停止增长并报告状态，不能静默连接不相邻位置。
- 记录最新有效接收的单调时钟时间，输入停止后将箭头显示为“位姿未更新”；不把正常静止误报为断流。

初始跳变阈值为单次 2m，作为配置项提供；仅用于避免遥移连线，不用于检测路口。与真实最大速度/采样间隔的进一步联动不属于首版。

## 6. 网页表现与通信

相机下方增加二维轨迹面板：深色背景、米制网格、绿色起点、蓝色路线和高对比朝向箭头；提供文本图例、当前 x/y/yaw 和定位来源。

- 地图自动适配完整已记录范围，x/y 等比例，起点与当前位置始终可见。
- Canvas 的屏幕 y 向下，绘图组件负责翻转；yaw 正方向在图上仍为逆时针。
- 原地旋转只转动箭头，不制造路线圈。
- 支持窗口尺寸变化与高 DPI；没有位姿时显示等待提示。
- 使用只读 `GET /api/trajectory` 返回完整版本化快照；首版页面按 5Hz 获取，最多一个请求在途，不把整条路线塞进现有高频 runtime 广播。
- 响应含 schema_version、session_id、revision、source、frame_id、origin、start、segments、current 和 freshness 状态。
- 使用 no-store 防止旧快照缓存；reset 的 session_id 使浏览器丢弃旧会话的迟到返回。
- 多个浏览器均读取同一后端状态，无独立客户端轨迹累计。

## 7. 验证与交付标准

单元测试：原点平移、朝向和角度边界、直线和转弯、原地旋转、静止不重复增点、无效数值、倒序帧、坐标系更换、跳变分段、容量控制、重置、输入过期。

适配测试：ROS 四元数→yaw、时间戳保留；MuJoCo pose/sim_time 转换；使用伪消息，不要求 ROS 安装。

前端测试：坐标等比例、y 轴翻转、箭头方向、空数据、断流显示、resize 和快照会话更换。

集成测试：HTTP 快照、手动/自动均采样、无模型服务器仍能显示路线、RESET 后旧轨迹不回流、浏览器重连完整恢复。

实际 MuJoCo 验证：启动独立测试端口，检查页面与轨迹接口；通过仿真手动运动执行前进、转弯、停车、reset，确认路线和箭头对应。可用浏览器自动化时检查页面截图与控制台；若环境缺少图形渲染条件，明确记录未验证部分，不能以 HTTP 200 代替真实画面验证。

本次不启动实车，不要求运行 Habitat，不修改模型推理，不迁移雷达或特殊节点代码。交付独立目录、MuJoCo 页面集成、运行说明及未来 EKF/Habitat 接口说明。
