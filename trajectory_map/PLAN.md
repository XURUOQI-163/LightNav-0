# 实时轨迹图实施计划

依据 DESIGN.md，用户已确认按 MuJoCo 位姿首版实施。当前会话在现有功能分支 scout-lightnav 实现，保留用户已有文件。

约束：核心 Python >=3.8、仅标准库；不导入 ROS/MuJoCo/模型。Pose2D(stamp_ns,x_m,y_m,yaw_rad,frame_id,source) → TrajectoryMap.update → snapshot；reset 开新 session。Odometry 提供纯转换器，Habitat 保留同一输入契约。

- [x] 核心与适配器：先测位移、旋转、重置、时序、无效数值、分段、容量、过期；再实现独立模型与打包。
- [x] MuJoCo 接入：先测 HTTP 快照、独立采样及 RESET；再添加 GET /api/trajectory、静态资源及采样接入。
- [x] 浏览器：相机下方 Canvas，5Hz 单请求轮询；等比例、高清、起点/路线/当前箭头与过期状态；实际浏览器测试。
- [x] 验证交付：独立与 demo 全套测试、打包资源检查、MuJoCo 浏览器运动/刷新/重置验证和截图，README 与输入接口示例。

审查重点：reset 旧响应、推理误影响记录、旋转箭头冻结、断点容量、网络断开误显示新鲜。

记录：执行已确认方案，不增加再次授权步骤；不运行实车、不推送、不提交用户文件。独立核心与宿主接入顺序实现，最后独立审查。

验证记录见 docs/VERIFICATION.md；核心和接入测试已完成红→绿验证。基线27项通过，新增21项通过。最终合跑 48 passed in 33.49s，详情见验证记录。
