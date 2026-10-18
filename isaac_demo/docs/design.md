# Carter / GRScenes 网页适配设计

保留 `habitat_demo/habitat_web` 的模型客户端、路径到速度换算、单步推理循环和网页。
新增 `isaac_demo/isaac_web`：主进程处理 HTTP 和 Room Memory；spawn 子进程在主线程创建
Isaac Sim 5.0、PhysX Carter 和相机。使用仅父子可见的 Pipe 传同步观测，不监听额外控制端口。

每次 step(v, w, dt) 将速度转换为轮关节目标，推进有限物理步，刹车后返回同一时刻的 RGB-D/GT。
等待模型时不推进物理时间，浏览器刷新也不推进。新增有限时长手动步进以独立检查车辆和画面。
停止/重置必须等当前步结束；关闭或进程错误清理自己启动的 worker。

输出世界坐标 X/Y 水平、Z 向上，机器人 +X 前、+Y 左；yaw 绕 +Z 逆时针。
深度采用 distance_to_image_plane（光轴 Z 深度、米），与 RGB 同相机。
用 IsaacGTPoseSource 直接映射 XY，不伪装成 Habitat 的 Y-up 坐标。
相机 optical→robot 外参以及实际内参一并输出，房间记忆核心不改。

GRScenes 与 Carter 只引用原始资产，场景修改在匿名运行层；不改下载的 USD。
默认只绑定一个 GPU，CPU PhysX；不改变环境包、驱动、其他用户进程。
出生位置需要由场景配置指定并用实际落地和 RGB 验证，不能随意将原点当可通行处。
复用 CLIP/Places365/YOLOE 各自子进程环境；新配置使用数据盘路径。

交付：可启动入口、默认家居场景配置、网页与 API、CPU 回归测试、真实 RGB-D/运动验证记录、安装手册2。
