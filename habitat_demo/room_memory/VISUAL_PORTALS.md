# 视觉发现门口与候选房间

在原 `room_memory` 上增加门口生命周期，未修改 LightNav、OmniVLA 推理或 Habitat 导航/控制器。模块只观察同一时刻的 RGB、depth、pose，不发动作，也不安排探索。

## 当前行为

1. 常驻 YOLOE-26s 子进程检测 RGB 开口候选，输出原图像素坐标。
2. 对应源帧深度过滤无效值，估计门口平面、宽度、门后可通行区域。允许单侧可见墙条，不依赖旧双侧门框检测成功。默认 Habitat 深度上限 15 m；这里是真实测到的远处深度，不将空洞或截断值当作自由空间。
3. 第一帧可信 RGB+depth 提议即建立蓝色 P0 和未进入 R1。跨帧世界坐标去重、更新置信度；同名房间不会自动合并。
4. 门内 ROI 单独给 Places365。类别分数达标时显示 `office?` 等；证据不足保留 `未知房间?`。标签变化不创建新节点。
5. 连续位姿穿过门口平面、满足门宽范围及两侧 0.20 m 迟滞后，P0 变红，R1 原编号转为正式节点。重新采集正常运动的多视角，语义请求 epoch 阻止迟到的门外分类覆盖正式分类。
6. 从同一 P0 返回使用原 R0；红色只代表穿越过，`search_count` 仍为预留的 0。候选节点是门后的示意位置，不宣称恢复了房间中心或轮廓。

`geometry.py` 的原检测器完整保留，继续输出辅助诊断；其 `portal_confirm_frames` 不再限制视觉节点建立。配置 `doorway.enabled=false` 可切回旧纯深度模式。去掉 `--room-memory` 可关闭整个模块。

## 文件与接口

| 文件 | 职责 |
| --- | --- |
| `doorway_detector.py` | YOLOE-26s 模型校验、文本提示缓存、持久 stdio 子进程、超时/取消清理、单图检查 CLI |
| `doorway_depth.py` | RGB 检测框到米制门口的辅助验证，不存原始点云 |
| `frontiers.py` | 世界去重、候选节点、门内 ROI、实际穿越、同编号晋升、已知门户返回 |
| `models.py` | 原 RoomNode/PortalEdge 的兼容扩展、ProspectiveRoom |
| `graph.py` | 原轨迹/图/JSON 保持，增加 `observe_portals(frame, measurements)` 和语义 epoch |
| `service.py` | 有界异步检测队列，源帧 pose 对齐，session 隔离；导航不等待检测 |
| `config/default.json` | `doorway` 配置与原 memory/geometry/semantic 配置 |
| `habitat_web/web/room_map.js` | 蓝/红门户、候选和正式节点、详情/代表图、JSON 导出 |
| `tools/replay_visible_portals.py` | 实际 Habitat RGB-D + YOLOE 离线回放，仅用于验证 |

JSON `schema_version=2`：`rooms` 存正式节点，`prospective_rooms` 存候选；两类都由 RoomGraphManager 管理。`PortalEdge` 新字段有 `status`、`observation_count`、`first_seen_time`、`last_seen_time`、`source`、`geometry_supported`。旧 `room_a/room_b`、轨迹、session、代表帧接口保留。

检测结果只绑定源 RGB-D 时间戳。若推理返回时已发生穿越，使用有界近期连续位姿补记，并纠正采样轨迹的 room_id；位姿跳跃、观测空洞不能算穿越。队列满时丢弃旧待处理帧，错误通过网页与 JSON 显示。首次载入、CPU 计算、深度渲染均有开销，异步并不代表零开销。

## 本机环境与运行

`omnivla-habitat` 运行仿真与核心；已有 `navigation` 环境运行 Ultralytics 8.4.78 / YOLOE-26s；已有 `places365` 环境运行房间语义。无需激活后两者，主程序自动启动本地子进程，不开额外端口，不合并或升级现有 conda 环境。

官方模型是 `yoloe-26s-seg.pt`，不是 YOLO26 的普通检测权重。模型、文本编码器和提示缓存放在本项目忽略目录 `.models/yoloe/`。正常运行只读取本地 YOLOE 权重和缓存，不联网下载，不自动安装包。接口依据 [Ultralytics YOLOE 文档](https://docs.ultralytics.com/models/yoloe/)。

本机资产已准备，直接使用 [命令行总结](/home/xrq/projects/命令行总结.md) 的两个终端命令。更改代码/JSON 后停止原 Habitat 终端（Ctrl+C），重启再刷新页面。无需重启未改动的 LightNav 模型服务。

单图检查（只输出 RGB 检测，不替代深度验证）：

```bash
cd ~/projects/LightNav-0/habitat_demo
/home/xrq/.conda/envs/navigation/bin/python -m room_memory.doorway_detector \
  --config room_memory/config/default.json --image /实际路径/原始RGB.jpg
```

只有改动提示词、权重或 Ultralytics 版本后才重新准备缓存。当前隔离的 `.models/yoloe/runtime/` 使用已有 navigation 包和独立的文本编码器依赖；正常推理不依赖这个准备环境：

```bash
cd ~/projects/LightNav-0/habitat_demo
.models/yoloe/runtime/bin/python -m room_memory.doorway_detector \
  --prepare --config room_memory/config/default.json
```

不能任意修改 labels 而继续使用旧缓存，程序会明确报不匹配。新机器需另行准备兼容 Ultralytics、官方模型及文本编码器；不要把本机绝对路径照搬到其他机器。

## 验证范围与局限

详见 [真实回放和网页验证记录](../docs/VISUAL_PORTALS_VERIFICATION.md)。已验证旧路线提前发现一个未进入门口并多帧去重，以及另一条可通行回放路线穿越后变红、原路返回复用 R0。

原路线实际通过的侧门仍被 RGB 模型漏检。视野中只剩局部门板、门槛被截断、深度量程不足或开口过窄时也可能不建节点。此版本改善了提前发现和生命周期，不能保证所有门的召回率。不同门通向同一房间的候选关联、建筑学房间分割和多楼层不是当前实现的可靠能力。

Places365 原始场景概率尚未校准；实际回放的分类低于阈值时保持 unknown。没有为了让图上出现房间名字而降低阈值，也没有使用场景的语义传感器或真实房间标签。代表 ROI 可含门框和走廊，只是暂定证据。
