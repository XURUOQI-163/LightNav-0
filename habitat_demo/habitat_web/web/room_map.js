"use strict";

// The optional memory view owns its requests and errors independently of navigation.
(() => {
  const element = (id) => document.getElementById(id);
  const panel = element("room-memory-panel");
  if (!panel) return;
  const svg = element("room-map");
  const detail = element("room-detail-content");
  const frame = element("room-representative");
  const frameNote = element("room-representative-note");
  const svgNS = "http://www.w3.org/2000/svg";
  const semanticLabels = {
    configured: "语义识别已配置", unconfigured: "未配置语义 API", collecting: "采集代表视角中",
    no_api: "未配置语义 API", disabled: "语义识别未启用", pending: "等待语义识别",
    queued: "等待语义识别", processing: "语义识别中", ready: "已识别", ok: "已识别",
    error: "语义识别失败", failed: "语义识别失败", unknown: "尚未识别", tentative: "门口视角暂定分类",
  };
  const observationLabels = {
    glanced: "少量观测", unseen: "未观测", well_observed: "较充分观测",
    partial: "部分观测", observing: "观测中", observed: "已记录观测",
    unobserved: "尚未观测", entered: "已进入", exited: "已离开",
  };
  let snapshot = null;
  let selectedId = null;
  let selectedPortalId = null;
  let sessionId = null;
  let frameKey = null;
  let controller = null;
  let timer = null;
  let stopped = false;

  const hasXY = (point) => point && Number.isFinite(point.x) && Number.isFinite(point.y);
  const items = (value) => Array.isArray(value) ? value : [];
  const text = (id, value) => { element(id).textContent = value; };
  const hasLabel = (value) => value && value !== "unknown";
  const roomType = (value) => hasLabel(value) ? String(value) : "等待识别";
  const scoreText = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
  const providerName = (value) => ({ places365: "Places365", clip: "CLIP", qwen: "Qwen" })[value] || "场景识别";
  const semantic = (value) => semanticLabels[value] || value || "尚未识别";
  const poseText = (value) => hasXY(value) ? `(${value.x.toFixed(2)}, ${value.y.toFixed(2)}) m` : "—";
  const timeText = (value) => Number.isFinite(value) ? `${value.toFixed(1)} s` : "—";
  // A promoted room keeps its ID; a formal record wins if an update includes both.
  const allRooms = () => [...new Map([
    ...items(snapshot?.prospective_rooms), ...items(snapshot?.rooms),
  ].map((room) => [room.room_id, room])).values()];
  const prospective = (room) => room.topological_status === "prospective"
    || (!items(snapshot?.rooms).some((item) => item.room_id === room.room_id)
      && items(snapshot?.prospective_rooms).some((item) => item.room_id === room.room_id));
  const typeLabel = (room) => `${roomType(room.room_type)}${prospective(room) ? "?" : ""}`;
  const explored = (portal) => portal.status === "explored"
    || (portal.status !== "unexplored" && portal.traversal_count > 0);
  const portalState = (portal) => explored(portal) ? "已穿越" : "未进入";
  const regionPolygons = (room) => items(room.region_polygons).filter((p) =>
    items(p.outer).length >= 3 && p.outer.every(hasXY));
  const roomColor = (id) => {
    let hash = 0;
    for (const char of String(id)) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
    return `hsl(${(hash * 137.508) % 360} 68% 64%)`;
  };

  function renderLive(next) {
    const source = element("camera-views")?.classList.contains("three-views") ? "导航相机关键帧" : "当前画面";
    text("room-live-title", `${source} · ${providerName(next.semantic_backend)}`);
    const live = next.live_semantic;
    element("room-live").hidden = !next.enabled || live?.enabled === false;
    if (!next.enabled || live?.enabled === false) return;
    if (!live) {
      text("room-live-status", "需重启 Habitat 服务");
      text("room-live-result", "等待实时识别服务");
      text("room-live-meta", "页面已更新；请重启 Habitat 后刷新页面。");
      element("room-live-error").hidden = true;
      return;
    }
    const statuses = { waiting: "等待图像", processing: live.completed_count ? "识别中" : "识别中（首次含模型加载）", ready: "已更新", error: "识别失败" };
    const duration = Number.isFinite(live.processing_seconds) ? ` · ${live.processing_seconds.toFixed(1)} s` : "";
    text("room-live-status", `${statuses[live.status] || live.status}${live.status === "processing" ? duration : ""}`);
    const prefix = live.status === "ready" ? "Top-1" : "上次结果";
    text("room-live-result", hasLabel(live.room_type)
      ? `${prefix}：${live.room_type} · ${scoreText(live.confidence)}` : "等待识别结果");
    const latency = Number.isFinite(live.latency_ms) ? `${Math.round(live.latency_ms)} ms` : "—";
    const age = Number.isFinite(live.age_seconds) ? `${live.age_seconds.toFixed(1)} s 前采集` : "等待采集";
    text("room-live-meta", `源图像 ${timeText(live.source_timestamp)} · ${age} · 识别耗时 ${latency} · 已完成 ${live.completed_count ?? 0} 次`);
    element("room-live-error").hidden = !live.error;
    text("room-live-error", live.error || "");
  }

  function selectRoom(roomId, portalId = null) {
    selectedId = roomId;
    selectedPortalId = portalId;
    renderMap();
    renderDetail();
  }

  function selectable(node, select) {
    node.addEventListener("click", select);
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); }
    });
  }

  function shape(tag, attrs = {}, value = null) {
    const node = document.createElementNS(svgNS, tag);
    for (const [name, item] of Object.entries(attrs)) node.setAttribute(name, String(item));
    if (value !== null) node.textContent = value;
    return node;
  }

  function clearFrame() {
    frameKey = null;
    frame.hidden = true;
    frame.removeAttribute("src");
    frameNote.textContent = "暂无代表画面";
  }

  function renderDetail() {
    const room = allRooms().find((item) => item.room_id === selectedId);
    detail.hidden = !room;
    element("room-detail-empty").hidden = Boolean(room);
    text("room-detail-title", room ? `节点 ${room.room_id}${room.room_id === "R0" ? " · 起始空间" : prospective(room) ? " · 候选空间" : ""}` : "房间详情");
    if (!room) { clearFrame(); return; }
    text("room-detail-type", typeLabel(room));
    text("room-detail-semantic", room.semantic_status === "ready" && room.room_type === "unknown"
      ? snapshot.semantic_backend === "places365"
        ? "旧版分类结果；请重启 Habitat 开始新记录" : "分类已完成，暂无明确类别"
      : semantic(room.semantic_status || snapshot.semantic_status));
    text("room-detail-topology", prospective(room) ? "候选空间，尚未进入；位置仅为显示锚点" : room.topological_status === "provisional"
      ? "暂定空间，尚未确认门户穿越" : room.topological_status === "crossing_confirmed"
        ? "已记录门户穿越" : "—");
    text("room-detail-confidence", hasLabel(room.room_type) ? scoreText(room.semantic_confidence) : "—");
    text("room-detail-observation", observationLabels[room.observation_state] || room.observation_state || "未知");
    text("room-detail-boundary", ({ partial: "部分范围，边界未闭合", enclosed: "观测范围已有边界支持", unobserved: "尚无区域观测" })[room.boundary_state] || "尚无区域观测");
    text("room-detail-area", Number.isFinite(room.region_area_m2) ? `${room.region_area_m2.toFixed(2)} m²（非完整房间面积）` : "—");
    text("room-detail-geometry", Number.isFinite(room.geometry_confidence) ? `${Math.round(room.geometry_confidence * 100)}%（非完整度）` : "—");
    const entered = !prospective(room) && room.visited;
    text("room-detail-visits", `${entered ? "已进入" : "未进入"} · 进入 ${room.visit_count ?? (entered ? 1 : 0)} 次`);
    text("room-detail-time", `${timeText(room.first_seen_time)} / ${timeText(room.last_seen_time)}`);
    text("room-detail-entry", poseText(room.entry_pose));
    text("room-detail-anchor", prospective(room) ? poseText(room.candidate_pose) : "—");
    text("room-detail-connected", items(room.connected_room_ids).join("、") || "尚无已记录连接");
    const linkedPortals = items(snapshot.portals).filter((item) => item.room_a === room.room_id || item.room_b === room.room_id);
    const portal = linkedPortals.find((item) => item.portal_id === selectedPortalId) || linkedPortals[0];
    element("room-detail-portal").hidden = !portal;
    if (portal) {
      text("room-detail-portal-id", String(portal.portal_id));
      text("room-detail-portal-status", `${portalState(portal)} · 穿越 ${portal.traversal_count ?? 0} 次`);
      text("room-detail-portal-confidence", Number.isFinite(portal.confidence) ? `${Math.round(portal.confidence * 100)}%` : "未知");
      text("room-detail-portal-width", Number.isFinite(portal.width) ? `${portal.width.toFixed(2)} m` : "—");
    }
    const frameId = room.representative_frame_id || items(room.representative_frame_ids)[0];
    const nextKey = frameId ? `${snapshot.session_id}/${frameId}` : null;
    if (nextKey === frameKey) return;
    clearFrame();
    if (!frameId) return;
    frameKey = nextKey;
    frame.alt = `${room.room_id} 的代表观测画面`;
    frameNote.textContent = "正在加载代表画面…";
    frame.src = `/api/room-memory/frames/${encodeURIComponent(frameId)}.jpg`;
  }

  frame.addEventListener("load", () => {
    if (!frameKey) return;
    frame.hidden = false;
    frameNote.textContent = "代表观测画面 · 仅反映当时可见区域";
  });
  frame.addEventListener("error", () => {
    frame.hidden = true;
    frameNote.textContent = "代表画面暂不可用";
  });

  function renderMap() {
    if (!snapshot?.enabled || panel.hidden) return;
    const trajectory = items(snapshot.trajectory).filter(hasXY);
    const anchors = new Map();
    const rooms = allRooms();
    for (const room of rooms) {
      if (prospective(room)) {
        if (hasXY(room.candidate_pose)) anchors.set(room.room_id, room.candidate_pose);
      } else if (hasXY(room.entry_pose)) anchors.set(room.room_id, room.entry_pose);
      else {
        const indices = new Set(items(room.trajectory_indices));
        const points = trajectory.filter((point) => point.room_id === room.room_id || indices.has(point.index));
        if (points.length) anchors.set(room.room_id, {
          x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
          y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
        });
      }
    }
    const portals = items(snapshot.portals).filter(hasXY);
    const allPoints = [...trajectory, ...anchors.values(), ...portals];
    const showRegions = element("room-regions-toggle").checked;
    const semanticMap = snapshot.semantic_map;
    const showSemantics = !!semanticMap?.enabled && element("room-semantic-toggle").checked;
    const legend = element("room-semantic-legend");
    legend.replaceChildren();
    if (showSemantics) items(semanticMap.labels).forEach((label, index) => {
      const entry = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.style.backgroundColor = semanticMap.colors[index];
      entry.append(swatch, document.createTextNode(label));
      legend.append(entry);
    });
    const statuses = {waiting: "等待同步 RGB-D", processing: "正在分类图像", ready: "已融合",
      filtered: "当前视图因低分或类别接近，已跳过", error: "投影失败", no_visible_depth: "当前帧无可投影深度", geometry_disabled: "需启用二维几何地图"};
    text("room-semantic-status", semanticMap?.enabled
      ? `语义投影（${semanticMap.view_mode === 'full' ? '整图' : semanticMap.view_mode === 'left_right' ? '左右子图' : '视图'}）：${statuses[semanticMap.status] || semanticMap.status} · 有证据格子 ${semanticMap.evidence_cells || 0} · 最近过滤帧 ${timeText(semanticMap.filter_timestamp)} · 采纳 ${semanticMap.accepted_views || 0} / 过滤 ${semanticMap.rejected_views || 0} · 已融合源帧 ${timeText(semanticMap.source_timestamp)}${semanticMap.error ? " · " + semanticMap.error : ""}`
      : "语义投影未启用（需要 CLIP 与二维几何地图）");
    const bounds = semanticMap?.bounds;
    if (showSemantics && Array.isArray(bounds) && bounds.length === 4 && bounds.every(Number.isFinite)) {
      allPoints.push({x: bounds[0], y: bounds[1]}, {x: bounds[2], y: bounds[3]});
    }

    if (showRegions) for (const room of rooms) for (const polygon of regionPolygons(room)) allPoints.push(...polygon.outer);
    if (hasXY(snapshot.current_pose)) allPoints.push(snapshot.current_pose);
    if (hasXY(snapshot.start_pose)) allPoints.push(snapshot.start_pose);
    if (hasXY(snapshot.candidate)) allPoints.push(snapshot.candidate);
    const width = Math.max(280, svg.clientWidth);
    const height = Math.max(280, svg.clientHeight);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const focusedId = svg.contains(document.activeElement) ? document.activeElement.id : null;
    svg.replaceChildren();
    element("room-map-empty").hidden = allPoints.length > 0;
    if (!allPoints.length) return;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const point of allPoints) {
      minX = Math.min(minX, point.x); maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y); maxY = Math.max(maxY, point.y);
    }
    // A symmetric local XY transform preserves metric proportions and positive CCW yaw.
    const scale = Math.min((width - 128) / Math.max(2, maxX - minX), (height - 104) / Math.max(2, maxY - minY));
    const centerX = (minX + maxX) / 2, centerY = (minY + maxY) / 2;
    const project = (point) => ({ x: width / 2 + (point.x - centerX) * scale, y: height / 2 - (point.y - centerY) * scale });
    const position = (point) => { const p = project(point); return `${p.x},${p.y}`; };
    for (let x = 24; x < width; x += 32) svg.append(shape("line", { x1: x, y1: 0, x2: x, y2: height, class: "room-grid" }));
    for (let y = 24; y < height; y += 32) svg.append(shape("line", { x1: 0, y1: y, x2: width, y2: y, class: "room-grid" }));
    if (showRegions) for (const room of rooms) {
      const color = showSemantics ? "#64748b" : roomColor(room.room_id);
      for (const polygon of regionPolygons(room)) {
        const rings = [polygon.outer, ...items(polygon.holes).filter((r) => Array.isArray(r) && r.length >= 3 && r.every(hasXY))];
        const path = rings.map((ring) => `M${ring.map(position).join(" L")} Z`).join(" ");
        const region = shape("path", {
          d: path, "fill-rule": "evenodd", fill: color, stroke: color,
          class: `room-region ${room.boundary_state === "enclosed" ? "is-enclosed" : "is-partial"}${prospective(room) ? " is-prospective" : ""}${room.room_id === selectedId ? " is-selected" : ""}`,
          "data-room-id": room.room_id, role: "button", tabindex: 0,
          "aria-label": `${room.room_id} · ${typeLabel(room)} · 房间观测范围`,
        });
        region.append(shape("title", {}, `${room.room_id} · ${typeLabel(room)} · ${room.boundary_state === "enclosed" ? "边界有观测支持" : "部分范围"}`));
        selectable(region, () => selectRoom(room.room_id));
        svg.append(region);
      }
      for (const segment of items(room.boundary_segments)) {
        if (segment.kind !== "wall" || !hasXY(segment.a) || !hasXY(segment.b)) continue;
        const a = project(segment.a), b = project(segment.b);
        svg.append(shape("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y,
          class: "room-structural-wall", stroke: color }));
      }
    }
    if (showSemantics && semanticMap.image?.startsWith("data:image/png;base64,") && bounds?.length === 4) {
      const topLeft = project({x: bounds[0], y: bounds[3]});
      const layer = shape("image", {href: semanticMap.image, x: topLeft.x, y: topLeft.y,
        width: (bounds[2] - bounds[0]) * scale, height: (bounds[3] - bounds[1]) * scale,
        preserveAspectRatio: "none", class: "room-semantic-raster", "pointer-events": "none"});
      svg.append(layer);
    }
    if (trajectory.length) svg.append(shape("polyline", { points: trajectory.map(position).join(" "), class: "room-trajectory" }));
    for (const portal of portals) {
      const from = anchors.get(portal.room_a), to = anchors.get(portal.room_b);
      const stateClass = explored(portal) ? "is-explored" : "is-unexplored";
      if (from && to) svg.append(shape("polyline", {
        points: [from, portal, to].map(position).join(" "), class: `room-portal-edge ${stateClass}`,
      }));
      const p = project(portal);
      const angle = Number.isFinite(portal.normal_yaw) ? portal.normal_yaw : 0;
      const halfWidth = Number.isFinite(portal.width) ? Math.max(4, portal.width * scale / 2) : 7;
      const dx = -Math.sin(angle) * halfWidth, dy = -Math.cos(angle) * halfWidth;
      const selected = portal.portal_id === selectedPortalId;
      const node = shape("g", {
        id: `room-map-portal-${encodeURIComponent(portal.portal_id)}`, "data-portal-id": portal.portal_id,
        class: `room-portal ${stateClass}${selected ? " is-selected" : ""}`,
        role: "button", tabindex: 0, "aria-pressed": selected,
        "aria-label": `${portal.portal_id} · ${portalState(portal)} · 穿越 ${portal.traversal_count ?? 0} 次`,
      });
      const endpoints = { x1: p.x - dx, y1: p.y - dy, x2: p.x + dx, y2: p.y + dy };
      node.append(shape("line", { ...endpoints, class: "room-portal-hit" }));
      node.append(shape("line", { ...endpoints, class: "room-portal-marker" }));
      node.append(shape("circle", { cx: p.x, cy: p.y, r: 4, class: "room-portal-center" }));
      node.append(shape("text", { x: p.x + 9, y: p.y - 10, class: "room-portal-label" }, `${portal.portal_id} · ${portalState(portal)}`));
      node.append(shape("title", {}, `${portal.portal_id} · ${portalState(portal)} · 穿越 ${portal.traversal_count ?? 0} 次`));
      selectable(node, () => {
        const linked = [portal.room_b, portal.room_a].map((id) => rooms.find((room) => room.room_id === id)).filter(Boolean);
        const target = linked.find(prospective) || linked[0];
        if (target) selectRoom(target.room_id, portal.portal_id);
      });
      svg.append(node);
    }
    if (hasXY(snapshot.candidate)) {
      const p = project(snapshot.candidate);
      const candidate = shape("circle", { cx: p.x, cy: p.y, r: 9, class: "room-portal-candidate" });
      candidate.append(shape("title", {}, "待确认的门口候选"));
      svg.append(candidate);
    }
    for (const room of rooms) {
      if (!anchors.has(room.room_id)) continue;
      const p = project(anchors.get(room.room_id));
      const selected = room.room_id === selectedId;
      const current = room.room_id === snapshot.current_room_id;
      const node = shape("g", {
        id: `room-map-node-${encodeURIComponent(room.room_id)}`, "data-room-id": room.room_id,
        class: `room-node ${prospective(room) ? "is-prospective" : room.visited ? "is-entered" : "is-unvisited"}${current ? " is-current" : ""}${selected ? " is-selected" : ""}`,
        role: "button", tabindex: 0, "aria-pressed": selected,
        "aria-label": `${room.room_id} · ${typeLabel(room)}${prospective(room) ? " · 候选空间，未进入" : ""}${current ? " · 当前房间" : ""}`,
      });
      node.append(shape("circle", { cx: p.x, cy: p.y, r: 12 }));
      // Keep labels inside the viewport; untrusted semantic strings remain plain text.
      const alignRight = p.x > width * .65;
      const labelX = p.x + (alignRight ? -18 : 18);
      const labelAttrs = { x: labelX, "text-anchor": alignRight ? "end" : "start" };
      let labelY = p.y - 3;
      const nearbyPortal = portals.find((portal) => {
        if (portal.room_a !== room.room_id && portal.room_b !== room.room_id) return false;
        const q = project(portal);
        return Math.hypot(q.x - p.x, q.y - p.y) < 90;
      });
      if (nearbyPortal) {
        // Keep nearby room text clear of the measured doorway segment and label.
        const q = project(nearbyPortal);
        const extent = Math.abs(Math.cos(nearbyPortal.normal_yaw || 0)) * (nearbyPortal.width || 0) * scale / 2;
        labelY = Math.min(height - 32, Math.max(p.y + 30, q.y + extent + 20));
        labelAttrs.x = Math.min(width - 75, Math.max(75, p.x));
        labelAttrs["text-anchor"] = "middle";
      }
      node.append(shape("text", { ...labelAttrs, y: labelY }, String(room.room_id)));
      const label = typeLabel(room);
      const typeNode = shape("text", { ...labelAttrs, y: labelY + 16, class: "room-node-type" }, label);
      typeNode.append(shape("title", {}, label));
      node.append(typeNode);
      selectable(node, () => selectRoom(room.room_id));
      svg.append(node);
    }
    // A nearby candidate's label can overlap the doorway at the plot boundary.
    // Keep the doorway's visible marker and hit area above those labels.
    for (const node of [...svg.querySelectorAll(".room-portal")]) svg.append(node);
    if (hasXY(snapshot.start_pose) || trajectory.length) {
      const p = project(hasXY(snapshot.start_pose) ? snapshot.start_pose : trajectory[0]);
      svg.append(shape("path", { d: `M${p.x},${p.y - 6} l6,6 -6,6 -6,-6 Z`, class: "room-start" }));
      svg.append(shape("text", { x: p.x + 9, y: p.y + 27, class: "room-start-label" }, "起点"));
    }
    if (hasXY(snapshot.current_pose)) {
      const p = project(snapshot.current_pose);
      const yaw = Number.isFinite(snapshot.current_pose.yaw) ? snapshot.current_pose.yaw : 0;
      svg.append(shape("path", {
        d: "M14,0 L-9,-8 L-5,0 L-9,8 Z", class: "room-current-pose",
        transform: `translate(${p.x},${p.y}) rotate(${-yaw * 180 / Math.PI})`,
      }));
    }
    if (focusedId) element(focusedId)?.focus({ preventScroll: true });
  }

  function render(next) {
    if (!next || typeof next.enabled !== "boolean") throw new Error("房间记忆响应格式无效");
    snapshot = next;
    renderLive(next);
    panel.hidden = !next.enabled;
    if (!next.enabled) { selectedId = null; selectedPortalId = null; clearFrame(); return; }
    if (next.session_id !== sessionId) {
      sessionId = next.session_id;
      selectedId = null;
      selectedPortalId = null;
      clearFrame();
    }
    const rooms = allRooms();
    if (!rooms.some((room) => room.room_id === selectedId)) {
      selectedId = rooms.some((room) => room.room_id === next.current_room_id)
        ? next.current_room_id : rooms[0]?.room_id ?? null;
    }
    if (!items(next.portals).some((portal) => portal.portal_id === selectedPortalId)) selectedPortalId = null;
    const labels = { recording: "记录中", ready: "记录中", idle: "等待观测", error: "模块异常", starting: "初始化中" };
    const dropped = next.dropped_frames ? ` · 跳过 ${next.dropped_frames} 帧` : "";
    const provider = `${providerName(next.semantic_backend)} · `;
    const enteredCount = rooms.filter((room) => !prospective(room) && room.visited).length;
    const prospectiveCount = rooms.filter(prospective).length;
    text("room-memory-status", `${labels[next.status] || next.status || "等待观测"} · 已进入空间 ${enteredCount} / 候选空间 ${prospectiveCount} / 门口 ${items(next.portals).length} · ${provider}${semantic(next.semantic_status)}${dropped}`);
    const error = element("room-memory-error");
    const structuralWarning = next.structure?.status === "floor_mismatch" ? "高度变化超出本层范围，已暂停区域更新；请使用同层场景或重置。"
      : next.structure?.status === "capacity_reached" ? "区域地图已达到配置范围，边界外观测未保存。" : "";
    const messages = [next.error ? `房间记忆：${next.error}` : "", next.doorway_error ? `门口检测：${next.doorway_error}` : "",
      next.structure?.error ? `房间范围：${next.structure.error}` : "", structuralWarning].filter(Boolean);
    error.hidden = !messages.length;
    error.textContent = messages.join("；");
    renderMap();
    renderDetail();
  }

  async function poll() {
    if (stopped) return;
    controller = new AbortController();
    const timeout = setTimeout(() => controller?.abort(), 5000);
    try {
      const response = await fetch("/api/room-memory", { cache: "no-store", signal: controller.signal });
      if (!response.ok) throw new Error(`请求失败（${response.status}）`);
      const next = await response.json();
      if (!stopped) render(next);
    } catch (error) {
      if (!stopped && snapshot?.enabled !== false) {
        panel.hidden = false;
        const message = error.name === "AbortError" ? "请求超时" : error.message;
        element("room-memory-error").hidden = false;
        text("room-memory-error", `房间记忆暂不可用：${message}。正在自动重试。`);
        if (!element("room-live").hidden) text("room-live-status", "连接中断 · 显示上次结果");
      }
    } finally {
      clearTimeout(timeout);
      controller = null;
      // Schedule after completion so slow requests never overlap.
      if (!stopped) timer = setTimeout(poll, 750);
    }
  }

  new ResizeObserver(renderMap).observe(svg);
  element("room-semantic-toggle").addEventListener("change", renderMap);
  element("room-regions-toggle").addEventListener("change", renderMap);
  window.addEventListener("pagehide", () => {
    stopped = true;
    clearTimeout(timer);
    controller?.abort();
  });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted && stopped) { stopped = false; poll(); }
  });
  poll();
})();
