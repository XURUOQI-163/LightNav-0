"use strict";

const byId = (id) => document.getElementById(id);
const ui = {
  form: byId("navigation-form"), backend: byId("backend"), instruction: byId("instruction"),
  start: byId("start-button"), stop: byId("stop-button"), reset: byId("reset-button"),
  image: byId("camera-image"), cameraMessage: byId("camera-message"),
  connection: byId("connection"), status: byId("run-status"), detail: byId("run-detail"),
  error: byId("error-message"), time: byId("sim-time"), step: byId("step-count"), latency: byId("latency"),
  viewer: byId("viewer-panel"), title: byId("viewer-title"), manual: byId("manual-controls"),
};
const manualButtons = ["forward", "backward", "left", "right"].map((direction) => ({
  direction, button: byId(`manual-${direction}`),
}));
let state = null;
let connected = false;
let initialized = false;
let backendEdited = false;
let instructionEdited = false;
let pendingAction = null;
let stopping = false;
let actionError = "";
let connectionError = "";
let displayedTimes = [];
let pageActive = true;
let pageGeneration = 0;
const pollTimers = new Set();
const views = [
  {name: "first", image: byId("first-camera-image"), message: byId("first-camera-message")},
  {name: "third", image: byId("third-camera-image"), message: byId("third-camera-message")},
  {name: "prediction", image: ui.image, message: ui.cameraMessage},
].map((view) => ({...view, imageUrl: null, lastFrame: null, error: ""}));
function schedule(callback, delay) {
  if (!pageActive) return;
  const timer = setTimeout(() => { pollTimers.delete(timer); callback(); }, delay);
  pollTimers.add(timer);
}
const controllers = new Set();
const pointingView = new HabitatPointing(byId("pointing-overlay"), byId("pointing-info"));

async function request(path, options = {}, timeoutMs = 5000, asImage = false) {
  const controller = new AbortController();
  controllers.add(controller);
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, { ...options, cache: "no-store", signal: controller.signal });
    if (asImage) {
      if (response.status === 204) return null;
      if (!response.ok) throw new Error(response.status === 503 ? "等待场景画面…" : `画面请求失败（${response.status}）`);
      let pointing = null;
      let waypoints = null;
      try { pointing = JSON.parse(response.headers.get("X-Pointing") || "null"); } catch (_) {}
      try { waypoints = JSON.parse(response.headers.get("X-Waypoints") || "null"); } catch (_) {}
      return {blob: await response.blob(), pointing, waypoints, frameId: response.headers.get("X-Frame-Id")};
    }
    const payload = await response.json();
    if (!response.ok || payload.ok === false) throw new Error(payload.error || `请求失败（${response.status}）`);
    return payload;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("请求超时，请检查服务连接。");
    throw error;
  } finally {
    clearTimeout(timeout);
    controllers.delete(controller);
  }
}

function render() {
  const status = state?.status || "loading";
  const active = status === "running";
  const loading = status === "loading";
  const unready = state?.scene_ready === false;
  const locked = active || status === "manual" || loading || unready || pendingAction !== null;
  ui.backend.disabled = locked;
  ui.instruction.disabled = locked;
  ui.start.disabled = !connected || locked || stopping || !ui.instruction.value.trim();
  // Stop remains available while a model request or a start/reset request is pending.
  ui.stop.disabled = stopping || unready;
  ui.stop.textContent = stopping ? "正在停止…" : "停止";
  ui.reset.disabled = !connected || loading || unready || pendingAction !== null || stopping;
  ui.start.textContent = pendingAction === "start" ? "正在启动…" : active ? "导航中" : "开始导航";
  ui.connection.textContent = connected ? "已连接" : "未连接";
  ui.connection.classList.toggle("online", connected);
  const simulator = state?.simulator || "Habitat";
  ui.title.textContent = `${simulator} Viewer`;
  ui.viewer.setAttribute("aria-label", `${simulator} 实时画面`);
  const live = state?.live_view === true;
  ui.image.alt = `${simulator} ${live ? "模型输入与预测点" : "第一人称 RGB"} 画面`;
  byId("camera-views").classList.toggle("three-views", live);
  document.querySelector(".shell").classList.toggle("multi-view", live);
  byId("first-view").hidden = !live;
  byId("third-view").hidden = !live;
  byId("prediction-title").textContent = live ? "模型输入与预测点" : "第一人称 RGB";
  byId("live-metrics").hidden = !live;
  byId("view-note").hidden = !live;
  byId("camera-label").textContent = live ? "三视图同步展示" : "第一人称 RGB";
  const recovery = state?.recovery;
  const depthScore = value => Number.isFinite(value) ? `${value.toFixed(2)} m` : "无有效深度";
  const recovering = recovery?.active === true;
  const recoveryNote = recovering ? `受阻解困：原地向${recovery.direction > 0 ? "左" : "右"}旋转；选向深度左 ${depthScore(recovery.left_m)} / 右 ${depthScore(recovery.right_m)}，累计转角预算已用 ${(recovery.turn_used_deg || 0).toFixed(0)}°。原地旋转按仿真假设跳过碰撞检查，前进仍检查。` : "";
  const tracking = state?.tracking;
  const retryLabels = {path_unexecutable: "反复修正/解困后仍不可执行", path_execution_timeout: "当前路径执行超时", path_no_progress: "当前路径持续无进展", motion_blocked: "多次运动受阻"};
  const retryNote = tracking?.last_reprediction_reason ? ` · 最近重预测：${retryLabels[tracking.last_reprediction_reason] || tracking.last_reprediction_reason}` : "";
  const trackingNote = tracking ? `${tracking.controller === "mpc" ? "MPC" : "简单跟踪器"} · ${tracking.realtime_update ? "逐控制步更新预测" : "整段执行，可行性失败时重预测"} · 请求 ${tracking.requests} 次 · 已完成 ${tracking.completed_paths} 段 · 本地修正 ${tracking.local_replans || 0} 次 / 失败重预测 ${tracking.model_replans || 0} 次${retryNote}` : "";
  const phase = unready ? (state?.startup?.message || "场景尚未就绪。") : recovering ? recoveryNote : status === "running" ? (state.motion_phase === "moving" ? "正在执行运动并更新画面。" : state.motion_phase === "correcting_path" ? "正在使用更新后的局部地图修正剩余路径。" : state.motion_phase === "tracking_path" ? "正在跟踪当前路径。" : "等待模型预测，仿真暂不推进。") : status === "manual" ? "正在执行手动运动。" : "仿真已暂停。";
  byId("view-note").textContent = `${phase} ${trackingNote} 右侧保留模型输入图及其预测点，与实时视角的更新时间可能不同。`;
  const ms = (value) => Number.isFinite(value) ? `${Math.round(value)} ms` : "—";
  byId("request-latency").textContent = ms(state?.latency_ms);
  byId("step-latency").textContent = ms(state?.step_wall_ms);
  byId("render-latency").textContent = ms(state?.render_ms);
  displayedTimes = displayedTimes.filter((t) => performance.now() - t < 2000);
  byId("display-fps").textContent = `${(displayedTimes.length / 2).toFixed(1)} FPS`;
  document.title = `${simulator} Navigation`;
  ui.manual.hidden = state?.manual_control !== true;
  for (const {button} of manualButtons) button.disabled = !connected || locked || stopping;
  const labels = { idle: "就绪", running: "运行中", manual: "手动步进中", stopped: "已停止", error: "运行异常", loading: "加载中" };
  const model = state?.backend === "omnivla" ? "OmniVLA" : "LightNav";
  const stopReasons = {
    step_limit: `达到设置的 ${state?.max_steps} 步上限，未判定找到目标。`,
    model_stop: `${model} 返回停止信号；是否找到目标需结合画面判断。`,
    empty_path: `${model} 未返回可执行路径，导航停止。`,
    zero_command: "当前路径计算出的速度为零，导航停止。",
    user_stop: "已手动停止，可修改指令后继续。",
    manual_step: "手动步进完成，可继续移动或输入导航指令。",
    recovery_turn_limit: "已达到无平移进展时的解困旋转上限，仍未恢复可通行方向。",
    recovery_insufficient_depth: "左右区域有效深度不足，无法选择解困方向。",
    recovery_zero_speed: "旋转速度为零，无法执行解困。",
    path_incomplete_safe_prefix: "仅完成安全前缀，整段预测仍不可达；已停止，未请求下一段。",
    path_execution_timeout: "当前路径超过执行时限仍未到达；已停止，未请求下一段。",
    path_already_reached: "本次路径已在到达容差内，无需运动；已停止以避免原地重复请求。",
    model_replan_limit: "无有效位移时，连续重新预测仍未得到可执行路径，已达到重试上限；未判定到达目标。",
    navigation_session_changed: "地图或定位发生重置，旧预测坐标已失效；请检查位姿后重新开始。",
  };
  const details = {
    idle: "选择模型并输入导航指令。",
    running: recovering ? recoveryNote : `${model} 正在根据 RGB 画面执行导航。`,
    manual: "正在执行有限时长的运动指令…",
    stopped: stopReasons[state?.stop_reason] || `导航已停止，原因：${state?.stop_reason || "未提供"}；未判定到达目标。`,
    error: "请检查错误信息，修复后重新开始或重置场景。",
    loading: state?.startup?.message || "正在加载仿真场景…",
  };
  const startup = state?.startup;
  byId("startup-panel").hidden = !startup;
  if (startup) {
    byId("startup-phase").textContent = startup.message;
    const elapsed = Number.isFinite(startup.elapsed_s) ? Math.floor(startup.elapsed_s) : 0;
    const phaseElapsed = Number.isFinite(startup.phase_elapsed_s) ? Math.floor(startup.phase_elapsed_s) : 0;
    byId("startup-time").textContent = `启动耗时 ${Math.floor(elapsed / 60)} 分 ${elapsed % 60} 秒`;
    const paths = startup.mdl_paths;
    const stats = paths ? `MDL：${paths.aliases} 个已解析路径 → ${paths.canonical_files} 个实际文件，更新 ${paths.rewritten} 处引用。` : "";
    byId("startup-detail").textContent = `${stats}${unready && status !== "error" ? ` 当前阶段已等待 ${phaseElapsed} 秒；网页已连接，场景仍在加载，控制按钮将在首帧就绪后开放。` : status === "error" && unready ? " 请检查终端错误后重启网页服务。" : ""}`;
  }
  ui.status.textContent = connected ? labels[status] || status : "连接中断";
  ui.detail.textContent = connected ? details[status] || "等待状态更新…" : "无法连接仿真服务，正在自动重试。";
  const error = connectionError || actionError || state?.error || "";
  ui.error.hidden = !error;
  ui.error.textContent = error;
  for (const view of views) {
    const message = !connected ? "服务连接中断，等待恢复…" : view.error || (!view.imageUrl ? "等待场景画面…" : "");
    view.message.hidden = !message;
    view.message.textContent = message;
  }
  ui.time.textContent = Number.isFinite(state?.sim_time) ? `${state.sim_time.toFixed(1)} s` : "—";
  ui.step.textContent = Number.isFinite(state?.step) ? String(state.step) : "—";
  ui.latency.textContent = Number.isFinite(state?.model_latency_ms) ? `${Math.round(state.model_latency_ms)} ms` : "—";
}

async function pollState() {
  const generation = pageGeneration;
  try {
    const next = await request("/api/state");
    if (!pageActive || generation !== pageGeneration) return;
    state = next;
    connected = true;
    connectionError = "";
    if (!initialized) {
      if (!backendEdited && ["lightnav", "omnivla"].includes(next.backend)) ui.backend.value = next.backend;
      if (!instructionEdited && next.instruction) ui.instruction.value = next.instruction;
      initialized = true;
    }
  } catch (error) {
    if (!pageActive || generation !== pageGeneration) return;
    connected = false;
    connectionError = `仿真服务不可达：${error.message}`;
  } finally {
    if (pageActive && generation === pageGeneration) {
      render();
      schedule(pollState, state?.live_view ? 200 : 500);
    }
  }
}

async function pollCamera(view) {
  const generation = pageGeneration;
  const live = view.name !== "prediction";
  try {
    if (!pageActive || !connected || state?.scene_ready === false || (live && state?.live_view !== true)) return;
    const frame = state?.frame_id;
    if (!live && frame != null && frame === view.lastFrame && view.imageUrl) return;
    const path = live ? `/api/display.jpg?view=${view.name}&after=${encodeURIComponent(view.lastFrame ?? "")}`
      : `/api/camera.jpg?frame_id=${encodeURIComponent(frame ?? Date.now())}`;
    const result = await request(path, {}, 5000, true);
    if (!pageActive || generation !== pageGeneration) return;
    if (!result) { view.error = ""; return; }
    const {blob, pointing, waypoints, frameId} = result;
    const nextUrl = URL.createObjectURL(blob);
    const decoded = new Image();
    decoded.src = nextUrl;
    try { await decoded.decode(); } catch (error) { URL.revokeObjectURL(nextUrl); throw error; }
    if (!pageActive || generation !== pageGeneration) { URL.revokeObjectURL(nextUrl); return; }
    const previousUrl = view.imageUrl;
    view.imageUrl = nextUrl;
    view.image.src = nextUrl;
    view.image.dataset.view = view.name;
    view.image.hidden = false;
    if (!live) pointingView.set(pointing, decoded.naturalWidth, decoded.naturalHeight, waypoints);
    if (previousUrl) URL.revokeObjectURL(previousUrl);
    view.lastFrame = live ? frameId : (frameId == null ? frame : Number(frameId));
    if (view.name === "first") displayedTimes.push(performance.now());
    view.error = "";
  } catch (error) {
    if (pageActive && generation === pageGeneration) view.error = error.message;
  } finally {
    if (pageActive && generation === pageGeneration) {
      render();
      // Independent serial loops: a slow/broken channel cannot stall another.
      schedule(() => pollCamera(view), live && state?.live_view ? 40 : 150);
    }
  }
}

async function action(name, body = {}) {
  if (name === "stop") {
    if (stopping) return;
    stopping = true;
  } else {
    if (pendingAction !== null) return;
    pendingAction = name;
  }
  actionError = "";
  render();
  try {
    await request(`/api/${name}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }, ["reset", "manual"].includes(name) ? 30000 : 10000);
    // A successful command is not proof of execution; state polling confirms it.
    if (name === "reset") for (const view of views) view.lastFrame = null;
  } catch (error) {
    actionError = `${({ start: "启动", stop: "停止", reset: "重置", manual: "步进" })[name]}失败：${error.message}`;
  } finally {
    if (name === "stop") stopping = false;
    else pendingAction = null;
    render();
  }
}

ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!ui.start.disabled) action("start", { backend: ui.backend.value, instruction: ui.instruction.value.trim() });
});
ui.stop.addEventListener("click", () => action("stop"));
ui.reset.addEventListener("click", () => action("reset"));
for (const {direction, button} of manualButtons) {
  button.addEventListener("click", () => {
    if (button.disabled || state?.manual_control !== true) return;
    const linear = Number.isFinite(state?.max_linear) ? state.max_linear : 0;
    const angular = Number.isFinite(state?.max_angular) ? state.max_angular : 0;
    action("manual", {
      linear: direction === "forward" ? linear : direction === "backward" ? -linear : 0,
      angular: direction === "left" ? angular : direction === "right" ? -angular : 0,
      duration: 0.33,
    });
  });
}
ui.backend.addEventListener("change", () => { backendEdited = true; });
ui.instruction.addEventListener("input", () => { instructionEdited = true; render(); });
for (const view of views) {
  view.image.addEventListener("error", () => {
    view.error = "画面加载失败，正在重试…";
    view.lastFrame = null;
    render();
  });
}
window.addEventListener("pagehide", () => {
  pageActive = false;
  pageGeneration += 1;
  for (const timer of pollTimers) clearTimeout(timer);
  pollTimers.clear();
  for (const controller of controllers) controller.abort();
  for (const view of views) {
    if (view.imageUrl) URL.revokeObjectURL(view.imageUrl);
    view.imageUrl = null;
    view.lastFrame = null;
    view.image.removeAttribute("src");
    view.image.hidden = true;
  }
  pointingView.set(null);
});
function beginPolling() {
  pollState();
  for (const view of views) pollCamera(view);
}
window.addEventListener("pageshow", (event) => {
  if (event.persisted) {
    pageActive = true;
    displayedTimes = [];
    beginPolling();
  }
});
beginPolling();
