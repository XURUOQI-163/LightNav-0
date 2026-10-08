"use strict";

const byId = (id) => document.getElementById(id);
const ui = {
  form: byId("navigation-form"), backend: byId("backend"), instruction: byId("instruction"),
  start: byId("start-button"), stop: byId("stop-button"), reset: byId("reset-button"),
  image: byId("camera-image"), cameraMessage: byId("camera-message"),
  connection: byId("connection"), status: byId("run-status"), detail: byId("run-detail"),
  error: byId("error-message"), time: byId("sim-time"), step: byId("step-count"), latency: byId("latency"),
};
let state = null;
let connected = false;
let initialized = false;
let backendEdited = false;
let instructionEdited = false;
let pendingAction = null;
let stopping = false;
let actionError = "";
let connectionError = "";
let cameraError = "";
let imageUrl = null;
let lastFrame = null;
const controllers = new Set();
const pointingView = new HabitatPointing(byId("pointing-overlay"), byId("pointing-info"));

async function request(path, options = {}, timeoutMs = 5000, asImage = false) {
  const controller = new AbortController();
  controllers.add(controller);
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, { ...options, cache: "no-store", signal: controller.signal });
    if (asImage) {
      if (!response.ok) throw new Error(response.status === 503 ? "等待场景画面…" : `画面请求失败（${response.status}）`);
      let pointing = null;
      try { pointing = JSON.parse(response.headers.get("X-Pointing") || "null"); } catch (_) {}
      return {blob: await response.blob(), pointing};
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
  const locked = active || loading || pendingAction !== null;
  ui.backend.disabled = locked;
  ui.instruction.disabled = locked;
  ui.start.disabled = !connected || locked || stopping || !ui.instruction.value.trim();
  // Stop remains available while a model request or a start/reset request is pending.
  ui.stop.disabled = stopping;
  ui.stop.textContent = stopping ? "正在停止…" : "停止";
  ui.reset.disabled = !connected || loading || pendingAction !== null || stopping;
  ui.start.textContent = pendingAction === "start" ? "正在启动…" : active ? "导航中" : "开始导航";
  ui.connection.textContent = connected ? "已连接" : "未连接";
  ui.connection.classList.toggle("online", connected);
  const labels = { idle: "就绪", running: "运行中", stopped: "已停止", error: "运行异常", loading: "加载中" };
  const model = state?.backend === "omnivla" ? "OmniVLA" : "LightNav";
  const stopReasons = {
    step_limit: `达到设置的 ${state?.max_steps} 步上限，未判定找到目标。`,
    model_stop: `${model} 返回停止信号；是否找到目标需结合画面判断。`,
    empty_path: `${model} 未返回可执行路径，导航停止。`,
    zero_command: "当前路径计算出的速度为零，导航停止。",
    user_stop: "已手动停止，可修改指令后继续。",
  };
  const details = {
    idle: "选择模型并输入导航指令。",
    running: `${model} 正在根据 RGB 画面执行导航。`,
    stopped: stopReasons[state?.stop_reason] || "导航已停止，可修改指令后重新开始。",
    error: "请检查错误信息，修复后重新开始或重置场景。",
    loading: "正在加载 Habitat 场景…",
  };
  ui.status.textContent = connected ? labels[status] || status : "连接中断";
  ui.detail.textContent = connected ? details[status] || "等待状态更新…" : "无法连接 Habitat 服务，正在自动重试。";
  const error = connectionError || actionError || state?.error || "";
  ui.error.hidden = !error;
  ui.error.textContent = error;
  const cameraMessage = !connected ? "服务连接中断，等待恢复…" : cameraError || (!imageUrl ? "等待场景画面…" : "");
  ui.cameraMessage.hidden = !cameraMessage;
  ui.cameraMessage.textContent = cameraMessage;
  ui.time.textContent = Number.isFinite(state?.sim_time) ? `${state.sim_time.toFixed(1)} s` : "—";
  ui.step.textContent = Number.isFinite(state?.step) ? String(state.step) : "—";
  ui.latency.textContent = Number.isFinite(state?.model_latency_ms) ? `${Math.round(state.model_latency_ms)} ms` : "—";
}

async function pollState() {
  try {
    const next = await request("/api/state");
    state = next;
    connected = true;
    connectionError = "";
    if (!initialized) {
      if (!backendEdited && ["lightnav", "omnivla"].includes(next.backend)) ui.backend.value = next.backend;
      if (!instructionEdited && next.instruction) ui.instruction.value = next.instruction;
      initialized = true;
    }
  } catch (error) {
    connected = false;
    connectionError = `Habitat 服务不可达：${error.message}`;
  } finally {
    render();
    setTimeout(pollState, 500);
  }
}

async function pollCamera() {
  try {
    const frame = state?.frame_id;
    if (connected && (frame == null || frame !== lastFrame || !imageUrl)) {
      const {blob, pointing} = await request(`/api/camera.jpg?frame_id=${encodeURIComponent(frame ?? Date.now())}`, {}, 5000, true);
      const nextUrl = URL.createObjectURL(blob);
      // Decode before atomically replacing the visible image and its pixel metadata.
      const decoded = new Image();
      decoded.src = nextUrl;
      try { await decoded.decode(); } catch (error) { URL.revokeObjectURL(nextUrl); throw error; }
      const previousUrl = imageUrl;
      imageUrl = nextUrl;
      ui.image.src = nextUrl;
      ui.image.hidden = false;
      pointingView.set(pointing, decoded.naturalWidth, decoded.naturalHeight);
      if (previousUrl) URL.revokeObjectURL(previousUrl);
      lastFrame = frame;
      cameraError = "";
    }
  } catch (error) {
    cameraError = error.message;
  } finally {
    render();
    setTimeout(pollCamera, 150);
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
    }, name === "reset" ? 30000 : 10000);
    // A successful command is not proof of execution; state polling confirms it.
    if (name === "reset") lastFrame = null;
  } catch (error) {
    actionError = `${({ start: "启动", stop: "停止", reset: "重置" })[name]}失败：${error.message}`;
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
ui.backend.addEventListener("change", () => { backendEdited = true; });
ui.instruction.addEventListener("input", () => { instructionEdited = true; render(); });
ui.image.addEventListener("error", () => {
  cameraError = "画面加载失败，正在重试…";
  lastFrame = null;
  render();
});
window.addEventListener("pagehide", () => {
  for (const controller of controllers) controller.abort();
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = null;
  lastFrame = null;
  pointingView.set(null);
  ui.image.removeAttribute("src");
  ui.image.hidden = true;
});
pollState();
pollCamera();
