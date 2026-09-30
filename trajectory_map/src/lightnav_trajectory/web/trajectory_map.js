/* Independent, read-only planar trajectory viewer. All geometry is in metres. */
(() => {
  "use strict";
  function createProjection(points, width, height) {
    const finite = points.filter(p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite));
    let minX = 0, maxX = 0, minY = 0, maxY = 0;
    for (const [x, y] of finite) {
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const scale = Math.min(Math.max(1, width - 80) / Math.max(4, maxX - minX),
      Math.max(1, height - 72) / Math.max(4, maxY - minY));
    return {scale, cx, cy, point: (x, y) => [width/2 + (x-cx)*scale, height/2 - (y-cy)*scale]};
  }
  function validSnapshot(d) {
    const point = p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite);
    return d && d.schema_version === 1 && typeof d.session_id === "string" &&
      Number.isInteger(d.revision) && d.revision >= 0 &&
      Number.isFinite(d.stale_after_s) && d.stale_after_s > 0 &&
      (d.age_s === null || (Number.isFinite(d.age_s) && d.age_s >= 0)) &&
      Array.isArray(d.segments) && d.segments.every(s => Array.isArray(s) && s.every(point)) &&
      (d.start === null || point(d.start)) && (d.current === null ||
        [d.current.x, d.current.y, d.current.yaw].every(Number.isFinite));
  }
  class TrajectoryView {
    constructor(canvas, elements = {}) {
      this.canvas = canvas; this.elements = elements; this.data = null;
      this.online = true; this.receivedAt = 0; this.generation = 0;
      this.busy = false; this.request = null; this.pollTimer = null;
      this.resize = new ResizeObserver(() => this.draw()); this.resize.observe(canvas);
      this.ageTimer = setInterval(() => this.draw(), 250);
      this.draw();
    }
    setSnapshot(data) {
      if (!validSnapshot(data)) return false;
      if (this.data?.session_id === data.session_id && this.data.revision > data.revision) return false;
      this.data = data; this.online = true; this.receivedAt = performance.now();
      this.draw(); return true;
    }
    invalidate() {
      // Cancel pre-reset reads; their eventual completion cannot restore old history.
      this.generation += 1; this.request?.abort(); this.data = null; this.draw();
    }
    start(endpoint = "/api/trajectory") {
      if (this.pollTimer) clearInterval(this.pollTimer);
      this.endpoint = endpoint; this.poll();
      this.pollTimer = setInterval(() => this.poll(), 200);
    }
    async poll() {
      if (this.busy) return;
      this.busy = true;
      const generation = this.generation;
      const controller = new AbortController(); this.request = controller;
      const timeout = setTimeout(() => controller.abort(), 2000);
      try {
        const response = await fetch(this.endpoint, {cache: "no-store", signal: controller.signal});
        if (!response.ok) throw new Error("trajectory unavailable");
        const data = await response.json();
        if (generation === this.generation && !this.setSnapshot(data)) throw new Error("invalid trajectory");
      } catch {
        if (generation === this.generation) { this.online = false; this.draw(); }
      } finally {
        clearTimeout(timeout); this.busy = false;
        if (this.request === controller) this.request = null;
      }
    }
    destroy() {
      clearInterval(this.pollTimer); clearInterval(this.ageTimer);
      this.generation += 1; this.request?.abort(); this.resize.disconnect();
    }
    draw() {
      const rect = this.canvas.getBoundingClientRect();
      const width = Math.max(1, rect.width), height = Math.max(1, rect.height);
      const ratio = window.devicePixelRatio || 1;
      const w = Math.round(width*ratio), h = Math.round(height*ratio);
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w; this.canvas.height = h;
      }
      const ctx = this.canvas.getContext("2d"); if (!ctx) return;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#101820"; ctx.fillRect(0, 0, width, height);
      const d = this.data, current = d?.current;
      const age = d?.age_s == null ? Infinity : d.age_s + (performance.now()-this.receivedAt)/1000;
      const stale = !this.online || d?.status !== "live" || age > (d?.stale_after_s || 1);
      const text = !this.online ? "连接中断 · 保留最后轨迹" : (!current ? "等待位姿" :
        (stale ? "位姿未更新" : "实时记录"));
      if (this.elements.status) {
        this.elements.status.textContent = text + (d?.history_full ? " · 历史已满" : "");
        this.elements.status.dataset.state = stale ? "stale" : "live";
      }
      if (this.elements.source) this.elements.source.textContent =
        d?.source === "mujoco_ground_truth" ? "MuJoCo 仿真位姿" :
          (d?.source === "ros_ekf" ? "EKF 融合位姿" : (d?.source || "等待定位来源"));
      if (this.elements.metrics) this.elements.metrics.textContent = current ?
        `X ${current.x.toFixed(2)} m   Y ${current.y.toFixed(2)} m   朝向 ${(current.yaw*180/Math.PI).toFixed(1)}°` : "—";
      const segments = d?.segments || [];
      const points = segments.flat(); if (current) points.push([current.x, current.y]);
      const projection = createProjection(points, width, height);
      const {scale, cx, cy} = projection;
      const rough = 70/scale, power = 10**Math.floor(Math.log10(rough));
      const step = [1, 2, 5, 10].find(v => v*power >= rough)*power;
      const left = cx-width/(2*scale), right = cx+width/(2*scale);
      const bottom = cy-height/(2*scale), top = cy+height/(2*scale);
      ctx.font = "10px ui-monospace, monospace"; ctx.lineWidth = 1;
      const label = value => Math.abs(value) < step/100 ? "0" : Number(value.toPrecision(4)).toString();
      for (let x = Math.ceil(left/step)*step; x <= right; x += step) {
        const [px] = projection.point(x, 0);
        ctx.strokeStyle = Math.abs(x) < step/100 ? "#40515f" : "#23323e";
        ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, height); ctx.stroke();
        ctx.fillStyle = "#8297a8"; ctx.fillText(label(x), px+4, height-10);
      }
      for (let y = Math.ceil(bottom/step)*step; y <= top; y += step) {
        const [, py] = projection.point(0, y);
        ctx.strokeStyle = Math.abs(y) < step/100 ? "#40515f" : "#23323e";
        ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(width, py); ctx.stroke();
        ctx.fillStyle = "#8297a8"; ctx.fillText(label(y), 8, py-4);
      }
      ctx.fillStyle = "#a9bac7"; ctx.fillText(`+Y ↑    +X →     网格 ${label(step)} m`, 12, 20);
      ctx.strokeStyle = "#62afff"; ctx.lineWidth = 2.5; ctx.lineJoin = "round";
      for (const segment of segments) {
        ctx.beginPath();
        segment.forEach(([x,y], i) => { const p = projection.point(x,y); if (i) ctx.lineTo(...p); else ctx.moveTo(...p); });
        ctx.stroke();
      }
      if (d?.start) {
        const [x,y] = projection.point(...d.start);
        ctx.beginPath(); ctx.arc(x,y,7,0,Math.PI*2); ctx.fillStyle = "#57d6a0"; ctx.fill();
        ctx.strokeStyle = "#102e24"; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = "#9af0c7"; ctx.fillText("起点", x+11, y+20);
      }
      if (current) {
        const [x,y] = projection.point(current.x, current.y);
        ctx.save(); ctx.translate(x,y); ctx.rotate(-current.yaw);
        ctx.beginPath(); ctx.moveTo(15,0); ctx.lineTo(-9,-8); ctx.lineTo(-5,0); ctx.lineTo(-9,8); ctx.closePath();
        ctx.fillStyle = stale ? "#8b9ba9" : "#ffcb72"; ctx.fill();
        ctx.strokeStyle = "#101820"; ctx.lineWidth = 2; ctx.stroke(); ctx.restore();
      }
      if (!current) {
        ctx.fillStyle = "#a9bac7"; ctx.font = "14px sans-serif"; ctx.textAlign = "center";
        ctx.fillText("收到位姿后开始绘制路线", width/2, height/2); ctx.textAlign = "start";
      }
    }
  }
  window.LightNavTrajectory = {TrajectoryView, createProjection};
})();
