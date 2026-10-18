/* Atomic spatial map + raw prediction + corrected path. Reads never move the robot. */
window.LocalNavigationMap = class LocalNavigationMap {
  constructor(canvas) { this.canvas = canvas; this.ctx = canvas.getContext('2d'); }
  project(point, state) {
    const scale = (this.canvas.width - 64) / (state.radius * 2);
    return [this.canvas.width / 2 + (point[0] - state.center[0]) * scale,
      this.canvas.height / 2 - (point[1] - state.center[1]) * scale];
  }
  draw(state, image) {
    const ctx = this.ctx, width = this.canvas.width, height = this.canvas.height;
    const scale = (width - 64) / (state.radius * 2), project = p => this.project(p, state);
    const line = (points, color, dashed = false) => {
      if (!points?.length) return;
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2.5;
      ctx.setLineDash(dashed ? [6,4] : []);
      points.forEach((p, i) => { const q = project(p); i ? ctx.lineTo(...q) : ctx.moveTo(...q); });
      ctx.stroke(); ctx.setLineDash([]);
    };
    ctx.clearRect(0,0,width,height); ctx.save();
    ctx.beginPath(); ctx.arc(width/2,height/2,scale*state.radius,0,Math.PI*2); ctx.clip();
    ctx.fillStyle = '#616976'; ctx.fillRect(0,0,width,height);
    if (state.origin && image) {
      const top = project([state.origin[0], state.origin[1] + state.size * state.resolution]);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(image,...top,state.size*state.resolution*scale,state.size*state.resolution*scale);
    }
    const prediction = state.prediction;
    if (prediction) {
      const raw = prediction.raw || [], path = prediction.corrected || [];
      ctx.globalAlpha = prediction.map_revision === state.map_revision ? 1 : .55;
      const capture = prediction.capture_pose || state.pose;
      raw.forEach((p,i) => {
        const status = prediction.segment_status?.[i];
        line([i ? raw[i-1] : capture, p], status === 'occupied' ? '#ff4d5e'
          : status && status !== 'clear' ? '#edbe4f' : '#238eff', true);
      });
      line(path, prediction.status === 'fallback' ? '#edbe4f' : '#00ae65');
      if (prediction.unvalidated_prefix_points) {
        line(path.slice(0, prediction.unvalidated_prefix_points + 1), '#edbe4f', true);
      }
      raw.forEach((p,i) => {
        const [x,y] = project(p), bad = prediction.point_status?.[i] === 'occupied';
        ctx.beginPath(); ctx.arc(x,y,4.5,0,Math.PI*2); ctx.fillStyle = '#238eff'; ctx.fill();
        if (bad) { ctx.beginPath(); ctx.arc(x,y,8,0,Math.PI*2); ctx.strokeStyle='#ff4d5e'; ctx.stroke(); }
        ctx.font='bold 13px monospace'; ctx.lineWidth=3; ctx.strokeStyle='#0e1013'; ctx.fillStyle='#75bdff';
        ctx.strokeText(String(i+1),x+7,y-7); ctx.fillText(String(i+1),x+7,y-7);
      });
      if (path.length) {
        const [x,y] = project(path[path.length-1]);
        ctx.beginPath(); ctx.arc(x,y,6,0,Math.PI*2);
        ctx.fillStyle=prediction.status === 'fallback' ? '#edbe4f' : '#00ae65'; ctx.fill();
      }
      ctx.globalAlpha=1;
    }
    if (state.command?.map_revision === state.map_revision) line(state.command.sweep,'#b850de');
    const [x,y,yaw] = state.pose, co=Math.cos(yaw), si=Math.sin(yaw);
    const body = state.footprint.map(([a,b]) => [x+co*a-si*b,y+si*a+co*b]);
    line([...body,body[0]], '#bd2be5');
    const nose = [x+co*.45,y+si*.45];
    line([[x,y],nose], '#bd2be5');
    line([[nose[0]-.14*Math.cos(yaw-.5),nose[1]-.14*Math.sin(yaw-.5)],nose,
      [nose[0]-.14*Math.cos(yaw+.5),nose[1]-.14*Math.sin(yaw+.5)]], '#bd2be5');
    ctx.restore(); ctx.beginPath(); ctx.arc(width/2,height/2,scale*state.radius,0,Math.PI*2);
    ctx.strokeStyle='#87919e'; ctx.lineWidth=1; ctx.stroke();
    ctx.fillStyle='#c5ccd6'; ctx.font='14px sans-serif';
    ctx.fillText(`半径 ${state.radius.toFixed(1)} m · 世界方向固定`,24,22);
  }
};

(() => {
  const panel = document.getElementById('local-navigation-panel');
  if (!panel) return;
  const view = new window.LocalNavigationMap(document.getElementById('local-map'));
  const detail = document.getElementById('local-map-detail');
  const status = document.getElementById('local-map-status');
  const modeLabel = document.getElementById('local-map-mode');
  const modeError = document.getElementById('local-map-error');
  const guardedButton = document.getElementById('local-map-guarded');
  const observeButton = document.getElementById('local-map-observe');
  const labels = {ready:'地图已更新', unchanged:'原始路径未发现已知障碍', corrected:'路径已修正',
    fallback:'沿原预测继续（未获得完整修正）',
    safe_prefix:'仅有安全前缀', blocked:'运动受阻', waiting_observation:'等待有效观测', observe_only:'手动采图'};
  let active=true, timer=null, controller=null, lastKey=null, generation=0;
  async function setMode(mode) {
    guardedButton.disabled = observeButton.disabled = true;
    modeError.textContent = '';
    try {
      const response = await fetch('/api/local-navigation-mode', {
        method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({mode})
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      lastKey = null;
    } catch (error) { modeError.textContent = error.message; }
    finally { guardedButton.disabled = observeButton.disabled = false; }
  }
  guardedButton.addEventListener('click', () => setMode('planning_only'));
  observeButton.addEventListener('click', () => setMode('observe_only'));
  async function poll() {
    const current = generation;
    controller = new AbortController();
    const timeout = setTimeout(() => controller?.abort(),5000);
    try {
      const response = await fetch('/api/local-navigation',{cache:'no-store',signal:controller.signal});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (!active || current !== generation) return;
      panel.hidden = !data.enabled;
      if (!data.enabled) return;
      const key = JSON.stringify([data.session_id,data.map_revision,data.prediction?.prediction_id,data.status,data.reason,data.mode,data.command]);
      if (key === lastKey) return;
      const image = new Image(); image.src=data.png; await image.decode();
      if (!active || current !== generation) return;
      view.draw(data,image); lastKey=key;
      const old = data.prediction && data.prediction.map_revision !== data.map_revision;
      const outside = (data.prediction?.raw || []).filter(p => Math.hypot(p[0]-data.center[0],p[1]-data.center[1])>data.radius).length;
      status.textContent = `${labels[data.status] || data.status} · ${data.reason}`;
      const turnNote = data.recovery_enabled ? '自动解困的原地旋转按仿真假设跳过碰撞检查。' : '自动旋转解困已关闭。';
      modeLabel.textContent = data.mode === 'observe_only'
        ? '手动采图中：仅允许手动控制，地图继续积累。启用轨迹纠正后可开始 VLN。'
        : data.mode === 'planning_only'
          ? '仅轨迹预判与纠正：不做执行碰撞拦截；当前位置与膨胀区重叠也不会据此停止。黄色路径段表示沿原预测继续，未验证可通行。'
        : data.start_check === 'unknown'
          ? `车身周围仍有未观测区域，前进被拦截；${turnNote}原地转向未必能覆盖车底附近盲区。`
          : data.start_check === 'occupied'
            ? `当前 footprint 或安全余量与障碍重叠，前进被拦截；${turnNote}`
            : `前进碰撞保护已启用：扫描证据优先，未知区域、窗口外和障碍禁止通行；${turnNote}`;
      guardedButton.setAttribute('aria-pressed', String(data.mode === 'planning_only'));
      observeButton.setAttribute('aria-pressed', String(data.mode === 'observe_only'));
      detail.textContent = `${data.profile} · 栅格 ${data.resolution} m · 每侧余量 ${data.margin} m · 地图 v${data.map_revision}`
        + ` · 原始 ${data.prediction?.raw?.length || 0} 点 / 修正 ${data.prediction?.corrected?.length || 0} 点`
        + ` · 窗口外 ${outside} 点 · 建图 ${data.mapping_ms ?? '—'} ms · 规划 ${data.prediction?.planning_ms ?? '—'} ms`
        + (data.prediction?.planning_policy === 'known_obstacles_only'
          ? ' · 预测可经过未知区，仅已知障碍触发纠正' : '')
        + (data.initial_clearance?.remaining_cells
          ? ` · 出生点空闲假设半径 ${data.initial_clearance.radius} m，剩余 ${data.initial_clearance.remaining_cells} 格（浅青色，非实测）` : '')
        + (data.near_blind_clearance?.remaining_cells
          ? ` · 近处下方盲区假设 ${data.near_blind_clearance.remaining_cells} 格（浅青色，非实测；上限 ${data.near_blind_clearance.max_distance} m）` : '')
        + (old ? ` · 淡色路径为上次推理结果（地图 v${data.prediction.map_revision}），不代表当前验证通过` : '');
    } catch (error) {
      if (active && current === generation && !panel.hidden) status.textContent=`地图更新失败：${error.message}`;
    } finally {
      clearTimeout(timeout);
      if (active && current === generation) timer=setTimeout(poll,panel.hidden?1000:200);
    }
  }
  window.addEventListener('pagehide',() => {active=false; generation++; clearTimeout(timer); controller?.abort();});
  window.addEventListener('pageshow',event => {if(event.persisted){active=true; generation++; lastKey=null; poll();}});
  poll();
})();
