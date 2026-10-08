/* Pixels are drawn over their own inference image, including object-fit letterboxing. */
window.HabitatPointing = class HabitatPointing {
  constructor(svg, info) { this.svg = svg; this.info = info; }
  set(data, width, height) {
    const svg = this.svg;
    svg.replaceChildren();
    // SVG has no native hidden property reflected to attributes.
    svg.setAttribute("hidden", "");
    this.info.hidden = true;
    if (!data || !Array.isArray(data.frame_size) || data.frame_size[0] !== width || data.frame_size[1] !== height) return;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    const ns = "http://www.w3.org/2000/svg";
    const element = (name, attrs) => {
      const node = document.createElementNS(ns, name);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      return node;
    };
    const meanings = { rot_left:"左转", rot_right:"右转", stop:"停止", not_visible:"不可见", none:"无预测", point:"无有效坐标" };
    const labels = [];
    for (const [channel, color] of [["apos", "#ffb454"], ["opos", "#6ee7b7"]]) {
      const pixel = data[`${channel}_px`];
      const name = channel.toUpperCase();
      if (!Array.isArray(pixel) || pixel.length !== 2 || !pixel.every(Number.isFinite)
          || pixel[0] < 0 || pixel[0] >= width || pixel[1] < 0 || pixel[1] >= height) {
        labels.push(`${name}: ${meanings[data[`${channel}_state`]] || "无预测"}`);
        continue;
      }
      const [x,y] = pixel;
      const edge = data[`${channel}_clamped`] ? " · 边界" : "";
      const label = `${name} (${Math.round(x)}, ${Math.round(y)})${edge}`;
      labels.push(label);
      const group = element("g", {"data-channel":channel});
      group.append(element("circle", {cx:x, cy:y, r:7, stroke:color, "stroke-width":2, fill:"none"}));
      group.append(element("path", {d:`M${x-11} ${y}H${x+11}M${x} ${y-11}V${y+11}`, stroke:color, "stroke-width":2}));
      const text = element("text", {x:Math.max(3, Math.min(width-180,x+12)), y:y<24?y+25:y-12,
        fill:color, "font-size":14, "font-family":"monospace", "font-weight":700,
        stroke:"#080a0e", "stroke-width":3, "paint-order":"stroke"});
      text.textContent = label;
      group.append(text); svg.append(group);
    }
    this.info.textContent = `模型观测帧 · ${labels.join("  |  ")}`;
    this.info.hidden = false;
    svg.removeAttribute("hidden");
  }
};
