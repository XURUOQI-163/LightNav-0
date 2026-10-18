// Lightweight SVG contract test; no GPU, browser, or npm dependencies.
const assert = require('node:assert/strict');
class Element {
  constructor(name) { this.name = name; this.attrs = {}; this.children = []; }
  setAttribute(key, value) { this.attrs[key] = value; }
  removeAttribute(key) { delete this.attrs[key]; }
  replaceChildren() { this.children = []; }
  append(...nodes) { this.children.push(...nodes); }
}
global.window = {};
global.document = {createElementNS: (_, name) => new Element(name)};
require('../../habitat_demo/habitat_web/web/pointing.js');
const svg = new Element('svg'), info = {};
const view = new window.HabitatPointing(svg, info);
const points = Array.from({length:10}, (_, i) => ({index:i+1, pixel:[100+i*10, 300-i*10]}));
const data = {frame_size:[640,480], count:10, points};
view.set(null, 640, 480, data);
assert.equal(svg.children[0].children.length, 9);
assert.equal(svg.children[1].children.filter(n => n.name === 'circle').length, 10);
assert.deepEqual(svg.children[1].children.filter(n => n.name === 'text').map(n => n.textContent),
                 ['1','2','3','4','5','6','7','8','9','10']);
assert.match(info.textContent, /10\/10/);
points[4].pixel = null;
view.set({frame_size:[640,480], apos_px:[300,200], opos_px:[400,200]}, 640,480,data);
assert.equal(svg.children[0].children.length, 7); // No line crossing an invisible point.
assert.equal(svg.children[2].attrs['data-channel'], 'apos');
assert.equal(svg.children[3].attrs['data-channel'], 'opos');
assert.match(info.textContent, /9\/10/);
view.set(null, 320, 240, data); // Stale dimensions must not overlay the wrong frame.
assert.equal(svg.children.length, 0);
assert.equal(info.hidden, true);
view.set(null);
assert.ok('hidden' in svg.attrs);
console.log('SVG waypoint overlay: PASS');
