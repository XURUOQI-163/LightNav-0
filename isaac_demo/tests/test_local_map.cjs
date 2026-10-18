const assert = require('node:assert/strict');
const calls = [];
const ctx = new Proxy({}, {get: (_, name) => (...args) => calls.push([name, ...args]), set: () => true});
global.window = {};
global.document = {getElementById: () => null};
require('../../habitat_demo/habitat_web/web/local_map.js');
const canvas = {width:640, height:640, getContext: () => ctx};
const map = new window.LocalNavigationMap(canvas);
assert.deepEqual(map.project([1,2], {center:[1,2], radius:5}), [320,320]);
const snapshot = {center:[0,0], radius:5, resolution:.05, size:203, origin:[-5.05,-5.05],
  pose:[0,0,0], footprint:[[-.2,-.1],[.2,-.1],[.2,.1],[-.2,.1]], map_revision:1,
  prediction:{map_revision:1, raw:[[1,0,0],[2,0,0]], corrected:[[0,0,0],[1,.5,0],[2,0,0]],
    point_status:['clear','occupied'], segment_status:['clear','occupied']}};
map.draw(snapshot, {});
assert.ok(calls.some(c => c[0] === 'fillText' && c[1] === '1'));
assert.ok(calls.some(c => c[0] === 'fillText' && c[1] === '2'));
assert.ok(calls.some(c => c[0] === 'drawImage'));
console.log('Local map drawing: PASS');
