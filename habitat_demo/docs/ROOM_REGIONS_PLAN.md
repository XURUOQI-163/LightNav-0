# Incremental room regions implementation plan

> Execution: writing-plans + executing-plans; approved design in the 2026-10-07 conversation. Implement inline; review independently after integration.

**Goal:** Display partial room extents from observed Habitat RGB-D and GT pose, keeping existing navigation, portal identity, semantics, sessions and export.

**Architecture:** A bounded structural grid retains horizontal observation coverage and repeated tall-surface wall evidence, not object occupancy. A separate region builder uses walls, finite portal cuts and room seeds to assign observed cells and produce polygons with uncertain boundaries. Geometry never creates rooms or issues actions.

**Stack:** Existing Python 3.9, NumPy, Pillow, aiohttp, SVG; no new SLAM, sensor, model or dependency.

## Constraints and decisions

- Preserve existing LightNav/OmniVLA commands, controller and detector thresholds.
- Visible portal creates prospective room before entry; crossing promotes same ID.
- No global cloud/object instances, no scene mesh/semantic/navmesh inputs to algorithm.
- Unknown/invalid depth never clears space. Coverage is room membership evidence, not a navigation free-space map.
- Single-level first: explicitly suspend structural updates outside the starting elevation tolerance. Multi-floor SLAM remains future work.
- Add `regions.enabled` independently of `--room-memory`; default enabled for CLI config, direct legacy constructors remain opt-in.
- Bound raster size and update rate. Show capacity/floor/geometry errors instead of silently claiming complete maps.
- Keep raw RGB-D ephemeral. Existing representative JPEGs remain.
- Current habitat_demo is wholly untracked user work. Work on a new branch in place, preserve a pre-edit source backup, do not commit pre-existing files or discard changes. This overrides a fresh worktree missing the active implementation/assets.

## Review focus

1. Delayed portal evidence must repartition already observed space and not reuse stale room ownership.
2. Unknown depth, occlusion, grid edges and partial views must not produce invented closed rooms.
3. Candidate promotion, reset and disabled regions must not break existing graph consumers.
4. Single-floor guard and invalid calibration must be visible, with no corrupted geometry.
5. Long trajectories/maps and slow perception must remain bounded and passive.

## Tasks

- [x] 1. Structural evidence (`room_memory/structure.py`, `tests/test_room_regions.py`). Add validated configuration, finite-grid projection, temporal wall support and height guard. Test ray geometry, invalid/occluded depth, wall vs low furniture, bounds and yaw; watch tests fail then pass.
- [x] 2. Region builder (`room_memory/regions.py`, same tests). Seeded propagation within observed coverage; wall and finite portal barriers; contour extraction retaining large unknown holes; partial boundary classification. Test two rooms separated by a portal, late portal repartition, candidate promotion stable ID, no unseen extension.
- [x] 3. Integrate (`config.py`, `models.py`, `graph.py`, `service.py`, `__main__.py`, `default.json`). `StructuralMapBuilder.observe(frame)` and `RoomRegionBuilder.build(structure, rooms, portals, trajectory)` are independent of model/control. Export region fields and diagnostics; validate/reset/disable and persist. Run full regression suite.
- [x] 4. Page (`room_map.js`, HTML/CSS, browser test). Render colored observed regions below trajectory/portals; dashed partial boundaries and solid supported wall edges; detail area/state; keep old snapshots compatible. Verify selection, viewport and reset in Chromium.
- [x] 5. Real Habitat replay and documentation. Validate actual RGB-D from 00770 plus known portal crossing replay, record cost/output/limits, inspect rendered map. Update CONFIGURATION, room-memory docs and `/home/xrq/projects/命令行总结.md` with working launch/toggle/restart commands. Review final changes and report measured verification.

## Progress

- Baseline: 144 tests passed using existing mujoco_demo/.venv test environment, including Chromium. Habitat Python has no pytest-asyncio; ambient ROS pytest plugins fail import, so use documented isolated test command. No environment changes.
- Pre-edit backup: `/tmp/lightnav-room-regions-before-1791351829`; branch `codex/room-regions-20261007`.

- Tasks 1–5 complete: structural/region tests exercised RED→GREEN; full demo suite 166 passed including Chromium; real 00337 and 00770 replay, actual CLI startup, screenshots and command reference updated.
- Final independent review: fixed bounded-neighbourhood validation and asynchronous error clearing with failing regressions first. No deferred minor review findings.
- Performance ruling: use native scalar BFS buffers instead of per-cell NumPy accesses; same 00770 room polygons/fields before and after. Sampling median 51 ms.
- Scope ruling: retain existing portal lifecycle; 00770 return-ID miss reproduced exactly with pre-edit code. It remains a documented limitation, not masked by manual labels.
- Runtime/clients/controllers/simulator and base models unchanged against pre-edit source. No user service restarted; temporary validation server closed.
- No commits created: preserve pre-existing untracked habitat_demo work and user document edits. Backup remains available for comparison.
