# Model assets and parameters

Checkpoint layout, the action decoder, `eval_config.json` and every server / CLI parameter.

## Checkpoint

A Hugging Face directory (or a directory containing `hf_ckpt/`) with `config.json`,
`model*.safetensors`, `tokenizer*`, `processor_config.json`, and ideally
`eval_config.json`. The architecture must be stock `Qwen3VLForConditionalGeneration`; the
trajectory / action tokens are ordinary rows of the embedding table.

## Action decoder

An **RVQ action-tokenizer bundle**: a directory with `manifest.json`, the per-level
codebook `.npy` files and the `jacobian_weights` file it names. The released checkpoints
ship it as `action_tokenizer/`, referenced from `eval_config.json`, so it is found without
flags; `--action_tokenizer_bundle` overrides it.

## `eval_config.json` (processing parameters)

If present next to or above the checkpoint (the search walks up four parents, so
`hf_ckpt/`, the `global_step_*` directory and the run root are all covered), it supplies the
processing parameters that must match training and cannot be guessed:

```json
{
  "version": 1,
  "common": {"video_size": [224, 320], "pool_enable": true, "pool_spatial": 2,
             "pool_mode": "avg", "pool_stage": "pre_vit"},
  "tasks": {
    "trackvla": {"num_history_frames": 64, "predict_horizon": 10, "video_fps": 4,
                 "slowfast_tiers": null,
                 "action_tokenizer": {"method": "rvq", "bundle_path": "/path/to/action_tokenizer"}},
    "vlnce":    {"num_history_frames": 64, "predict_horizon": 10, "video_fps": 4,
                 "action_tokenizer": {"method": "rvq", "bundle_path": "action_tokenizer"}}
  }
}
```

| Key | Meaning |
|---|---|
| `common.video_size` | model input frame size `[H, W]`; every frame is resized to it |
| `common.pool_*` | spatial pooling of vision tokens (`pool_stage` `pre_vit` or `post_vit`) |
| `tasks.<task>.num_history_frames` | history window (frames) fed to the model per step |
| `tasks.<task>.predict_horizon` | trajectory horizon `H` (rows per chunk) |
| `tasks.<task>.video_fps` | frame rate the checkpoint was trained at (send frames at this rate or faster; waypoint rows carry no time base of their own, see [DEPLOYMENT.md](DEPLOYMENT.md)) |
| `tasks.<task>.slowfast_tiers` | optional multi-rate history layout (SlowFast checkpoints keep the whole episode) |
| `tasks.<task>.action_tokenizer` | decoder snapshot: `{"method": "rvq", "bundle_path"}`; relative paths resolve against the config's directory |

`tasks` is keyed by task: `trackvla` for tracking (`--task tracking`), `vlnce` for
navigation (`--task vln`, and the Habitat evaluation). Values are read automatically; the CLIs'
`--num_history_frames` (and `--pool_spatial`, where offered) override them. Without the file the code
falls back to conservative defaults (`num_history_frames=16`, no pooling,
`video_size=(224, 320)`) which will not match a trained checkpoint — treat a missing
`eval_config.json` as a warning sign. Full schema: `src/lightnav/eval_config.py`.


### Shipping the action decoder with the checkpoint

`tasks.<task>.action_tokenizer.bundle_path` may be a **relative path**; it is resolved
against the directory holding `eval_config.json` (falling back to the checkpoint
directory). The released checkpoints ship their decoder this way:

```
hf_ckpt/
  config.json, model-*.safetensors, tokenizer*, processor_config.json
  eval_config.json          # bundle_path: "action_tokenizer" (both tasks)
  action_tokenizer/         # manifest.json + codebook_l*.npy + jacobian_weights.npy (+ alpha)
```

With such a layout `lightnav-serve`, `lightnav-predict`, `lightnav-eval-habitat` and
`build_tracking_agent()` need no `--action_tokenizer_bundle`: the
decoder is taken from the task entry that matches the server task (`--task tracking` →
`trackvla`, `--task vln` → `vlnce`), then from the other tasks, then from the sibling
directories `action_tokenizer/<task>` or `action_tokenizer/`. The explicit flag always
wins.

## Server / CLI parameters

`lightnav-serve` flags (environment names are what `scripts/start_servers.sh` and
`docker/entrypoint.sh` read):

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `--model_path` | `MODEL_PATH` | required | checkpoint directory |
| `--action_tokenizer_bundle` | `ACTION_TOKENIZER_BUNDLE` | from the checkpoint | RVQ bundle dir; overrides the decoder the checkpoint ships |
| `--task` | `TASK` | `tracking` | `tracking` (tracking prompt, `tasks.trackvla`) or `vln` (VLN prompt, `tasks.vlnce`) |
| `--backend` | `BACKEND` | `vllm_local` | `vllm_local` or `hf` |
| `--gpu_memory_utilization` | `GPU_MEM_UTIL` | `0.85` | fraction of GPU memory handed to the vLLM engine |
| `--max_batch_size` | `MAX_BATCH_SIZE` | `8` | max sessions per scheduler tick; also sizes vLLM `max_num_seqs`. `1` = strictly serial |
| `--max_wait_ms` | `MAX_WAIT_MS` | `8` | max wait to fill a batch; a lone request is flushed after 2 ms |
| `--quantization` | `VLLM_QUANT` | none (bf16) | `fp8_llm_only`: fp8 LLM, bf16 ViT (~1.5× on Jetson Thor). On SM 11.0 also set `VLLM_DISABLED_KERNELS`; `scripts/serve_thor.sh` does both. See [JETSON_THOR.md](JETSON_THOR.md) |
| `--vit_cache_entries` | `VLN_VIT_CACHE_ENTRIES` | auto (512 w/ SlowFast) | ViT tubelet LRU capacity (`vllm_local`; speed only, never output). Long robot sessions want `1024` |
| `--num_history_frames` | `NUM_HISTORY_FRAMES` | checkpoint config | history-window override (normally leave unset) |
| `--pool_spatial` | `POOL_SPATIAL` | checkpoint config | spatial-pooling override |
| `--aspect_mode` | `ASPECT_MODE` | `stretch` | `stretch`: resize every frame to `video_size` (training behaviour); `keep`: per-session size with the camera's aspect ratio at the same pixel budget (4:3 → 288×384 for a 256×448 checkpoint) |
| `--max_new_tokens` | `MAX_NEW_TOKENS` | `8` | lower bound for the per-step decode cap (see below) |
| `--host` / `--port` | `HOST` / `PORT` | `0.0.0.0` / `8050` | bind address |
| `--ready_file` | – | none | file touched once the port is bound (used by the launchers) |
| `--no_warmup` | – | off | skip the synthetic warm-up inference before binding the port |
| `--record_dir` | `RECORD_DIR` | off | record every connection's episodes (frames + per-step records) for `lightnav-render` (see [DEPLOYMENT.md](DEPLOYMENT.md)) |
| `--record_fps`, `--record_timeline`, `--no_record_images` | `RECORD_FPS`, `RECORD_TIMELINE`, `RECORD_IMAGES` | `10`, `realtime`, images on | recording defaults written to the manifest / frame storage |
| `--cam_hfov_deg`, `--cam_height`, `--traj_forward_offset`, `--waypoint_dt_s` | `CAM_HFOV_DEG`, `CAM_HEIGHT`, `TRAJ_FORWARD_OFFSET`, `WAYPOINT_DT_S` | `90`, `0.5`, auto, `0.1` | client-camera geometry and HUD convention used only by the rendered overlay |

### MuJoCo demo parameters

`mujoco_demo` is a separate `uv` project. Run `./run.sh` for its bundled
TurtleBot, or use `uv run --extra microduck vln-mujoco ...` when selecting the
optional MicroDuck backend. MicroDuck needs two external files, the MJCF from
`pollen-robotics/microduck_rl` and an ONNX policy from
`pollen-robotics/microduck-policies`; the download steps are in
[mujoco_demo/README.md](../mujoco_demo/README.md#microduck-optional).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8088` | web-console bind address |
| `--vln-server` | empty | default LightNav WebSocket URL shown in the console |
| `--robot` | `turtlebot` | robot backend: `turtlebot` or `microduck` |
| `--robot-model` | none | external MicroDuck MJCF; required only with `--robot microduck` |
| `--walking-policy` | none | external MicroDuck ONNX policy; required only with `--robot microduck` |

Engine-level environment variables (no flag):

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLN_KV_CACHE_GIB` | auto | vLLM KV-cache size in GiB (auto-scales with `max_num_seqs`, floor 2 GiB) |
| `VLN_VLLM_ENFORCE_EAGER` | `0` | `1` disables CUDA-graph capture (faster start, slower steps) |
| `LIGHTNAV_ATTN` | `sdpa` | attention implementation for the `hf` backend (e.g. `flash_attention_2`) |
| `VLN_EVAL_TEMPERATURE` / `TOP_P` / `TOP_K` | greedy | sampling knobs for experiments; benchmark numbers assume they are unset |

**Decode token budget.** A step emits a *grounding prefix* followed by the action token(s):
one `<act_l{d}_*>` per RVQ level. The server
probes the tokenizer for the grounding families it knows and caps generation at
`max(--max_new_tokens, prefix + action_tokens)`, which saves the decode step otherwise spent
reaching `eos` without ever truncating the action.

| checkpoint family | grounding prefix | tokens |
|---|---|---|
| tracking, legacy | `<tpos_k>` | 1 |
| tracking + grid pointing | `<opos_k>` | 1 |
| navigation + dual pointing | `<apos_A><opos_O>` | 2 |

**GPU sizing.** The 4B checkpoint needs ~10 GB for weights in bf16 plus the vLLM KV cache;
one 24 GB GPU comfortably serves one engine with `--gpu_memory_utilization 0.85`. When
several servers share a GPU, `scripts/start_servers.sh` divides the utilisation by
`SERVERS_PER_GPU`.

Note what `--gpu_memory_utilization` does *not* do here. The engine passes an explicit
`kv_cache_memory_bytes` (`VLN_KV_CACHE_GIB`, floor 2 GiB) and therefore skips vLLM's
profile run, so real usage is roughly weights + that KV cache + CUDA graphs (~15 GB for
the released 4B checkpoint) whatever the fraction says. The fraction still acts as the
ceiling vLLM checks against free memory at start-up, so lower it when the GPU is shared
with another job, and raise `VLN_KV_CACHE_GIB` — not the fraction — when you want a
bigger KV cache.

---

## Input aspect ratio (`--aspect_mode`)

The checkpoints are trained on frames resized to a fixed `common.video_size` (256×448 for
the released checkpoint, ≈16:9). By default (`stretch`) every client frame is resized to
that size regardless of its aspect ratio, exactly as in training — a 4:3 camera therefore
arrives horizontally squeezed.

`--aspect_mode keep` (env `ASPECT_MODE=keep`; also `lightnav-predict --aspect_mode`,
`lightnav-eval-habitat --aspect_mode`, `build_tracking_agent(aspect_mode=)`) instead picks,
per session from its first frame, the size with the **source aspect ratio** whose area is
closest to `video_size` and whose sides are multiples of 32 (patch × merge; and of the pre-ViT
pooling factors when the checkpoint pools before the ViT). The vision-token count per frame
stays at the training budget, so history length and sequence length are unchanged:

| camera | `stretch` | `keep` |
|---|---|---|
| 16:9 (480×270, 1920×1080) | 256×448 | 256×448 (identical) |
| 4:3 (640×480) | 256×448 (squeezed) | 288×384 |
| 1:1 | 256×448 | 352×352 |

Frames within one session must share one size (the first frame decides; `reset` starts
over). `keep` avoids geometric distortion but changes the token grid the model sees, which
the released checkpoint was not trained on — validate on your robot before relying on it.
Full native resolution (feeding the camera's own pixel count) is not supported.

## Simple Habitat web viewer

The independent `habitat_demo/` provides a headless Habitat RGB viewer and language
controls for LightNav or the minimal full OmniVLA service. Its commands, defaults,
model URLs and camera/control parameters are documented in
[habitat_demo/CONFIGURATION.md](../habitat_demo/CONFIGURATION.md), with complete
launch examples in [habitat_demo/README.md](../habitat_demo/README.md).
It does not modify the existing MuJoCo or model inference entry points.
For HM3D, pass `--scene-dataset-config` with the matching
`.scene_dataset_config.json` alongside `--scene`; see the launch examples above.
Optional passive room memory is enabled by `--room-memory`, with thresholds and
local Places365 inference options in `--room-memory-config` and per-session artifacts under
`--room-memory-output` (default `runs/room-memory`). It does not issue policy actions.
The shipped room-memory config defaults to CLIP in the omnivla CPU worker; Grounding DINO uses its separate CUDA worker when selected.
Places365 remains selectable as a persistent CPU worker; neither needs an API key
or a separate service launch. Qwen remains an explicitly selected optional backend.
Places365 now retains the raw 365-class Top-1 without a confidence cutoff. The
camera panel shows a separate live single-frame result (default at most once per
second), while room nodes retain multiview averaging. `semantic.live_enabled`
and `semantic.live_interval` control this observer; `live_semantic` in the memory
API reports timestamps, processing time, counts and errors. Legacy
`memory.semantic_min_confidence` is accepted but no longer filters labels.
Use `--room-semantic-backend clip` (default), `groundingdino` for the independent CUDA
Swin-T region detector, or `places365` for the original classifier. Detection
boxes are tied to their source RGB-D frame; neither backend changes navigation.
See [Habitat configuration](../habitat_demo/CONFIGURATION.md).

Room memory now enables a separate local YOLOE-26s worker by default. RGB doorway
proposals require synchronized depth clearance before creating blue unexplored
portals and prospective rooms; actual crossings promote them under the same IDs.
`doorway` options (worker environment, local weights/prompt cache, confidence,
queue limits and depth thresholds) are documented in
[habitat_demo/CONFIGURATION.md](../habitat_demo/CONFIGURATION.md).
Set `doorway.enabled=false` for the previous depth-only mode, or omit
`--room-memory` to disable all memory processing. No navigation model flags change.

`regions.enabled=true` adds bounded RGB-D structural evidence and partial room
polygons using Habitat GT pose. Set it to `false` to retain the original graph
without region computation. The browser checkbox only toggles region rendering.
JSON v3 adds region fields and structural diagnostics; each enabled session saves
`structure.npz` alongside the graph. See the Habitat configuration reference for
resolution, update interval, depth limits and single-floor height protection.

The Habitat viewer defaults to `--max-steps 0` (unlimited). A positive integer enables
an explicit action limit; `/api/state.stop_reason` and the page report why navigation stopped.

## Isaac Sim / Carter Web Adapter

The independent `isaac_demo/run.sh` entry uses the existing `omninavbench` environment.
It keeps Habitat's model client. Isaac LightNav supports concurrent inference and tracking;
OmniVLA keeps its existing step-locked loop. Default web port: **8091**.
Full CLI flags, JSON sensor/physics parameters, model launch environment variables and HTTP contracts:
[`isaac_demo/README.md`](../isaac_demo/README.md#参数).

```bash
cd isaac_demo
MODEL_GPU=1 ./start_model.sh lightnav  # separate terminal; or omnivla
./run.sh --gpu 0 --host 0.0.0.0 --port 8091 --room-memory --room-semantic-backend groundingdino
```

`--max-steps 0` means unlimited actions but preserves model STOP. `--scene` overrides require
an explicit `--spawn X Y Z YAW_DEG`; JSON spawn yaw is radians, CLI yaw is degrees.
`ISAAC_PYTHON` can select a different already-configured Isaac Python. `MODEL_GPU` defaults to 1
in `start_model.sh`; LightNav's `PORT` defaults to 8050. No model parameters or checkpoints are modified.

Isaac LightNav tracking/scheduling switches (OmniVLA and Habitat retain their existing loops):

| Flag | Default | Behaviour |
| --- | --- | --- |
| `--mpc` / `--no-mpc` | `--no-mpc` | Reuse `mujoco_demo/vln_mujoco/mpc.py`'s CasADi/IPOPT controller instead of the simple follower. |
| `--realtime-update` / `--no-realtime-update` | `--realtime-update` | One asynchronous request while tracking the current path; swap predictions between control steps. Disabled: request after path completion / confirmed local execution failure. |
| `--path-position-tolerance` | `0.10` m | Completion position error, positive and at most 0.5 m. |
| `--path-yaw-tolerance-deg` | `10` degrees | Completion heading error, positive and at most 45 degrees. |
| `--path-timeout` | `60` simulated seconds | Per-prediction simulated-time budget including recovery. Local batch mode requests a replacement on expiry; without local navigation it stops. |

Use `--mpc --no-realtime-update` for segment execution; add `--local-navigation` for incremental corrections and failure-triggered predictions.
The official horizon (5), timestep (0.1 s), Q/R costs and acceleration limits are retained;
`--max-linear`/`--max-angular` set the velocity bounds. MPC uses 0.1 simulated seconds
per executed command regardless of `--dt`; `--dt` still applies to simple tracking,
manual defaults and recovery. Rendering/IPC time means this is not guaranteed wall-clock 10 Hz.
CasADi is already available in this installation's `omninavbench` environment; no environment
changes are required. Imports are lazy and no MPC solver is constructed with `--no-mpc`.

Isaac LightNav uses `isaac_demo/isaac_web/execution.py`; the original shared
`Runtime._run` remains available for Habitat and OmniVLA.
`realtime_prediction.py` owns at most one pending model request. It freezes input JPEG,
timestamp, calibration and pose together; new predictions are applied between control steps.
Current safe motion continues during inference. At the end of the available path the robot
holds still until the result arrives. STOP/reset/error drains the pending request, and a
model STOP received during a solver call prevents that stale command from being executed.
`tracking.asynchronous_prediction` and `tracking.inference_pending` expose this scheduling.
The JSON `render_frames` now controls startup warmup only (default 4). Steady RGB-D
observations and previews each schedule one explicit synchronous sensor capture, retrying
only absent/empty buffers. Capture does not advance physics; copied RGB-D shares its pose.
Capture-time pose transforms the path to fixed world coordinates. Tracking advances
monotonically along the path and checks final position AND heading; ten output points
do not imply ten control steps or exact visits to every point. With local navigation,
completion refers to the reachable, corrected final goal. A safe prefix or detour to an
earlier waypoint cannot count as full completion. Local batch mode continues repairing the original prediction rather than stopping at that prefix.
Current RGB-D continues to update the local map and every command still passes the guard.
Blocked commands try local correction and the existing depth-directed recovery. In batch
mode, `incremental_execution.py` retains the original world path and periodically repairs its remaining suffix.
After repeated local failures and one recovery attempt per prediction, it requests a fresh model
path from the latest image. In realtime mode recovery can request a fresh path immediately.
`path_already_reached` stops stationary/near-zero predictions to avoid an idle request loop.
Model STOP, user STOP, solver failures and `--max-steps` take precedence.
The page and `/api/state.tracking` expose controller, update mode, request count, completed
paths and path progress. `local_replans`, `model_replans`, `failure_confirmations` and
`last_reprediction_reason` distinguish geometric repairs from failure-triggered model requests.
`/api/state.tracking_config` records launch options. Restart only
the Isaac web process after changing flags; the model service is unchanged.

Local batch execution (`--local-navigation --no-realtime-update`) uses these JSON fields
in `isaac_demo/config/local_navigation.json`:

| Field | Default | Meaning |
| --- | --- | --- |
| `path_replan_interval_s` | 0.5 | Minimum simulated time between periodic repairs on a newer map (0.1–5). Prefix completion, guard rejection and stalls force an earlier repair. |
| `path_failure_confirmations` | 3 | Failed local attempts on refreshed observations before recovery/fallback (integer 2–20). |
| `path_no_progress_s` | 8 | Simulated seconds without meaningful movement; try recovery once, then replace the prediction if still stalled (1–120). |
| `path_progress_translation_m` | 0.05 | Net translation that refreshes the progress timer (0.01–0.5); pure-turn predictions can also advance by heading error reduction. |
| `model_replan_limit` | 3 | Maximum additional failed-prediction retries without translation (integer 1–10); stop with `model_replan_limit` if exhausted. |

Retry budget resets on actual translation of `recovery_reset_translation_m` (default 0.2 m)
or completion of a full path. A failed planning attempt is not proof of global geometric
unreachability: the detector means unexecutable within current observations and bounded budgets.
Stationary confirmation steps collect fresh RGB-D while commanding zero velocity. The model
request always uses the latest frame; no overlapping requests or automatic socket reconnects
are introduced. Pose/map session discontinuities stop instead of reusing old world coordinates.
The blue map points retain the entire original prediction; `remaining_start_index` marks progress,
and a corrected `goal_index` refers to the original prediction, not the shortened suffix.

Isaac `--display-fps` overrides JSON `display_fps` (default 15, range 0–60; 0 disables
intermediate display frames). It is a simulated-time sampling target, not guaranteed browser FPS.
JSON `chase_distance` / `chase_height` / `chase_target_height` default to 1.8 / 1.3 / 0.3 metres.
The chase camera is display-only. Navigation RGB-D and Room Memory remain on the front camera.
The page displays live first person, third person and the prediction-aligned input snapshot
side by side (stacked on narrow screens), with independent polling and no view switching.
`/api/display.jpg?view=first|third&after=FRAME_ID` serves the latest display cache (204 if unchanged).
Additional state timing fields are `physics_ms`, `render_ms`, `step_wall_ms`, `motion_phase`;
physics still waits for the model between bounded actions. See the adapter README for timing definitions.


Isaac JSON `normalize_mdl_paths` is a strict boolean, default `true`.
`--no-normalize-mdl-paths` disables it for comparison with the previous loading path.
The normalizer rewrites resolved local MDL sourceAsset aliases to canonical real paths in
an anonymous USD session layer; it does not save asset files or merge shader parameters.
Remote, unresolved, packaged and time-varying sources are skipped. Synchronous asset loading stays enabled.

The Isaac CLI starts HTTP before scene initialization finishes. `/api/state.scene_ready`
is false until the initial RGB-D observation is published; start/stop/reset/manual remain
blocked during loading or failed initialization. `/api/state.startup` provides `phase`,
`message`, `elapsed_s`, `phase_elapsed_s`, and optional `mdl_paths` statistics
(`rewritten`, `aliases`, `canonical_files`, `skipped`). Elapsed time is computed by the web
process even while the worker waits for materials; no percentage or completion ETA is inferred.
Startup failures remain visible over HTTP until the operator restarts the service.
The default `--startup-timeout 1800` still limits the worker startup, independent of
model `--timeout` and subsequent scene `--command-timeout` calls.

`startup.phase=readback` identifies initial RGB/depth/third-person buffer validation after
warm-up renders. Each completed initial render updates the message; it is not a material
compilation percentage. Initialization tracebacks are sent before Kit shutdown. An unexpected
worker EOF includes its PID, available exit code and last reported phase.

Isaac JSON `camera_ready_max_renders` (integer 1–600, default 60) and
`camera_ready_timeout_s` (finite >0 and <=300 seconds, default 30) bound extra
sensor capture attempts after normal warm-up renders. `startup.phase=sensor_wait`
reports missing front RGB, depth or chase RGB. Missing buffers trigger a zero-delta
Replicator capture with physics updates guarded off; nonempty wrong-sized buffers fail.
The time budget is checked between calls, not a native-call interrupt; worker startup
and command timeouts still bound blocking calls. These are JSON settings, not CLI flags.

Once fallback enables explicit capture, that worker explicitly schedules each subsequent
observation and preview too: Replicator STEPPED state must not allow old nonempty buffers
to masquerade as fresh frames. This can add rendering wall time without changing action dt.

Isaac `--robot-profile original|half_xy` overrides JSON `robot_profile` (default `original`).
The profiles live in `isaac_demo/config/robots/carter_{original,half_xy}.json` and are
authoritative for wheel radius, wheel track, camera offset and footprint; edit these files
rather than the legacy duplicated wheel/camera fields in the scene JSON.
`half_xy` halves horizontal dimensions while retaining overall height. Wheel radius/track
become 0.12/0.27 m; front camera offset becomes [0.2,0,0.6] m. FOV, resolution and spawn
height stay unchanged. Circular wheels are resized uniformly and moved down by the lost
radius. Body geometry, mass and joint anchors are adjusted before PhysX initialization
in anonymous runtime layers; source assets are never saved. Dynamic rigid-body TRS and
initial velocities are authored in the runtime root layer so PhysX can update them;
static geometry and joint changes use the session layer. Only Carter V1 is supported
by the half_xy adapter. This is a navigation proxy, not a calibrated Limo dynamics model.
Footprints are conservative root-frame rectangles: original x=[-0.431,0.279],
y=[-0.315,0.315]; half_xy halves these bounds. Both room doorway detectors receive the
selected width. Minimum opening width becomes max(profile width + safety margin,
configured minimum opening width * horizontal scale); safety margins stay in metres.
`/api/state.robot_geometry` exposes the selected dimensions. Without `--local-navigation`,
footprint configuration alone does not add obstacle avoidance. Restart the Isaac
web process to change profiles; the model service can remain running.

Isaac `--local-navigation` opts into a separate spatial occupancy map, footprint-aware
path correction. Its default `planning_only` mode does not collision-filter velocity commands. It leaves model inference
and the existing follower unchanged. `--local-navigation-config PATH` defaults to
`isaac_demo/config/local_navigation.json`; `--local-map-radius METRES` overrides its
`window_radius_m` (default **5.0**, allowed 1–10). The window is a robot-centred circle;
there is no observation TTL. Room Memory and room detection are independent optional features.

The web panel shows raw/corrected paths and offers trajectory-correction/manual-observation modes.
`initial_clear_radius_m` is currently 1.5 m in the user-edited JSON (the Python
fallback when omitted is 2.0 m; allowed 0–3 m, clipped to the window; 0 disables).
With the current forward camera, visible floor begins roughly 1.3 m ahead of the
robot; a 1 m prior can leave an unknown gap. Use 2 m only when the birth area is
known to be clear.
On startup or explicit reset, unknown cells near the birth pose receive a one-time
free-space prior, separately tracked from observed floor. Measured obstacles always
win, including the first scan. New floor/obstacle evidence replaces prior cells;
the remaining prior stays at its original world position and is never replenished
after spatial eviction or a pose discontinuity. The page shows it in pale cyan.
Unknown space remains unknown in the map; it does not veto commands in planning-only mode.
`near_blind_clear_distance_m` (default **1.5 m**, allowed 0–3; **0 disables**) adds
a separate near-camera prior on each new calibrated observation. After depth
integration, only currently unknown floor cells in front of the robot, inside the
camera horizontal FOV but projecting below its bottom image boundary, and within
this robot-centred distance, can be initialized as free. This is an AND condition;
invalid depth inside the visible image does not qualify, nor do rear/side unknowns.
The prior stays in world coordinates until real floor/obstacle evidence replaces it
or it leaves the spatial window. Each cell is initialized at most once while retained
in that window. Occupancy and footprint inflation still take precedence. These
cells are pale cyan and counted separately in `near_blind_clearance.remaining_cells`
on `GET /api/local-navigation`. This affects only the local navigation map, not
Room Memory geometry or semantics. Restart the Isaac web process after changing it.
Observation mode forbids automatic navigation but does **not** collision-filter manual
commands. Mode changes require idle state; reset clears the map and restores planning-only mode.
`GET /api/local-navigation` is read-only. `POST /api/local-navigation-mode` takes
`{"mode":"planning_only"}` or `{"mode":"observe_only"}`. The explicit legacy `guarded` API mode is retained for comparison; the page does not enable it. Parameters, limitations and JSON
contracts are in [`local_navigation/README.md`](../isaac_demo/local_navigation/README.md).

Local path correction now uses `planning_policy=known_obstacles_only`: unknown
waypoints are retained; only intersections with known footprint-inflated occupancy
trigger a detour search. Swept segments are checked too, including known map crossings
between unknown endpoints. Outside-window prediction suffixes are retained if no
known collision is found; validation clips its sampling to the finite map.
Detour search stays inside the local window but may use unknown cells, with clearance
costs based on known obstacles only. Map labels are unchanged. The default execution
path does not check occupancy, unknown space, or window bounds before sending a
command. An occupied start retains the original departure prefix up to the first
clear model point and repairs the suffix; it does not clear occupied cells. If no
correction is available, `status=fallback` retains the original prediction and
records the reason. Yellow path segments are unvalidated, not certified clear.
`execution_collision_check=false` identifies this policy; `collided` observations
do not trigger a stop or recovery. Model STOP, user stop, invalid data/solver errors,
step limits and existing timeout/no-progress handling remain effective. This is the default local-navigation
planning behaviour, needs no new CLI flag, and takes effect after restarting the Isaac
web process. `--no-realtime-update` and realtime request scheduling are unaffected.

Only in the explicit legacy `guarded` mode, `config/local_navigation.json` controls
depth-guided recovery: `recovery_max_turn_deg=0` (disabled by default; a positive
value explicitly enables the optional behavior described below),
`recovery_angular_speed=0.35` rad/s (capped by `--max-angular`),
`recovery_forward_distance_m=0.5`, and `recovery_reset_translation_m=0.2`.
Left/right image regions use x=5–45% / 55–95%, y=`recovery_roi_top=0.25` to
`recovery_roi_bottom=0.65`. Select the greater `recovery_depth_percentile=30`
of valid depth; each region needs `recovery_min_valid_fraction=0.2` valid samples.
Invalid depth is not interpreted as open space. Each burst keeps its selected
direction, updates RGB-D/map after every small turn, and requests fresh model
waypoints once the forward swept footprint is clear. Consecutive bursts share
the turn budget until the configured translation threshold is reached.
Only recovery's zero-linear-velocity rotation bypasses collision checks, per the
user's simulation assumption. Translation and manual commands retain existing
checks. Stop/Reset and nonzero `--max-steps` still apply; missing depth or exhausted
budget ends recovery. `/api/state.recovery` exposes direction, scores and budget.
No model-service changes or extra CLI flags are required; restart the Isaac web
process and hard-refresh the browser. Habitat's default recovery hooks do nothing.

### Selectable CLIP / Grounding DINO semantic evidence (Habitat / Isaac)

Use `--room-memory --room-semantic-backend groundingdino`. The default room-memory
JSON defaults to CLIP; the flag selects the isolated `groundingdino` GPU worker. `semantic_backends.groundingdino.box_threshold=0.25`
filters candidate phrase scores, `semantic_backends.groundingdino.device=cuda:0` selects a visible GPU,
and `semantic_projection.interval=1.5` controls detection cadence. Only source-frame
bbox depth directions are projected into captured long-term geometry. No fixed
projection radius or Portal dependency. Every cell retains multi-class evidence.
The first image shows the completed detection source image with boxes and scores;
the third-person and navigation-model views are unaffected.

Set `semantic_projection.enabled=false` to disable map projection while retaining
detection. The checkbox only hides colors. `/api/room-memory/detection.jpg` serves
source-aligned annotated RGB; JSON and sparse evidence are saved in the session.
No candidate means no new evidence; scores are not calibrated correctness probabilities.
CLIP and DINO recognizers/projection workers are parallel implementations selected through one factory. CLIP defaults to full-frame classification; `left_right` remains available. CLIP skips views when Top1 < 0.35 OR Top3 spread < 0.10, preserving previous evidence. Configure models under `semantic_backends.<backend>` and projection under `semantic_projection.backends.<backend>`. Both bind results to captured RGB-D, pose, timestamp and long-term geometry. See [backend configuration](../habitat_demo/room_memory/BACKENDS.md).
See [details](../habitat_demo/room_memory/SEMANTIC_PROJECTION.md).

### Geometry-only room partition (Habitat / Isaac)

With `--room-memory`, the shipped configs enable `partition.enabled=true`.
`partition.peak_neighborhood_m=0.4`, `opening_max_width_m=1.6`, and
`opening_ratio=0.75` control geometric catchment seeds and bottleneck boundaries.
No area filter removes thin candidates. Semantics are aggregated only after
partitioning; they never determine room boundaries. Set `partition.enabled=false`
to display the original online Room/Portal graph instead. The API adds
`room_partition`; session files add `room_partition.json` and `room_partition.npz`.
See [algorithm, identity and limitations](../habitat_demo/room_memory/ROOM_PARTITION.md).

Room partition cleanup also supports `internal_hole_max_area_m2=2.0` and
`internal_hole_max_extent_m=4.0` for fully enclosed, wall-free holes.
`suppress_thin_regions=true`, `thin_max_width_m=0.45`, and
`thin_min_aspect=3.0` merge narrow elongated basins into adjacent regions while
retaining coverage. External gap cleanup keeps its smaller existing limits.
See `habitat_demo/room_memory/ROOM_PARTITION.md` for criteria and limitations.

### Isaac LightNav zero-command handling

A controller command with both `abs(v)` and `abs(w)` below `1e-6` is executed
as one stationary control tick; the executed zero becomes MPC's previous control.
It no longer causes immediate `zero_command` termination. This allows
acceleration-limited angular direction changes to cross zero.

In the regular executor, `path_failure_confirmations` (default 3) consecutive
zero ticks retire the route and request another prediction. The count survives
real-time path replacement. `model_replan_limit` (default 3) bounds additional
zero-command retries; a nonzero accepted command resets this zero-command budget.
These values come from the local-navigation config, or its defaults when mapping
is disabled. The existing batch executor with local mapping already confirms
failures across stationary ticks and retains its own no-progress/retry rules.

State `tracking.zero_command_steps`, `zero_command_replans`, and
`last_reprediction_reason=zero_command_persistent` expose this regular-executor
fallback. Real-time inference remains single-flight. Explicit model STOP, user
STOP, empty predictions, solver errors, step limits and explicit guard vetoes
retain their existing handling. Automatic turning recovery is not enabled by
this change. Restart the simulation process to apply it.

Room partition additionally uses `opening_min_width_m=0.45` to reject raster
cracks as portals. `fill_internal_wall_islands=true`,
`wall_island_max_area_m2=0.10`, and `wall_island_max_extent_m=0.60` allow bounded,
fully enclosed wall fragments to be filled in the derived room envelope only.
Raw wall evidence and navigation maps are unchanged; larger or boundary-connected
walls remain. Set `fill_internal_wall_islands=false` to retain every wall cell.

### Carter stability profiles (2026-10-20)

`isaac_demo/config/robots/carter_original.json` and `carter_half_xy.json` now
include a `stability` section. The selected robot profile owns these settings;
edit that profile file to tune them. No new CLI switch is required. Existing
`--max-linear` / `--max-angular` values are capped by the profile; lower requested
values remain unchanged. The startup line prints requested and effective limits.

| Setting | original | half_xy | Units |
| --- | --- | --- | --- |
| max_linear | 0.60 | 0.35 | m/s |
| max_angular | 1.0 | 0.8 | rad/s |
| linear_acceleration | 0.60 | 0.35 | m/s² |
| linear_deceleration | 0.80 | 0.50 | m/s² |
| angular_acceleration | 1.5 | 1.0 | rad/s² |
| max_lateral_acceleration | 0.50 | 0.25 | m/s², bounds speed × turn rate |
| wheel_max_torque | 12 | 2 | Nm per driven wheel |
| wheel_damping | 20 | 5 | authored USD angular-drive damping |
| com_lowering | 0.04 | 0.08 | m, chassis/ballast local COM offset |
| max_depenetration_velocity | 0.5 | 0.5 | m/s |
| solver_position_iterations | 64 | 64 | iterations |
| solver_velocity_iterations | 4 | 4 | iterations |

The default scene uses `physics_dt=1/120` and `settle_steps=180` (same 1.5 s
settling duration). Display FPS and model/control periods do not change.
`DriveLimiter` ramps wheel targets at every physics tick, preserves its state
across frame/request boundaries, and resets with the physical simulation.
Returned `applied_command` is the endpoint wheel-speed target, not a measured
chassis velocity; the LightNav MPC adapter uses it as its previous actuator
command. Physical body motion continues to be simulated with contacts/gravity.

Original asset drives had angular damping about 17453 and unlimited torque.
The half-XY geometry scaled mass and inertia but retained these drive settings.
The new finite drives/low COM are conservative navigation-proxy tuning, not
identified real Carter or Limo dynamics. Mass, visual geometry, wheel shape,
footprint and camera transform are preserved. All USD changes remain in anonymous
runtime layers; source assets are untouched. No roll/pitch lock, hidden upright
reset, or automatic turning recovery is added.

`/api/state` exposes `stability`, `applied_command`, `body_attitude` (roll/pitch
in degrees), and `chassis_pose`. Restart terminal B and refresh the browser;
terminal A does not need restarting. CPU/USD tests do not replace a physical
simulation check in the actual scene.

Reference: [NVIDIA articulation stability guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/dev_guide/guides/articulation_stability_guide.html).
