"""Offline MPC diagnosis; never connects to a model or running simulator.

Run from repo root with mujoco_demo/.venv/bin/python and --recording JSON.
Uses the production solver/reference selection and TurtleBot's midpoint planar
integration. Virtual time removes CPU scheduling noise; solver actuation delay is
explicitly 30 ms. Recorded body paths are re-anchored to replay capture poses,
not claimed to reproduce the original visual closed loop exactly.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from vln_mujoco.mpc import MpcTracker, build_pose_aligned_reference, HORIZON, Q_WEIGHTS


def run(name, path, events=(), initial=(0., 0., 0.), duration=20.):
    tracker = MpcTracker()
    pose = np.array(initial, dtype=float)
    tracker.set_body_path(path, (0., 0., 0.))
    command = (0., 0.)
    history = [pose.copy()]
    samples = []
    event_index = 0
    pending = None
    dt = .005
    for tick in range(round(duration / dt)):
        t = tick * dt
        while event_index < len(events) and events[event_index]['t'] <= t + 1e-9:
            event = events[event_index]
            capture_tick = max(0, min(len(history)-1, round((t-event['delay'])/dt)))
            tracker.set_body_path(event['path'], history[capture_tick])
            event_index += 1
        if pending is not None and tick >= pending[0]:
            result = pending[1]
            if result.generation == tracker._generation:
                command = result.command
                tracker.previous_command = command
            pending = None
        if tick % 20 == 0:
            reference = build_pose_aligned_reference(tracker._trajectory, pose, horizon=HORIZON, weights=Q_WEIGHTS)
            result = tracker._solve(tracker._generation, pose.copy(), reference, tracker.previous_command)
            pending = (tick+6, result)
            samples.append([t, *pose.tolist(), *command])
        v, w = command
        heading = pose[2] + w * dt / 2
        pose[0] += v * math.cos(heading) * dt
        pose[1] += v * math.sin(heading) * dt
        pose[2] = math.atan2(math.sin(pose[2]+w*dt), math.cos(pose[2]+w*dt))
        history.append(pose.copy())
    tracker.close()
    a = np.asarray(samples)
    active = a[a[:,0] >= duration/2]
    signs = np.sign(active[np.abs(active[:,5])>.25,5])
    summary = {'case':name,'last_half_turn_reversals':int(np.sum(signs[1:] != signs[:-1])),
               'last_half_max_abs_w':float(np.max(np.abs(active[:,5]))),
               'last_half_y_span':float(np.ptp(active[:,2])), 'final_y':float(pose[1]),
               'path_updates':event_index}
    print(json.dumps(summary),flush=True)
    return {'summary':summary,'columns':['t','x','y','yaw','v','w'],'samples':samples}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recording',type=Path,required=True)
    args=parser.parse_args()
    rows=json.loads(args.recording.read_text())
    events=[];last=None
    for row in rows:
        v=row['vln']
        if v['latency_ms'] is not None and v['latency_ms'] != last and v['waypoints'] and not v['stop']:
            events.append({'t':row['elapsed'],'delay':v['latency_ms']/1000,'path':v['waypoints']})
            last=v['latency_ms']
    x=np.arange(.15,25,.15)
    line=np.column_stack((x,np.zeros_like(x),np.zeros_like(x))).tolist()
    curve=np.column_stack((x,.3*np.sin(x/2),np.arctan(.15*np.cos(x/2)))).tolist()
    results=[]
    for name,path,initial in [('straight_aligned',line,(0.,0.,0.)),('straight_offset',line,(0.,.3,0.)),('straight_heading_error',line,(0.,0.,.5)),('fixed_smooth_curve',curve,(0.,0.,0.)),('frozen_first_model_path',events[0]['path'],(0.,0.,0.))]:
        results.append(run(name,path,initial=initial))
    results.append(run('recorded_updates_measured_delay',events[0]['path'],events))
    results.append(run('recorded_updates_zero_delay',events[0]['path'],[{**e,'delay':0.} for e in events]))
    for delay in [0., .3, .65]:
        repeated=[{'t':float(t),'delay':delay,'path':line[:10]} for t in np.arange(.65,20,.65)]
        results.append(run(f'constant_body_straight_delay_{delay}',events[0]['path'],repeated))
    Path(__file__).with_name('replay_results.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
