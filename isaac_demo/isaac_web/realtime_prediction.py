"""One in-flight inference, with control-owned path swaps at step boundaries."""
import asyncio
from copy import deepcopy


class PendingPrediction:
    def __init__(self, runtime, client, instruction):
        self.runtime, self.client, self.instruction = runtime, client, instruction
        self.task = None
        self.capture = None

    def request(self):
        if self.task is not None:
            return
        r = self.runtime
        # Use the published RGB-D observation, never display-only previews. Freeze
        # input geometry now, before any later physics steps or model await.
        self.capture = (r.jpeg, r.state['sim_time'], deepcopy(r.camera_calibration),
                        deepcopy(r.state['pose']))
        r.state['tracking']['requests'] += 1
        r.state['tracking'].update(inference_pending=True,
                                   requested_sim_time=self.capture[1])
        self.task = asyncio.create_task(self.client.predict(self.capture[0], self.instruction))

    def ready(self):
        return self.task is not None and self.task.done()

    async def take(self):
        self.request()
        prediction = await self.task
        capture = self.capture
        self.task = self.capture = None
        self.runtime.state['tracking']['inference_pending'] = False
        return prediction, capture

    async def close(self):
        # Join cancellation before closing the model connection or acknowledging
        # STOP/RESET. A completed but unused exception is retrieved as well.
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.task = self.capture = None
        self.runtime.state['tracking']['inference_pending'] = False


async def run_realtime(runtime, tracker, client, instruction):
    from .execution import execute_path, publish_prediction

    r = runtime
    pending = PendingPrediction(r, client, instruction)
    try:
        while not r.max_steps or r.state['step'] < r.max_steps:
            r.state['motion_phase'] = 'waiting_model'
            if not pending.ready():
                # The previous route has ended (or this is the first request).
                # Physics holds still while waiting; do not retain its last speed.
                tracker.previous = (0., 0.)
                r.state['tracking']['command'] = [0., 0.]
            prediction, (jpeg, stamp, camera, pose) = await pending.take()
            publish_prediction(r, prediction, jpeg, stamp, camera)
            r.state['tracking']['prediction_capture_time'] = stamp
            if prediction.stop or not prediction.waypoints:
                r.state['stop_reason'] = 'model_stop' if prediction.stop else 'empty_path'
                break
            r.state['motion_phase'] = 'tracking_path'
            if not await execute_path(r, tracker, prediction.waypoints, pose, refresh=pending):
                break
        else:
            r.state['stop_reason'] = 'step_limit'
    finally:
        await pending.close()
