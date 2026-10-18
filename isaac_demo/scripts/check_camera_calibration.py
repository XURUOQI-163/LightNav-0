"""Small GPU integration check: real RTX depth -> exported K -> local floor map.

Run from isaac_demo with its Isaac environment and PYTHONPATH=.:../habitat_demo.
No GRScenes assets, model service, or running web instance are used.
"""
import argparse
import json
import math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp(dict(headless=True, width=320, height=240,
                             active_gpu=args.gpu, physics_gpu=args.gpu,
                             multi_gpu=False, renderer='RayTracedLighting',
                             disable_viewport_updates=True))
    try:
        import carb
        import numpy as np
        import omni.replicator.core as rep
        from isaac_web.geometry import camera_extrinsic, configure_pinhole_camera
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera
        from local_navigation.depth_mapping import DepthMapper
        from local_navigation.models import NavigationConfig
        from local_navigation.rolling_grid import RollingGrid
        from pxr import Gf, UsdGeom

        world = World(stage_units_in_meters=1., backend='numpy', device='cpu')
        floor = UsdGeom.Cube.Define(world.stage, '/World/Floor')
        floor.CreateSizeAttr(1.)
        floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.05))
        floor.AddScaleOp().Set(Gf.Vec3d(50, 50, .1))
        world.reset()
        camera = Camera('/World/TestCamera', resolution=(320, 240))
        configure_pinhole_camera(camera, 90.)
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()
        config = NavigationConfig(initial_clear_radius_m=0.)
        grid, mapper = RollingGrid(config), DepthMapper(config, .663)
        reports = []
        for index, (x, yaw) in enumerate(((0., 0.), (.4, math.pi / 2))):
            camera.set_world_pose(np.array([x, 0., 1.]),
                                  np.array([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]),
                                  camera_axes='world')
            carb.settings.get_settings().set('/app/player/playSimulations', False)
            for _ in range(4):
                rep.orchestrator.step(delta_time=0., pause_timeline=False, wait_for_render=True)
            depth = camera.get_depth()
            assert depth is not None, 'depth sensor not ready'
            depth = np.array(depth, copy=True)
            k = camera.get_intrinsics_matrix()
            v, _ = np.indices(depth.shape)
            ground_z = 1 - (v - k[1, 2]) * depth / k[1, 1]
            mask = (v > 180) & np.isfinite(depth) & (depth > 0) & (depth < 8)
            error = float(np.median(np.abs(ground_z[mask])))
            assert error < .03, f'rendered depth/K floor error: {error} m'
            observation = dict(sim_time=float(index), pose=dict(x=x, y=0., yaw=yaw), depth=depth,
                               depth_camera=dict(width=320, height=240, fx=k[0, 0], fy=k[1, 1],
                                                 cx=k[0, 2], cy=k[1, 2],
                                                 camera_to_robot=camera_extrinsic((0., 0., 1.))))
            mapper.update(grid, observation)
            known = int(np.count_nonzero(grid.labels() == 1))
            assert known > 500, f'floor did not populate local map: {known} cells'
            assert not grid.hits.any(), 'flat floor incorrectly classified as obstacle'
            reports.append(dict(x=x, yaw=yaw, floor_error_m=error, measured_free_cells=known))
        assert reports[1]['measured_free_cells'] > reports[0]['measured_free_cells'], reports
        print('CAMERA_CALIBRATION_PASS ' + json.dumps(reports), flush=True)
    finally:
        app.close()


if __name__ == '__main__':
    main()
