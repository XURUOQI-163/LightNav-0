"""Display-only ground-plane projection of the ten predicted body-frame points."""

import numpy as np


def project_waypoints(waypoints, camera):
    """Optical camera axes are right/down/forward; body axes forward/left/up.

    Keep out-of-view points as null pixels with their original sequence numbers.
    Missing calibration disables the overlay, never the navigation task.
    """
    if not camera or not waypoints:
        return None
    try:
        width, height = int(camera["width"]), int(camera["height"])
        intrinsics = np.asarray([camera[k] for k in ("fx", "fy", "cx", "cy")], dtype=float)
        transform = np.asarray(camera["camera_to_robot"], dtype=float)
        if (width <= 0 or height <= 0 or not np.isfinite(intrinsics).all()
                or min(intrinsics[:2]) <= 0 or transform.shape != (4, 4)
                or not np.isfinite(transform).all()):
            return None
        camera_from_body = np.linalg.inv(transform)
        fx, fy, cx, cy = intrinsics
        points = []
        for index, waypoint in enumerate(waypoints[:10], 1):
            body = np.array([waypoint[0], waypoint[1], 0, 1], dtype=float)
            x, y, z, _ = camera_from_body @ body
            pixel = None
            if np.isfinite([x, y, z]).all() and z > 1e-6:
                u, v = fx * x / z + cx, fy * y / z + cy
                if 0 <= u < width and 0 <= v < height:
                    pixel = [round(float(u), 2), round(float(v), 2)]
            points.append(dict(index=index, pixel=pixel))
        return dict(frame_size=[width, height], count=len(points), points=points,
                    visible_count=sum(p["pixel"] is not None for p in points))
    except (KeyError, TypeError, ValueError, IndexError, np.linalg.LinAlgError):
        return None
