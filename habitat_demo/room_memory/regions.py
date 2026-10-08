"""Portal-constrained, partial room footprints. No navigation or semantic logic.

Repartition coverage on every revision, so late portals do not leave stale room
ownership behind. Seeded geodesic propagation never leaves observed coverage.
Open boundaries and unobserved holes stay explicit; no convex-hull completion.
"""

import math
from collections import defaultdict, deque

import numpy as np
from PIL import Image, ImageDraw


def dilate(mask, radius):
    result = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sy, sx = max(0, -dy), max(0, -dx)
            ty, tx = max(0, dy), max(0, dx)
            nh, nw = h - abs(dy), w - abs(dx)
            if nh > 0 and nw > 0:
                result[ty : ty + nh, tx : tx + nw] |= mask[sy : sy + nh, sx : sx + nw]
    return result


def _rings(mask):
    """Trace exact cell edges, retaining holes and disconnected components."""
    padded = np.pad(mask, 1)
    edges = {}
    for side, neighbor in enumerate(
        (padded[:-2, 1:-1], padded[1:-1, 2:], padded[2:, 1:-1], padded[1:-1, :-2])
    ):
        ys, xs = np.nonzero(mask & ~neighbor)
        for y, x in zip(ys.tolist(), xs.tolist()):
            ends = (
                ((x, y), (x + 1, y)),
                ((x + 1, y), (x + 1, y + 1)),
                ((x + 1, y + 1), (x, y + 1)),
                ((x, y + 1), (x, y)),
            )
            a, b = ends[side]
            edges[(a, b)] = (x + (side == 1) - (side == 3), y + (side == 2) - (side == 0))
    outgoing = defaultdict(set)
    for a, b in edges:
        outgoing[a].add(b)
    remaining = set(edges)
    rings = []
    while remaining:
        a, b = next(iter(remaining))
        start, ring = a, []
        while True:
            ring.append((a, edges[(a, b)]))
            remaining.remove((a, b))
            outgoing[a].remove(b)
            if b == start:
                break
            direction = (b[0] - a[0], b[1] - a[1])

            def turn(p):
                dx, dy = p[0] - b[0], p[1] - b[1]
                return (
                    direction[0] * dy - direction[1] * dx,
                    direction[0] * dx + direction[1] * dy,
                )

            nxt = max(outgoing[b], key=turn)
            a, b = b, nxt
        rings.append(ring)
    return rings


def _simplify(points):
    return [
        p
        for i, p in enumerate(points)
        if (
            (p[0] - points[i - 1][0]) * (points[(i + 1) % len(points)][1] - p[1])
            != (p[1] - points[i - 1][1]) * (points[(i + 1) % len(points)][0] - p[0])
        )
    ]


def _area(points):
    return sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(points, points[1:] + points[:1])) / 2


def _inside(point, ring):
    x, y = point
    inside = False
    for a, b in zip(ring, ring[1:] + ring[:1]):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    return inside


class RoomRegionBuilder:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def empty():
        return dict(
            region_polygons=[],
            boundary_segments=[],
            region_area_m2=0.0,
            boundary_state="unobserved",
            geometry_confidence=0.0,
        )

    def build(self, structure, rooms, portals, trajectory):
        result = {rid: self.empty() for rid in rooms}
        if structure.origin is None or not structure.coverage.any():
            return result
        c = self.config
        ys, xs = np.nonzero(structure.coverage)
        # Work only on observed bounding box; the bounded master grid owns world origin.
        x0, x1 = max(0, int(xs.min()) - 2), min(c.grid_size, int(xs.max()) + 3)
        y0, y1 = max(0, int(ys.min()) - 2), min(c.grid_size, int(ys.max()) + 3)
        observed = structure.coverage[y0:y1, x0:x1].astype(bool)
        walls = dilate(structure.walls[y0:y1, x0:x1], math.ceil(c.wall_padding / c.cell_size))
        h, w = observed.shape
        cut_image = Image.new("L", (w, h))
        draw = ImageDraw.Draw(cut_image)
        for portal in portals.values():
            half = portal.width / 2 + c.portal_padding
            dx, dy = -math.sin(portal.normal_yaw) * half, math.cos(portal.normal_yaw) * half
            ends = [
                structure.cell(portal.x - dx, portal.y - dy),
                structure.cell(portal.x + dx, portal.y + dy),
            ]
            draw.line([(x - x0, y - y0) for x, y in ends], fill=1, width=3)
        cuts = np.asarray(cut_image).astype(bool)
        allowed = observed & ~walls & ~cuts
        owners = np.full((h, w), -1, dtype=np.int16)
        queue = deque()
        rids = sorted(rooms)
        radius = int(math.ceil(c.seed_radius / c.cell_size))

        def seed(pose, owner):
            if pose is None:
                return
            for portal in portals.values():
                rid = rids[owner]
                if rid not in (portal.room_a, portal.room_b):
                    continue
                dx, dy = pose.x - portal.x, pose.y - portal.y
                co, si = math.cos(portal.normal_yaw), math.sin(portal.normal_yaw)
                signed, lateral = co * dx + si * dy, -si * dx + co * dy
                expected = -1 if rid == portal.room_a else 1
                if (
                    signed * expected < 0
                    and abs(signed) < portal.width + c.portal_padding
                    and abs(lateral) < portal.width / 2 + c.portal_padding
                ):
                    return  # Crossing hysteresis can leave an old label beyond the plane.
            x, y = structure.cell(pose.x, pose.y)
            x, y = x - x0, y - y0
            if not (0 <= x < w and 0 <= y < h):
                return
            # Snap a seed only a small metric distance, never across a blocked
            # doorway/wall. Seeds inside a cut are omitted; other poses suffice.
            if walls[y, x] or cuts[y, x]:
                return
            if not allowed[y, x]:
                near = [
                    (dx * dx + dy * dy, x + dx, y + dy)
                    for dy in range(-radius, radius + 1)
                    for dx in range(-radius, radius + 1)
                    if 0 <= x + dx < w
                    and 0 <= y + dy < h
                    and allowed[y + dy, x + dx]
                    and not walls[
                        min(y, y + dy) : max(y, y + dy) + 1, min(x, x + dx) : max(x, x + dx) + 1
                    ].any()
                    and not cuts[
                        min(y, y + dy) : max(y, y + dy) + 1, min(x, x + dx) : max(x, x + dx) + 1
                    ].any()
                ]
                if not near:
                    return
                _, x, y = min(near)
            if owners[y, x] < 0:
                owners[y, x] = owner
                queue.append((x, y, owner))

        # Candidate seeds are real portal-side anchors; they can label observed
        # space without implying entry or manufacturing an unseen room polygon.
        for i, rid in enumerate(rids):
            room = rooms[rid]
            seed(getattr(room, "candidate_pose", None) or room.entry_pose, i)
        by_id = {rid: i for i, rid in enumerate(rids)}
        for point in trajectory:
            if point.room_id in by_id:
                seed(point, by_id[point.room_id])
        # Scalar NumPy indexing inside BFS dominates CPU time. Native Python
        # scalar buffers keep this identical four-neighbour propagation bounded
        # and substantially cheaper without adding scipy/OpenCV to Habitat.
        walkable = allowed.ravel().tolist()
        labels = owners.ravel().tolist()
        pending = deque(y * w + x for x, y, _ in queue)
        size = w * h
        while pending:
            at = pending.popleft()
            owner, column = labels[at], at % w
            for nxt in (at - 1 if column else -1, at + 1 if column + 1 < w else -1, at - w, at + w):
                if 0 <= nxt < size and walkable[nxt] and labels[nxt] < 0:
                    labels[nxt] = owner
                    pending.append(nxt)
        owners = np.asarray(labels, dtype=np.int16).reshape(h, w)
        for i, rid in enumerate(rids):
            mask = owners == i
            if np.count_nonzero(mask) * c.cell_size**2 < c.min_region_area:
                continue
            result[rid] = self._polygonize(mask, walls, cuts, structure, x0, y0)
        return result

    def _polygonize(self, mask, walls, cuts, structure, x0, y0):
        rings = _rings(mask)
        outers, holes, segments = [], [], []
        supported, total = 0, 0
        h, w = mask.shape

        def world(p):
            return structure.world(p[0] + x0, p[1] + y0)

        for ring in rings:
            points = [item[0] for item in ring]
            simple = _simplify(points)
            if len(simple) < 3:
                continue
            (outers if _area(simple) > 0 else holes).append(simple)
            # Merge consecutive collinear boundary edges of the same evidence type.
            for j, (a, neighbor) in enumerate(ring):
                b = ring[(j + 1) % len(ring)][0]
                nx, ny = neighbor
                kind = "unknown"
                if 0 <= nx < w and 0 <= ny < h:
                    kind = "portal" if cuts[ny, nx] else "wall" if walls[ny, nx] else "unknown"
                total += 1
                supported += kind != "unknown"
                if kind == "unknown":
                    continue  # Full ring is dashed; overlay supported portions only.
                wa, wb = world(a), world(b)
                if segments and segments[-1]["kind"] == kind and segments[-1]["b"] == wa:
                    last = segments[-1]
                    cross = (wa["x"] - last["a"]["x"]) * (wb["y"] - wa["y"]) - (
                        wa["y"] - last["a"]["y"]
                    ) * (wb["x"] - wa["x"])
                    if abs(cross) < 1e-9:
                        last["b"] = wb
                        continue
                segments.append(dict(a=wa, b=wb, kind=kind))
        polygons = []
        for outer in sorted(outers, key=lambda p: (-abs(_area(p)), min(p))):
            contained = [hole for hole in holes if _inside(hole[0], outer)]
            polygons.append(
                dict(
                    outer=[world(p) for p in outer],
                    holes=[[world(p) for p in hole] for hole in contained],
                )
            )
        # Confidence is a support fraction, NOT a calibrated probability or an
        # estimate of how much of the architectural room has been explored.
        return dict(
            region_polygons=polygons,
            boundary_segments=segments,
            region_area_m2=round(float(mask.sum()) * self.config.cell_size**2, 3),
            boundary_state="enclosed" if total and supported == total and not holes else "partial",
            geometry_confidence=round(supported / max(1, total), 3),
        )
