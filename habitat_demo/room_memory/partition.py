"""Passive geometric catchments on long-term coverage, independent of room labels.

Unknown coverage boundaries are provisional, not measured walls. Bounded cleanup
removes compact specks and merges thin branches while retaining their coverage. This is not a navigation free-space map.
"""
import heapq
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi

from .regions import RoomRegionBuilder


@dataclass
class PartitionConfig:
    enabled: bool = True
    peak_neighborhood_m: float = .4
    opening_min_width_m: float = .45
    opening_max_width_m: float = 1.6
    opening_ratio: float = .75
    cleanup_enabled: bool = True
    smooth_radius_m: float = .10
    fill_max_area_m2: float = .09
    fill_max_extent_m: float = .35
    speck_max_area_m2: float = .04
    speck_max_extent_m: float = .25
    internal_hole_max_area_m2: float = 2.0
    internal_hole_max_extent_m: float = 4.0
    fill_internal_wall_islands: bool = True
    wall_island_max_area_m2: float = .10
    wall_island_max_extent_m: float = .60
    suppress_thin_regions: bool = True
    thin_max_width_m: float = .45
    thin_min_aspect: float = 3.0

    def __post_init__(self):
        if any(type(getattr(self, name)) is not bool for name in
               ('enabled', 'cleanup_enabled', 'suppress_thin_regions', 'fill_internal_wall_islands')):
            raise ValueError('partition switches must be boolean')
        for key, value in asdict(self).items():
            if key in ('enabled', 'cleanup_enabled', 'suppress_thin_regions', 'fill_internal_wall_islands'):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('invalid partition parameter: ' + key)
        if not .05 <= self.peak_neighborhood_m <= 2 or not .2 <= self.opening_max_width_m <= 5 or self.opening_ratio >= 1:
            raise ValueError('partition geometry parameters outside supported range')
        if not .02 <= self.smooth_radius_m <= .5 or max(self.fill_max_extent_m, self.speck_max_extent_m) > 1 or max(self.fill_max_area_m2, self.speck_max_area_m2) > 1:
            raise ValueError('partition cleanup parameters exceed small-noise limits')
        if not .1 <= self.opening_min_width_m <= self.opening_max_width_m:
            raise ValueError('opening_min_width_m must be between .1 and opening_max_width_m')
        if self.wall_island_max_area_m2 > .5 or self.wall_island_max_extent_m > 1:
            raise ValueError('wall-island cleanup exceeds small internal-island limits')
        if self.internal_hole_max_area_m2 > 10 or self.internal_hole_max_extent_m > 6 or self.thin_max_width_m > 1 or self.thin_min_aspect < 2:
            raise ValueError('invalid internal-hole or thin-region thresholds')


class GeometricRoomPartition:
    def __init__(self, config=None):
        self.config = config if isinstance(config, PartitionConfig) else PartitionConfig(**(config or {}))
        self.owners = None
        self.geometry_revision = None
        self.next_id = 0
        self.next_portal_id = 0
        self.retired = set()
        self.visited = set()
        self.parents = {}
        self.records = []
        self.portals = []
        self.key = None
        self.state = dict(enabled=self.config.enabled, status='waiting', rooms=[], portals=[],
                          current_room_id=None, retired_room_ids=[], source='geometry_partition')

    @staticmethod
    def _contacts(labels, distance):
        """Unique neighbouring basin pairs and their widest saddle clearance."""
        pairs, coords, clearance = [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            h, w = labels.shape
            x0, x1 = max(0, -dx), w-max(0, dx)
            a, b = labels[:h-dy, x0:x1], labels[dy:, x0+dx:x1+dx]
            contact = (a >= 0) & (b >= 0) & (a != b)
            if dy and dx:
                contact &= (labels[:h-dy, x0+dx:x1+dx] < 0) & (labels[dy:, x0:x1] < 0)
            yy, xx = np.nonzero(contact)
            if len(xx):
                xx = xx + x0
                pairs.append(np.sort(np.stack((labels[yy, xx], labels[yy+dy, xx+dx]), axis=1), axis=1))
                coords.append(np.column_stack((xx + dx / 2, yy + dy / 2)))
                clearance.append(np.minimum(distance[yy, xx], distance[yy+dy, xx+dx]))
        if not pairs:
            return []
        pairs, coords, clearance = np.concatenate(pairs), np.concatenate(coords), np.concatenate(clearance)
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        return [(int(a), int(b), float(clearance[inverse == i].max()), coords[inverse == i])
                for i, (a, b) in enumerate(unique)]

    def _clean_coverage(self, structure):
        raw = structure.coverage.astype(bool) & ~structure.walls
        c, cell = self.config, structure.config.cell_size
        self.cleanup_stats = dict(filled_cells=0, removed_cells=0, filled_wall_island_cells=0)
        if not c.cleanup_enabled or not raw.any():
            return raw
        ys, xs = np.nonzero(raw)
        margin = max(2, math.ceil(c.smooth_radius_m / cell)+1)
        y0, y1 = max(0, ys.min()-margin), min(raw.shape[0], ys.max()+margin+1)
        x0, x1 = max(0, xs.min()-margin), min(raw.shape[1], xs.max()+margin+1)
        mask = raw[y0:y1, x0:x1].copy()
        walls = structure.walls[y0:y1, x0:x1].copy()
        radius = max(1, math.ceil(c.smooth_radius_m / cell))
        yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        disk = xx*xx + yy*yy <= radius*radius

        def small_components(candidate, max_area, max_extent):
            labels, _ = ndi.label(candidate, structure=np.ones((3, 3)))
            for label, box in enumerate(ndi.find_objects(labels), 1):
                if box is None:
                    continue
                part = labels[box] == label
                if part.sum()*cell*cell <= max_area + 1e-9 and max(part.shape)*cell <= max_extent + 1e-9:
                    yield box, part

        # Opening identifies protrusions, but only short/small residual pieces
        # may be removed. Long thin rays survive; connectors are protected.
        removals = mask & ~ndi.binary_opening(mask, structure=disk)
        for box, part in small_components(removals, c.speck_max_area_m2, c.speck_max_extent_m):
            local_box = tuple(slice(max(0, b.start-2), min(mask.shape[i], b.stop+2)) for i, b in enumerate(box))
            local = mask[local_box].copy()
            before = ndi.label(local, structure=np.ones((3, 3)))[1]
            offsets = tuple(slice(b.start-local_axis.start, b.stop-local_axis.start) for b, local_axis in zip(box, local_box))
            local[offsets] &= ~part
            if ndi.label(local, structure=np.ones((3, 3)))[1] <= before:
                mask[box] &= ~part
        removed = raw[y0:y1, x0:x1] & ~mask
        # Only fully enclosed small islands may be abstracted away in the
        # room envelope. Raw wall hits/coverage are never changed. Eight-way
        # background connectivity protects even diagonal links to the exterior.
        holes = ndi.binary_fill_holes(mask, structure=np.ones((3, 3))) & ~mask
        filled_wall_cells = 0
        if c.fill_internal_wall_islands:
            for box, part in small_components(holes, c.wall_island_max_area_m2, c.wall_island_max_extent_m):
                if (walls[box] & part).any():
                    filled_wall_cells += int((walls[box] & part).sum())
                    walls[box][part] = False
                    mask[box] |= part
        # Larger enclosed holes still must contain no retained wall evidence.
        for box, part in small_components(holes, c.internal_hole_max_area_m2, c.internal_hole_max_extent_m):
            if not (walls[box] & part).any():
                mask[box] |= part
        # Small open notches: do not bridge separate components or known walls.
        additions = ndi.binary_closing(mask, structure=disk) & ~mask & ~holes & ~removed
        for box, part in small_components(additions, c.fill_max_area_m2, c.fill_max_extent_m):
            if (walls[box] & part).any():
                continue
            local_box = tuple(slice(max(0, b.start-1), min(mask.shape[i], b.stop+1)) for i, b in enumerate(box))
            local = mask[local_box].copy()
            labels, _ = ndi.label(local, structure=np.ones((3, 3)))
            insert = np.zeros(local.shape, bool)
            offsets = tuple(slice(b.start-local_axis.start, b.stop-local_axis.start) for b, local_axis in zip(box, local_box))
            insert[offsets] = part
            touching = np.unique(labels[ndi.binary_dilation(insert, structure=np.ones((3, 3)))])
            if np.count_nonzero(touching) <= 1:
                mask[box] |= part
        cleaned = raw.copy()
        cleaned[y0:y1, x0:x1] = mask & ~walls
        self.cleanup_stats = dict(filled_cells=int((cleaned & ~raw).sum()), removed_cells=int((raw & ~cleaned).sum()),
                                  filled_wall_island_cells=filled_wall_cells)
        return cleaned

    def _partition(self, structure):
        observed = self._clean_coverage(structure)
        owners = np.full(observed.shape, -1, np.int32)
        yy, xx = np.nonzero(observed)
        if not len(xx):
            return owners, np.zeros(observed.shape)
        # Crop bounds and pad: the master-grid edge must also act as unknown.
        y0, y1, x0, x1 = yy.min(), yy.max()+1, xx.min(), xx.max()+1
        mask = np.pad(observed[y0:y1, x0:x1], 1)
        cell = structure.config.cell_size
        distance = ndi.distance_transform_edt(mask) * cell
        radius = max(1, math.ceil(self.config.peak_neighborhood_m / cell))
        peaks = mask & (distance == ndi.maximum_filter(distance, size=2*radius+1, mode='constant'))
        markers, count = ndi.label(peaks, structure=np.ones((3, 3)))
        # Every connected piece, even a single cell, must receive a marker.
        components, _ = ndi.label(mask, structure=np.ones((3, 3)))
        seeded = np.unique(components[peaks])
        for component in np.setdiff1d(np.unique(components), np.r_[0, seeded]):
            candidates = np.flatnonzero(components == component)
            at = candidates[np.argmax(distance.ravel()[candidates])]
            count += 1
            markers.ravel()[at] = count
        # Priority flood from widest interiors. Unlike IFT with all marker
        # heights treated equally, a thin corridor marker must not grow into
        # wider rooms before their descending-clearance fronts reach the neck.
        h, w = mask.shape
        labels = (markers - 1).ravel().tolist()
        walkable, clearance = mask.ravel().tolist(), distance.ravel().tolist()
        pending = []
        for at in np.flatnonzero(markers):
            at = int(at)
            heapq.heappush(pending, (-clearance[at], 0, labels[at], at))
        expanded = bytearray(h*w)
        while pending:
            level, length, owner, at = heapq.heappop(pending)
            if expanded[at] or (labels[at] >= 0 and labels[at] != owner):
                continue
            labels[at] = owner
            expanded[at] = 1
            column = at % w
            for nxt in (at-w, at+w, at-1 if column else -1, at+1 if column+1 < w else -1,
                        at-w-1 if column else -1, at+w-1 if column else -1,
                        at-w+1 if column+1 < w else -1, at+w+1 if column+1 < w else -1):
                diagonal = abs(nxt-at) in (w-1, w+1)
                if diagonal and 0 <= nxt < h*w and (walkable[(at//w)*w + nxt%w] or walkable[(nxt//w)*w + at%w]):
                    continue
                if 0 <= nxt < h*w and walkable[nxt] and not expanded[nxt] and (labels[nxt] < 0 or labels[nxt] == owner):
                    heapq.heappush(pending, (max(level, -clearance[nxt]), length+1, owner, nxt))
        basins = np.asarray(labels, np.int32).reshape(mask.shape)
        basins[~mask] = -1
        peak = ndi.maximum(distance, basins + 1, index=np.arange(1, count+1))
        parents = np.arange(count)

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        for a, b, saddle, _ in sorted(self._contacts(basins, distance), key=lambda item: -item[2]):
            a, b = root(a), root(b)
            if a == b:
                continue
            # A wide connection, or no meaningful relative constriction, is one space.
            if 2*saddle > self.config.opening_max_width_m or saddle + cell >= self.config.opening_ratio * max(peak[a], peak[b]):
                parents[b] = a
                peak[a] = max(peak[a], peak[b])
        lookup = np.array([root(i) for i in range(count)])
        basins[mask] = lookup[basins[mask]]
        # A short narrow connector between two wider interiors is the opening
        # itself, not a third room. Assign its cells to one side; never delete
        # coverage or terminal thin branches (which have only one neighbour).
        contacts = self._contacts(basins, distance)
        neighbours = {}
        for a, b, _, _ in contacts:
            neighbours.setdefault(a, set()).add(b)
            neighbours.setdefault(b, set()).add(a)
        for middle, linked in sorted(neighbours.items()):
            if len(linked) != 2:
                continue
            yy, xx = np.nonzero(basins == middle)
            if not len(xx):
                continue
            extent = max(np.ptp(xx)+1, np.ptp(yy)+1) * cell
            mid_peak = float(distance[basins == middle].max())
            if (extent <= self.config.opening_max_width_m and
                    all((basins == other).any() and mid_peak < self.config.opening_ratio * distance[basins == other].max()
                        for other in linked)):
                basins[basins == middle] = min(linked)
        if self.config.suppress_thin_regions:
            # Thin observed rays remain covered, but do not create their own
            # room/portal. Merge them into the adjacent wider catchment.
            ids = np.unique(basins[basins >= 0])
            thin = []
            for i in ids:
                yy, xx = np.nonzero(basins == i)
                width = 2*float(distance[basins == i].max())
                extent = max(np.ptp(xx)+1, np.ptp(yy)+1)*cell
                if width <= self.config.thin_max_width_m and extent >= self.config.thin_min_aspect*width:
                    thin.append((width, int(i)))
            for _, i in sorted(thin):
                if not (basins == i).any():
                    continue
                linked = {b if a == i else a for a, b, _, _ in self._contacts(basins, distance) if i in (a, b)}
                if linked:
                    target = max(linked, key=lambda j: (float(distance[basins == j].max()), int((basins == j).sum()), -j))
                    basins[basins == i] = target
        owners[y0:y1, x0:x1] = basins[1:-1, 1:-1]
        full_distance = np.zeros(observed.shape)
        full_distance[y0:y1, x0:x1] = distance[1:-1, 1:-1]
        return owners, full_distance

    def _track_ids(self, fresh):
        ids = np.unique(fresh[fresh >= 0])
        mapping, used = {}, set()
        lineage = {int(i): [] for i in ids}
        if self.owners is not None:
            valid = (fresh >= 0) & (self.owners >= 0)
            pairs, counts = np.unique(np.column_stack((fresh[valid], self.owners[valid])), axis=0, return_counts=True)
            for (new, old), _ in sorted(zip(pairs.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0][1], item[0][0])):
                lineage[new].append(f'R{old}')
                if new not in mapping and old not in used:
                    mapping[new] = old
                    used.add(old)
        for i in ids:
            if i not in mapping:
                mapping[i] = self.next_id
                self.next_id += 1
        result = np.full(fresh.shape, -1, np.int32)
        for new, stable in mapping.items():
            result[fresh == new] = stable
            self.parents[stable] = sorted(set(self.parents.get(stable, []) + lineage[int(new)]))
        if self.owners is not None:
            self.retired.update(set(np.unique(self.owners[self.owners >= 0])) - set(mapping.values()))
        return result

    def _opening_contacts(self, distance):
        # A room pair can have multiple disjoint openings. Do not place a
        # fictitious portal midway through the wall between two real contacts.
        for a, b, _, coords in self._contacts(self.owners, distance):
            pixels = np.floor(coords).astype(int)
            lo, hi = pixels.min(axis=0), pixels.max(axis=0)
            mask = np.zeros((hi[1]-lo[1]+1, hi[0]-lo[0]+1), bool)
            local = pixels-lo
            mask[local[:, 1], local[:, 0]] = True
            components, count = ndi.label(mask, structure=np.ones((3, 3)))
            assignments = components[local[:, 1], local[:, 0]]
            for component in range(1, count+1):
                group = coords[assignments == component]
                lower, upper = np.floor(group).astype(int), np.ceil(group).astype(int)
                saddle = float(np.minimum(distance[lower[:, 1], lower[:, 0]],
                                          distance[upper[:, 1], upper[:, 0]]).max())
                yield a, b, saddle, group

    def _geometry(self, structure):
        fresh, distance = self._partition(structure)
        self.owners = self._track_ids(fresh)
        self.geometry_revision = structure.revision
        builder = RoomRegionBuilder(structure.config)
        walls, cuts = structure.walls & (self.owners < 0), np.zeros(self.owners.shape, bool)
        previous_portals, used_portals = self.portals, set()
        self.portals = []
        self.cleanup_stats['suppressed_narrow_portals'] = 0
        for a, b, saddle, coords in self._opening_contacts(distance):
            width = max(structure.config.cell_size, 2*saddle)
            if width + 1e-9 < self.config.opening_min_width_m:
                self.cleanup_stats['suppressed_narrow_portals'] += 1
                continue  # Retain region coverage, but a raster crack is not a doorway.
            center = coords.mean(axis=0) + .5
            xy = structure.world(*center)
            matches = [(math.hypot(old['x']-xy['x'], old['y']-xy['y']), old['portal_id'])
                       for old in previous_portals if old['room_a'] == f'R{a}' and old['room_b'] == f'R{b}'
                       and old['portal_id'] not in used_portals]
            if matches and min(matches)[0] <= self.config.opening_max_width_m:
                portal_id = min(matches)[1]
            else:
                portal_id = f'P{self.next_portal_id}'
                self.next_portal_id += 1
            used_portals.add(portal_id)
            # Principal interface axis gives a cross-opening marker.
            centered = coords - coords.mean(axis=0)
            _, vectors = np.linalg.eigh(centered.T @ centered)
            tangent = vectors[:, -1]
            normal = math.atan2(tangent[1], tangent[0]) + math.pi/2
            width = max(structure.config.cell_size, 2*saddle)
            ends = [structure.world(*(center + sign * tangent * width / (2*structure.config.cell_size))) for sign in (-1, 1)]
            self.portals.append(dict(portal_id=portal_id, room_a=f'R{a}', room_b=f'R{b}',
                **xy, width=width, normal_yaw=normal, endpoints=ends, source='geometry_partition',
                status='unexplored', traversal_count=0, confidence=None, boundary_state='provisional'))
            rounded = np.round(coords).astype(int)
            cuts[rounded[:, 1], rounded[:, 0]] = True
        self.records = []
        for i in np.unique(self.owners[self.owners >= 0]):
            mask = self.owners == i
            yy, xx = np.nonzero(mask)
            centroid = structure.world(float(xx.mean())+.5, float(yy.mean())+.5)
            # Geometric centroid may lie outside a concave region; do not silently move it.
            fields = builder._polygonize(mask, walls, cuts, structure, 0, 0)
            self.records.append(dict(room_id=f'R{i}', numeric_id=int(i), centroid=centroid,
                source='geometry_partition', parent_room_ids=self.parents[int(i)],
                **fields, observation_state='partial', representative_frame_ids=[],
                connected_room_ids=sorted({p['room_b'] if p['room_a'] == f'R{i}' else p['room_a']
                    for p in self.portals if f'R{i}' in (p['room_a'], p['room_b'])})))

    def update(self, structure, evidence=None, pose=None, trajectory=(), output_dir=None):
        if not self.config.enabled or structure is None or not structure.config.enabled or structure.origin is None:
            return self.state
        key = (structure.revision, id(evidence), getattr(evidence, 'revision', -1), len(trajectory),
               None if pose is None else (pose.x, pose.y))
        if key == self.key:
            return self.state
        start = time.perf_counter()
        if structure.revision != self.geometry_revision:
            self._geometry(structure)
        for point in trajectory:
            x, y = structure.cell(point.x, point.y)
            if 0 <= y < self.owners.shape[0] and 0 <= x < self.owners.shape[1] and self.owners[y, x] >= 0:
                self.visited.add(int(self.owners[y, x]))
        current = None
        if pose is not None:
            x, y = structure.cell(pose.x, pose.y)
            if 0 <= y < self.owners.shape[0] and 0 <= x < self.owners.shape[1] and self.owners[y, x] >= 0:
                current = f'R{int(self.owners[y, x])}'
                self.visited.add(int(self.owners[y, x]))
        semantic_key = (structure.revision, id(evidence), getattr(evidence, 'revision', -1))
        refresh_semantics = semantic_key != getattr(self, '_semantic_key', None)
        semantic_records = {} if refresh_semantics else self._semantic_records
        rooms = []
        for record in self.records:
            record = dict(record)
            i = record.pop('numeric_id')
            record.update(room_type='unknown', room_type_probabilities={}, semantic_confidence=0.,
                          semantic_status='collecting', semantic_cells=0, visited=i in self.visited,
                          topological_status='entered' if i in self.visited else 'prospective')
            if not refresh_semantics:
                record.update(semantic_records.get(i, {}))
            elif evidence is not None and evidence.structure is structure:
                selected = (self.owners == i) & (evidence.weights > 0)
                if selected.any():
                    mass = evidence.weights[selected].astype(float)
                    scores = (evidence.values[selected] * mass[:, None]).sum(axis=0) / mass.sum()
                    best = int(np.argmax(scores))
                    record.update(room_type=evidence.labels[best], room_type_probabilities=dict(zip(evidence.labels, scores.tolist())),
                        semantic_confidence=float(scores[best]), semantic_status='ready', semantic_cells=int(selected.sum()))
            semantic_records[i] = {k: record[k] for k in ('room_type', 'room_type_probabilities',
                                  'semantic_confidence', 'semantic_status', 'semantic_cells')}
            rooms.append(record)
        self._semantic_key, self._semantic_records = semantic_key, semantic_records
        self.state = dict(enabled=True, status='ready', source='geometry_partition', rooms=rooms,
            portals=self.portals, current_room_id=current, retired_room_ids=[f'R{i}' for i in sorted(self.retired)],
            geometry_revision=structure.revision, semantic_revision=getattr(evidence, 'revision', None),
            cell_size=structure.config.cell_size, build_ms=round((time.perf_counter()-start)*1000, 2),
            params=asdict(self.config), cleanup=self.cleanup_stats)
        self.key = key
        save_key = (semantic_key, current, tuple(sorted(self.visited)))
        if output_dir is not None and save_key != getattr(self, '_save_key', None):
            path = output_dir / 'room_partition.json'
            path.with_suffix('.tmp').write_text(json.dumps(self.state, ensure_ascii=False, allow_nan=False))
            path.with_suffix('.tmp').replace(path)
            if getattr(self, '_saved_revision', None) != structure.revision:
                path = output_dir / 'room_partition.npz'
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream, owners=self.owners, origin=structure.origin,
                                        cell_size=structure.config.cell_size)
                path.with_suffix('.tmp').replace(path)
                self._saved_revision = structure.revision
            self._save_key = save_key
        return self.state
