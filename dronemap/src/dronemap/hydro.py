"""Surface-water analysis on a DTM: depression filling, D8 flow directions,
flow accumulation, catchments, depressions and drainage lines."""
from __future__ import annotations

import heapq
import math

import numpy as np

# 8 neighbours: (drow, dcol, distance factor)
NB = [(-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)), (0, -1, 1.0), (0, 1, 1.0),
      (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2))]

EPS = 1e-6  # metres added per cell when draining flats (keeps every cell draining)


def priority_flood(z: np.ndarray) -> np.ndarray:
    """Fill depressions so every valid cell drains to the edge of the data (Priority-Flood + epsilon).
    NaN cells are treated as outside. Returns float64 filled surface (NaN outside)."""
    h, w = z.shape
    valid = np.isfinite(z)
    filled = np.where(valid, z, np.nan).astype(np.float64)
    closed = ~valid
    heap: list = []
    # seeds: valid cells on the grid edge or next to no-data
    pad = np.pad(valid, 1, constant_values=False)
    interior = (pad[:-2, :-2] & pad[:-2, 1:-1] & pad[:-2, 2:] & pad[1:-1, :-2] & pad[1:-1, 2:]
                & pad[2:, :-2] & pad[2:, 1:-1] & pad[2:, 2:])
    seeds = np.argwhere(valid & ~interior)
    for r, c in seeds:
        heapq.heappush(heap, (filled[r, c], int(r), int(c)))
        closed[r, c] = True
    fz = filled  # local alias
    while heap:
        e, r, c = heapq.heappop(heap)
        for dr, dc, _ in NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not closed[rr, cc]:
                closed[rr, cc] = True
                v = fz[rr, cc]
                if v <= e:
                    v = e + EPS
                    fz[rr, cc] = v
                heapq.heappush(heap, (v, rr, cc))
    return filled


def d8_receivers(filled: np.ndarray, res: float) -> np.ndarray:
    """Flat index of the steepest-descent neighbour for each cell, -1 if none (outlet) or no data."""
    h, w = filled.shape
    best = np.zeros((h, w))
    rec = np.full((h, w), -1, dtype=np.int64)
    big = np.where(np.isfinite(filled), filled, np.inf)
    zpad = np.pad(big, 1, constant_values=np.inf)
    ipad = np.pad(np.arange(h * w, dtype=np.int64).reshape(h, w), 1, constant_values=-1)
    with np.errstate(invalid="ignore"):
        for dr, dc, f in NB:
            nb = zpad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
            nidx = ipad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
            drop = (big - nb) / (f * res)
            drop[~np.isfinite(drop)] = -np.inf
            better = drop > best
            best = np.where(better, drop, best)
            rec = np.where(better, nidx, rec)
    rec[~np.isfinite(filled)] = -1
    return rec


def accumulate(filled: np.ndarray, rec: np.ndarray, cell_area: float) -> tuple[np.ndarray, np.ndarray]:
    """Upstream contributing area (m²) per cell, plus the processing order (high to low)."""
    flat = filled.ravel()
    valid = np.isfinite(flat)
    order = np.argsort(-np.where(valid, flat, -np.inf), kind="stable")
    order = order[valid[order]]
    acc = np.where(valid, cell_area, 0.0)
    r = rec.ravel()
    for i in order:
        j = r[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc.reshape(filled.shape), order


def catchment(target: np.ndarray, rec: np.ndarray, order_high_to_low: np.ndarray) -> np.ndarray:
    """Cells whose flow path reaches any target cell."""
    drains = target.ravel().copy()
    r = rec.ravel()
    for i in order_high_to_low[::-1]:  # downstream cells first
        if not drains[i]:
            j = r[i]
            if j >= 0 and drains[j]:
                drains[i] = True
    return drains.reshape(target.shape)


def depressions(z: np.ndarray, filled: np.ndarray, res: float, min_depth: float = 0.05,
                min_area: float = 4.0) -> tuple[np.ndarray, list]:
    """Natural low spots where rain water ponds. Returns (depth array, list of dicts)."""
    from scipy import ndimage

    depth = np.where(np.isfinite(z), filled - z, np.nan)
    wet = np.nan_to_num(depth) > min_depth
    lab, n = ndimage.label(wet, structure=np.ones((3, 3)))
    out = []
    if n:
        idx = np.arange(1, n + 1)
        counts = ndimage.sum(np.ones_like(depth), lab, idx)
        vols = ndimage.sum(np.nan_to_num(depth), lab, idx) * res * res
        maxd = ndimage.maximum(np.nan_to_num(depth), lab, idx)
        coms = ndimage.center_of_mass(wet, lab, idx)
        for k in range(n):
            area = counts[k] * res * res
            if area < min_area:
                continue
            out.append({"label": int(idx[k]), "area_m2": float(area), "volume_m3": float(vols[k]),
                        "max_depth_m": float(maxd[k]), "row": float(coms[k][0]), "col": float(coms[k][1])})
    out.sort(key=lambda d: -d["volume_m3"])
    depth[~wet] = 0.0
    return depth, out


def stream_polylines(stream: np.ndarray, rec: np.ndarray) -> list:
    """Trace drainage lines as lists of (row, col), split at confluences."""
    h, w = stream.shape
    s = stream.ravel()
    r = rec.ravel()
    indeg = np.zeros(h * w, dtype=np.int32)
    src = np.nonzero(s)[0]
    for i in src:
        j = r[i]
        if j >= 0 and s[j]:
            indeg[j] += 1
    starts = [i for i in src if indeg[i] != 1]  # heads (0) and confluences (>1)
    lines = []
    for i in starts:
        path = [i]
        j = r[i]
        while j >= 0 and s[j]:
            path.append(j)
            if indeg[j] != 1:
                break
            j = r[j]
        if len(path) >= 2:
            lines.append([(p // w, p % w) for p in path])
    return lines
