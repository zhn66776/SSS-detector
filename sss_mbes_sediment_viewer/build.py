#!/usr/bin/env python3
"""Stream Costa MBES XYZ grids into local web tile pyramids and export official vectors.

Never extracts the source archives.  Each source line is read from the ZIP
stream; the derived cache under ``derived/`` is the only thing written.

Evidence about the bathymetry members (see README):

* ``..._2m_UTM30N_LAT_wip00.pts`` and ``..._4m_...`` carry real LAT elevations
  (about -3 m to -77 m at Costa Head).
* ``..._0.5m_UTM30N_LAT_wip00.pts`` does NOT: its third column is an integer
  1..35 that stripes along survey lines and is uncorrelated with the 2 m
  elevations (r = -0.18 over a shared 300 m box).  It behaves like a per-cell
  sounding count.  The real 0.5 m LAT grid is listed as "held offline" in the
  package README, so the 2 m member is the finest genuine elevation available.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import resource
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import shapefile
from PIL import Image

ROOT = Path(__file__).parent
OLD = ROOT.parent / 'sss_contact_inspector'
GIS = OLD / 'source_packages/13_86_GIS_Shapefiles.zip'
BATHY = OLD / 'source_packages/13_86_Processed_Bathymetry_Data.zip'
MEMBERS = {
    '2m': ('C13007_Costa_AllData_XYZ_2m_UTM30N_LAT_wip00.pts', 2.0),
    '4m': ('C13007_Costa_AllData_XYZ_4m_UTM30N_LAT_wip00.pts', 4.0),
}
TILE = 256
ROUGH_WINDOW = 11  # cells; 22 m at 2 m
HILLSHADE = {'azimuth_deg': 315, 'altitude_deg': 45, 'vertical_exaggeration': 8.0}  # flat shelf: metre-scale relief needs strong exaggeration
# Colour ramps: (position, rgb).  Elevation is a bathymetric blue->sand ramp,
# roughness a single-hue sequential ramp so it never reads as sediment class.
ELEVATION_RAMP = [(0, (8, 29, 88)), (.25, (33, 102, 172)), (.5, (67, 162, 202)), (.75, (171, 217, 187)), (1, (250, 240, 200))]
ROUGHNESS_RAMP = [(0, (255, 255, 235)), (.5, (245, 160, 60)), (1, (128, 18, 30))]


def write_json(path: Path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(',', ':')) if compact else json.dumps(value, indent=2))


# ----------------------------------------------------------------------------- vectors
def extract_inner(outer: Path, inner: str, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outer) as z, zipfile.ZipFile(io.BytesIO(z.read(inner))) as q:
        q.extractall(target)


def round_coords(coords, ndigits=2):
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], ndigits), round(coords[1], ndigits)]
    return [round_coords(c, ndigits) for c in coords]


def features(shp: Path, normalise=False):
    reader = shapefile.Reader(str(shp))
    names = [f[0] for f in reader.fields[1:]]
    out = []
    for i, s in enumerate(reader.iterShapeRecords()):
        props = dict(zip(names, s.record))
        geo = s.shape.__geo_interface__
        if normalise and 'Layer' in props:
            layer = str(props['Layer'])
            props['Layer_normalized'] = 'Gravelly SAND' if layer.startswith('Gravelly SAND') else layer
        geo = {'type': geo['type'], 'coordinates': round_coords(geo['coordinates'])}
        out.append({'type': 'Feature', 'id': i, 'properties': props, 'geometry': geo})
    return out


def gis(out: Path):
    temp = out / '_gis'
    if not temp.exists():
        for n in ['SBF Polygons.zip', 'SSS Listing.zip', 'Sand Wave.zip', 'Planned Survey Lines.zip', 'Trackplots.zip']:
            extract_inner(GIS, n, temp / n[:-4])
    spec = {
        'sediments': ('SBF Polygons', 'C13007a_CostaHead_SBF_Polygons.shp', True),
        'sediment_lines': ('SBF Polygons', 'C13007a_CostaHead_SBF_Polylines.shp', False),
        'sss_contacts': ('SSS Listing', 'C13007a_CostaHead_SSSListing.shp', False),
        'sand_waves': ('Sand Wave', 'C13007a_CostaHead_SandWavePoint.shp', False),
    }
    layers, counts = {}, {}
    for key, (folder, name, norm) in spec.items():
        matches = list((temp / folder).rglob(name))
        if not matches:
            raise FileNotFoundError(f'official shapefile missing: {name}')
        f = features(matches[0], norm)
        layers[key] = {'type': 'FeatureCollection', 'features': f}
        counts[key] = len(f)
    write_json(out / 'data' / 'vectors.json', layers, compact=True)
    return counts


# ----------------------------------------------------------------------------- points
def stream_points(member: str, limit=0):
    """Yield (x, y, value) from one XYZ member without extracting the archive."""
    n = 0
    with zipfile.ZipFile(BATHY) as z, z.open(member) as raw:
        for line in io.TextIOWrapper(raw, encoding='ascii'):
            a = line.split()
            if len(a) != 3:
                continue
            n += 1
            yield float(a[0]), float(a[1]), float(a[2])
            if limit and n >= limit:
                return


def scan(points):
    """Extent, value range and duplicate count of a point stream; bounded memory.

    Also reports whether every value is an integer: that is how the 0.5 m
    member betrays itself as a count rather than an elevation.
    """
    n = 0
    bounds = [math.inf, math.inf, -math.inf, -math.inf]
    vmin, vmax = math.inf, -math.inf
    integer_valued = 0
    prev = None
    duplicates = 0
    for x, y, v in points:
        n += 1
        if x < bounds[0]: bounds[0] = x
        if y < bounds[1]: bounds[1] = y
        if x > bounds[2]: bounds[2] = x
        if y > bounds[3]: bounds[3] = y
        if v < vmin: vmin = v
        if v > vmax: vmax = v
        if v == int(v): integer_valued += 1
        if prev == (x, y): duplicates += 1
        prev = (x, y)
    return {'points': n, 'bounds': bounds, 'value_min': vmin, 'value_max': vmax,
            'integer_valued_fraction': integer_valued / n if n else None, 'duplicate_consecutive_points': duplicates}


def check_elevation_like(summary: dict, member: str):
    """Refuse a member whose third column is an integer count, not elevation."""
    if summary['points'] and summary['integer_valued_fraction'] == 1 and summary['value_max'] - summary['value_min'] < 100:
        raise ValueError(f'{member}: every value is an integer in [{summary["value_min"]}, {summary["value_max"]}]; '
                         'this column is a count, not an LAT elevation (the 0.5 m wip00 member has this property).')


def grid_spec(bounds, res, region=None):
    """Cell-centred grid covering the node bounds (or a region), north-up."""
    x0, y0, x1, y1 = bounds
    left, top = x0 - res / 2, y1 + res / 2
    right, bottom = x1 + res / 2, y0 - res / 2
    if region:
        left, bottom = max(left, region[0]), max(bottom, region[1])
        right, top = min(right, region[2]), min(top, region[3])
    w, h = math.ceil((right - left) / res), math.ceil((top - bottom) / res)
    if w <= 0 or h <= 0:
        raise ValueError('region does not intersect the bathymetry extent')
    return {'left': left, 'top': top, 'right': left + w * res, 'bottom': top - h * res, 'width': w, 'height': h, 'resolution_m': res}


def fill_grid(points, spec):
    """Second pass: one float32 grid, NaN where no sounding node exists."""
    res, left, top = spec['resolution_m'], spec['left'], spec['top']
    w, h = spec['width'], spec['height']
    grid = np.full((h, w), np.nan, np.float32)
    kept = duplicates = 0
    for x, y, v in points:
        c = int((x - left) // res)
        r = int((top - y) // res)
        if 0 <= r < h and 0 <= c < w:
            if not math.isnan(grid[r, c]):
                duplicates += 1
            grid[r, c] = v
            kept += 1
    return grid, kept, duplicates


# ----------------------------------------------------------------------------- rasters
def box_sum(a: np.ndarray, k: int) -> np.ndarray:
    """Sum over a k x k window centred on every cell (zero padded), via separable cumsums."""
    h = k // 2
    p = np.pad(a.astype(np.float64), h)
    c = np.cumsum(p, axis=0)
    c = np.vstack([np.zeros((1, c.shape[1])), c])
    s = c[k:] - c[:-k]
    c = np.cumsum(s, axis=1)
    c = np.hstack([np.zeros((c.shape[0], 1)), c])
    return c[:, k:] - c[:, :-k]


def rough(z: np.ndarray, window: int = ROUGH_WINDOW, band: int = 512) -> np.ndarray:
    """Absolute residual from the local ``window``-cell mean, processed in row bands.

    The local mean removes the broad depth trend; what remains is a geometry
    roughness diagnostic.  Cells need at least half of the window valid, so
    edges and holes stay transparent instead of leaking bias.
    """
    h = window // 2
    min_count = math.ceil(window * window / 2)
    ok = np.isfinite(z)
    out = np.full(z.shape, np.nan, np.float32)
    rows = z.shape[0]
    for r0 in range(0, rows, band):
        r1 = min(rows, r0 + band)
        top, bottom = max(0, r0 - h), min(rows, r1 + h)
        pad = ((h - (r0 - top), h - (bottom - r1)), (0, 0))
        blk_ok = np.pad(ok[top:bottom], pad)
        blk = np.pad(np.where(ok[top:bottom], z[top:bottom], 0.0), pad)
        sums = box_sum(blk, window)[h:h + (r1 - r0)]
        cnt = box_sum(blk_ok, window)[h:h + (r1 - r0)]
        mean = sums / np.maximum(cnt, 1)
        valid = ok[r0:r1] & (cnt >= min_count)
        out[r0:r1] = np.where(valid, np.abs(z[r0:r1] - mean), np.nan)
    return out


def reduce2(g: np.ndarray, band: int = 1024) -> np.ndarray:
    """Half-resolution grid by 2x2 NaN-aware block mean (row-banded, bounded memory)."""
    hh, ww = g.shape
    h2, w2 = (hh + 1) // 2, (ww + 1) // 2
    out = np.full((h2, w2), np.nan, np.float32)
    for r0 in range(0, h2, band):
        r1 = min(h2, r0 + band)
        blk = np.full(((r1 - r0) * 2, w2 * 2), np.nan, np.float32)
        src = g[r0 * 2:r1 * 2]
        blk[:src.shape[0], :ww] = src
        v = blk.reshape(r1 - r0, 2, w2, 2)
        ok = np.isfinite(v)
        s = np.where(ok, v, 0).sum((1, 3))
        c = ok.sum((1, 3))
        out[r0:r1] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out


def ramp_lut(stops):
    """256-entry RGB lookup from (position, rgb) stops."""
    xs = np.array([s[0] for s in stops])
    lut = np.zeros((256, 3), np.uint8)
    t = np.linspace(0, 1, 256)
    for ch in range(3):
        lut[:, ch] = np.interp(t, xs, [s[1][ch] for s in stops]).round()
    return lut


def hillshade(z: np.ndarray, res: float) -> np.ndarray:
    """GDAL-style shaded relief in [0, 1]; flat (1.0) wherever a gradient is undefined."""
    az, alt = math.radians(HILLSHADE['azimuth_deg']), math.radians(HILLSHADE['altitude_deg'])
    vex = HILLSHADE['vertical_exaggeration']
    if min(z.shape) < 2:
        return np.ones(z.shape, np.float32)
    dzdy, dzdx = np.gradient(z.astype(np.float64), res)
    dzdy = -dzdy  # rows run south, so flip to a north-positive derivative
    slope = np.arctan(vex * np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    shade = math.sin(alt) * np.cos(slope) + math.cos(alt) * np.sin(slope) * np.cos(az - math.pi / 2 - aspect)
    return np.where(np.isfinite(shade), np.clip(shade, 0, 1), 1.0).astype(np.float32)


def hillshade_window(level: np.ndarray, r0: int, c0: int, res: float) -> np.ndarray:
    """Shade for the tile at (r0, c0) using a one-cell halo, so seams shade consistently."""
    halo = level[max(0, r0 - 1):r0 + TILE + 1, max(0, c0 - 1):c0 + TILE + 1]
    oy, ox = 1 - min(1, r0), 1 - min(1, c0)  # halo row/col 0 is the tile's own edge only at the grid edge
    sh = np.ones((TILE + 2, TILE + 2), np.float32)
    sh[oy:oy + halo.shape[0], ox:ox + halo.shape[1]] = hillshade(halo, res)
    return sh[1:TILE + 1, 1:TILE + 1]


def render(values: np.ndarray, lo: float, hi: float, lut: np.ndarray, shade: np.ndarray | None = None) -> Image.Image:
    """RGBA image with a fixed value range so tiles share one colour scale; NaN transparent."""
    valid = np.isfinite(values)
    u = np.clip((np.nan_to_num(values) - lo) / (hi - lo + 1e-9), 0, 1)
    idx = (u * 255).astype(np.uint8)
    rgb = lut[idx].astype(np.float32)
    if shade is not None:
        rgb *= (0.3 + 0.7 * shade)[..., None]
    im = np.zeros((*values.shape, 4), np.uint8)
    im[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    im[..., 3] = valid * 255
    return Image.fromarray(im)


def write_pyramid(grid: np.ndarray, out_dir: Path, lo: float, hi: float, lut: np.ndarray, res: float | None = None):
    """Native tiles at z=max_zoom, then coarser levels from value-space 2x2 means.

    With ``res`` the tiles are hillshaded (one-cell halo so tile seams shade
    consistently); each level shades its own resolution.
    """
    max_zoom = max(0, math.ceil(math.log2(max(grid.shape) / TILE)))
    present = {}
    level = grid
    for z in range(max_zoom, -1, -1):
        keys = []
        h, w = level.shape
        level_res = res * 2 ** (max_zoom - z) if res else None
        for ty in range(math.ceil(h / TILE)):
            for tx in range(math.ceil(w / TILE)):
                r0, c0 = ty * TILE, tx * TILE
                block = level[r0:r0 + TILE, c0:c0 + TILE]
                if not np.isfinite(block).any():
                    continue
                full = np.full((TILE, TILE), np.nan, np.float32)
                full[:block.shape[0], :block.shape[1]] = block
                shade = hillshade_window(level, r0, c0, level_res) if level_res else None
                d = out_dir / str(z)
                d.mkdir(parents=True, exist_ok=True)
                render(full, lo, hi, lut, shade).save(d / f'{tx}_{ty}.png', compress_level=3)
                keys.append(f'{tx}_{ty}')
        present[str(z)] = keys
        if z:
            level = reduce2(level)
    return {'max_zoom': max_zoom, 'tile_size': TILE, 'present': present, 'tiles': sum(len(v) for v in present.values())}


def percentiles(grid: np.ndarray, q, sample=2_000_000):
    """Percentiles of the valid cells from a strided subsample; keeps the sort bounded."""
    valid = grid[np.isfinite(grid)]
    if not valid.size:
        return [0.0 for _ in q]
    valid = valid[::max(1, valid.size // sample)]
    return [float(v) for v in np.percentile(valid, q)]


def build_mbes(out: Path, member_key: str, region, max_points=0):
    member, res = MEMBERS[member_key]
    started = time.time()
    summary = scan(stream_points(member, max_points))
    check_elevation_like(summary, member)
    spec = grid_spec(summary['bounds'], res, region)
    grid, kept, dup_cells = fill_grid(stream_points(member, max_points), spec)
    if kept == 0:
        raise ValueError('no soundings inside the requested region')
    elev_lo, elev_hi = percentiles(grid, [1, 99])
    roughness = rough(grid)
    rough_hi = percentiles(roughness, [98])[0] or 1.0
    d = out / 'data' / 'mbes'
    elev = write_pyramid(grid, d / 'elevation', elev_lo, elev_hi, ramp_lut(ELEVATION_RAMP), res)
    rgh = write_pyramid(roughness, d / 'roughness', 0.0, rough_hi, ramp_lut(ROUGHNESS_RAMP))
    return {
        'member': member, 'source_column': 'LAT elevation (m), negative below LAT', 'crs': 'EPSG:32630',
        'resolution_m': res, 'bounds': {k: spec[k] for k in ('left', 'bottom', 'right', 'top')},
        'grid_size': [spec['width'], spec['height']], 'region': region,
        'scan': summary, 'points_kept': kept, 'duplicate_cells': dup_cells,
        'coverage_fraction': float(np.isfinite(grid).mean()),
        'tile_size': TILE, 'max_zoom': elev['max_zoom'], 'tiles': {'elevation': elev['tiles'], 'roughness': rgh['tiles']},
        'present': elev['present'], 'present_roughness': rgh['present'],
        'elevation': {'range': [elev_lo, elev_hi], 'range_rule': '1st-99th percentile of valid cells', 'units': 'm LAT',
                      'hillshade': HILLSHADE},
        'roughness': {'method': f'{ROUGH_WINDOW}x{ROUGH_WINDOW} ({ROUGH_WINDOW * res:g} m) local-mean detrended absolute residual',
                      'window_m': ROUGH_WINDOW * res, 'units': 'm', 'range': [0.0, rough_hi], 'range_rule': '0 to 98th percentile',
                      'valid_support': f'at least {math.ceil(ROUGH_WINDOW ** 2 / 2)}/{ROUGH_WINDOW ** 2} cells; edge/no-data transparent',
                      'caveat': 'diagnostic geometry metric, not sediment ground truth'},
        'elapsed_s': time.time() - started,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=ROOT / 'derived')
    p.add_argument('--member', choices=sorted(MEMBERS), default='2m', help='XYZ member with genuine LAT elevations')
    p.add_argument('--region', nargs=4, type=float, metavar=('X0', 'Y0', 'X1', 'Y1'), help='restrict the grid to this EPSG:32630 box')
    p.add_argument('--max-points', type=int, default=0, help='stop after N source points (quick test builds)')
    p.add_argument('--dry-run', action='store_true', help='export vectors and the scan audit only')
    a = p.parse_args()
    a.out.mkdir(exist_ok=True)
    start = time.time()
    counts = gis(a.out)
    audit = {
        'crs': 'EPSG:32630 WGS 84 / UTM zone 30N',
        'crs_evidence': 'official GIS .prj and Costa ERS NUTM30/WGS84',
        'bathymetry_zip_bytes': BATHY.stat().st_size,
        'vector_counts': counts,
        'member_notes': {
            '0.5m': 'C13007_Costa_AllData_XYZ_0.5m_UTM30N_LAT_wip00.pts third column is an integer 1..35 striping along survey lines, uncorrelated with the 2 m elevations; treated as a sounding count and never rendered as bathymetry',
            '2m': 'genuine LAT elevations, cell-centred 2 m nodes',
            '4m': 'genuine LAT elevations, cell-centred 4 m nodes',
        },
        'dry_run': a.dry_run,
    }
    if a.dry_run:
        member, _ = MEMBERS[a.member]
        audit['scan'] = scan(stream_points(member, a.max_points or 250000))
    else:
        audit['mbes'] = build_mbes(a.out, a.member, a.region, a.max_points)
    audit['elapsed_s'] = time.time() - start
    audit['peak_rss_kb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 if sys.platform == 'darwin' else 1)
    write_json(a.out / 'data' / 'audit.json', audit)
    printable = dict(audit)
    if 'mbes' in printable:
        printable['mbes'] = {k: v for k, v in printable['mbes'].items() if not k.startswith('present')}
    print(json.dumps(printable, indent=2))


if __name__ == '__main__':
    main()
