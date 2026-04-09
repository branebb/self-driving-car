import numpy as np
from svgpathtools import svg2paths, Line, QuadraticBezier, CubicBezier, Arc
from scipy.ndimage import gaussian_filter1d

# Helper functions to sample points from SVG path segments
def complex_to_array(z):
    return np.array([z.real, z.imag])

def sample_line(seg, tvals):
    p0 = complex_to_array(seg.start)
    p1 = complex_to_array(seg.end)
    t = tvals[:, None]
    return p0 + (p1 - p0) * t

def sample_quadratic(seg, tvals):
    P0 = complex_to_array(seg.start)
    P1 = complex_to_array(seg.control)
    P2 = complex_to_array(seg.end)

    t = tvals[:, None]
    return (
        (1 - t)**2 * P0 +
        2*(1 - t)*t * P1 +
        t**2 * P2
    )

def sample_cubic(seg, tvals):
    P0 = complex_to_array(seg.start)
    P1 = complex_to_array(seg.control1)
    P2 = complex_to_array(seg.control2)
    P3 = complex_to_array(seg.end)

    t = tvals[:, None]
    return (
        (1 - t)**3 * P0 +
        3*(1 - t)**2 * t * P1 +
        3*(1 - t)*t**2 * P2 +
        t**3 * P3
    )


def sample_arc(seg, tvals):
    cubics = seg.as_cubic_curves()
    pts = []
    for c in cubics:
        pts.append(sample_cubic(c, tvals))
    return np.concatenate(pts, axis=0)


def sample_segment(seg, tvals):
    if isinstance(seg, Line):
        return sample_line(seg, tvals)

    elif isinstance(seg, QuadraticBezier):
        return sample_quadratic(seg, tvals)

    elif isinstance(seg, CubicBezier):
        return sample_cubic(seg, tvals)

    elif isinstance(seg, Arc):
        return sample_arc(seg, tvals)

    else:
        raise TypeError(f"Unsupported segment type: {type(seg)}")


def build_arc_length_lookup(path):
    segments = list(path)
    lengths = np.array([seg.length() for seg in segments])
    cum_lengths = np.concatenate(([0], np.cumsum(lengths)))
    return segments, cum_lengths


def sample_path_arc_length(path, N):
    segments, cum_lengths = build_arc_length_lookup(path)
    total_length = cum_lengths[-1]

    s_vals = np.linspace(0, total_length, N, endpoint=False)
    points = np.zeros((N, 2))

    for i, s in enumerate(s_vals):
        seg_idx = np.searchsorted(cum_lengths, s) - 1
        seg_idx = np.clip(seg_idx, 0, len(segments) - 1)

        seg = segments[seg_idx]
        local_s = s - cum_lengths[seg_idx]

        t = seg.ilength(local_s)
        points[i] = sample_segment(seg, np.array([t]))[0]

    return points

def extract_centerline_raw(svg_file, N=2000, path_index=0):
    paths, _ = svg2paths(svg_file)
    #find path with longest length
    track_path = paths[path_index]
    print(f"Number of paths: {len(paths)}")

    centerline = sample_path_arc_length(track_path, N)

    return centerline

# Helper function to compute boundaries with normal smoothing
def compute_boundaries_with_normal_smoothing(centerline, track_width, sigma=9):

    half_width = track_width / 2

    forward = np.roll(centerline, -1, axis=0)
    backward = np.roll(centerline, 1, axis=0)

    tangents = forward - backward

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    tangents = tangents / norms

    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    normals[:, 0] = gaussian_filter1d(normals[:, 0], sigma=sigma, mode="wrap")
    normals[:, 1] = gaussian_filter1d(normals[:, 1], sigma=sigma, mode="wrap")

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normals = normals / norms

    left_boundary = centerline + half_width * normals
    right_boundary = centerline - half_width * normals

    return left_boundary, right_boundary, tangents, normals

# Main function to get track info
def get_track_info(svg_file, path_index=0, track_width=10, N=2000, sigma=9, 
                   start_point_index=None, horizontal_flip=False, vertical_flip=False, change_direction=False,
                   original_track_length=None):
    centerline = extract_centerline_raw(svg_file, N, path_index)
    if start_point_index is not None:
        centerline = np.vstack((centerline[start_point_index:], centerline[:start_point_index]))

    if horizontal_flip:
        centerline[:, 0] = -centerline[:, 0]
    
    if vertical_flip:
        centerline[:, 1] = -centerline[:, 1]

    if change_direction:
        centerline = centerline[::-1]
    
    if not np.array_equal(centerline[0], centerline[-1]):
        centerline = np.vstack([centerline, centerline[0]])

    if original_track_length is not None:
        diffs = np.diff(centerline, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(segment_lengths)
        scale_factor = original_track_length / total_length
        centerline *= scale_factor

    min_x = np.min(centerline[:, 0])
    min_y = np.min(centerline[:, 1])
    centerline[:, 0] -= min_x
    centerline[:, 1] -= min_y

    left_boundary, right_boundary, tangents, normals = compute_boundaries_with_normal_smoothing(centerline, track_width, sigma)
    return centerline, left_boundary, right_boundary, tangents, normals