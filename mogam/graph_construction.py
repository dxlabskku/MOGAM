"""Build the object co-occurrence graph of one vlog (Section 3.6, Eq. 1).

Frames are sampled at 1 fps and passed through YOLOv5x (COCO weights) with a
confidence threshold of 0.70. For every frame, the set of detected classes is
recorded once each (multiple instances of a class in one frame count once).
The adjacency matrix counts, for each pair of classes, the number of frames in
which both were detected, and divides by the number of frames so that vlogs of
different length remain comparable. The diagonal counts frames in which the
class itself was detected.

Usage:
    from mogam.graph_construction import cooccurrence_graph
    A = cooccurrence_graph([{0, 41}, {0}, {41, 45}], n_frames=3)   # 80 x 80
"""
import numpy as np

from .coco import NUM_CLASSES


def cooccurrence_graph(frame_detections, n_frames=None):
    """frame_detections: iterable of sets of COCO class indices, one per frame.
    n_frames: number of frames (defaults to len(frame_detections)).
    Returns a float32 (80, 80) symmetric matrix normalised by n_frames."""
    frames = list(frame_detections)
    n = n_frames if n_frames is not None else len(frames)
    A = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    for det in frames:
        idx = sorted(set(int(i) for i in det))
        for i in idx:
            A[i, i] += 1.0
            for j in idx:
                if j > i:
                    A[i, j] += 1.0
                    A[j, i] += 1.0
    if n > 0:
        A /= float(n)
    return A


def from_yolo_results(results, conf=0.70):
    """Convert a list of per-frame YOLOv5 detection arrays
    (each row: x1, y1, x2, y2, conf, cls) into class sets at the given threshold."""
    sets = []
    for r in results:
        r = np.asarray(r)
        if r.size == 0:
            sets.append(set())
            continue
        keep = r[:, 4] >= conf
        sets.append(set(r[keep, 5].astype(int).tolist()))
    return sets
