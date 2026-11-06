"""Record the engagement distance proxy (face-box height / frame height) at distances YOU measured with a tape.

    python3 tools/calibrate_engagement.py --camera 0 --distances 0.5 1.0 2.0 --out docs/results/engagement_calibration.json

For each distance it asks the visitor to stand there (the face toward the camera), records ~`--seconds` of frames with the same
face detector the live demo uses, and stores the ratio statistics. No images, embeddings or identities are kept: only distances
and ratios. It prints a table and a SUGGESTION for enter_ratio / exit_ratio; it never edits a config.

The ratio depends on the camera, its height and angle, and on whose face it is (a child's face is smaller), so calibrate per
site and with a person of typical face size, and treat the numbers as the visitor you measured, not everyone.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional


def summarise(ratios: List[float]) -> Optional[dict]:
    if not ratios:
        return None
    r = sorted(ratios)
    q = lambda p: r[min(len(r) - 1, int(p * len(r)))]
    return {"n": len(r), "median": round(statistics.median(r), 3), "p10": round(q(0.10), 3), "p90": round(q(0.90), 3),
            "min": round(r[0], 3), "max": round(r[-1], 3)}


def suggest(by_distance: Dict[float, Optional[dict]], engage_distance: float, exit_factor: float = 0.75) -> Optional[dict]:
    """enter_ratio = the 10th percentile at the engage distance (so the visitor standing there almost always counts);
    exit_ratio = exit_factor x enter_ratio. A suggestion only: it ignores smaller faces (children)."""
    s = by_distance.get(engage_distance)
    if not s:
        return None
    enter = round(s["p10"], 3)
    return {"engage_distance_m": engage_distance, "enter_ratio": enter, "exit_ratio": round(enter * exit_factor, 3)}


def record(detect_ratio: Callable[[], Optional[float]], seconds: float, sleep: Callable[[float], None] = time.sleep,
           now: Callable[[], float] = time.time, period: float = 0.1) -> List[float]:
    """Call detect_ratio() (largest face's height ratio or None) repeatedly for `seconds`."""
    out, end = [], now() + seconds
    while now() < end:
        r = detect_ratio()
        if r is not None:
            out.append(r)
        sleep(period)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--distances", type=float, nargs=3, required=True, metavar="M", help="three distances in metres, measured with a tape")
    ap.add_argument("--engage-distance", type=float, default=None, help="which of the three should trigger engagement (default: the middle one)")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--detector", choices=["opencv", "mtcnn"], default="mtcnn")
    ap.add_argument("--no-prompt", action="store_true", help="do not wait for Enter between distances (smoke test only)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import os
    import cv2
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # repo root, so `modules` imports
    from modules.face_webcam.face_id import FaceIdentifier
    fid = FaceIdentifier(detector=a.detector)                      # detection only: never enrols, saves or loads
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        print(f"cannot open camera {a.camera}", file=sys.stderr); return 1
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def largest_ratio() -> Optional[float]:
        ok, f = cap.read()
        if not ok:
            return None
        faces = fid.identify_all(f, max_faces=4)
        if not faces:
            return None
        return max((b[3] - b[1]) for _, _, b in faces) / float(f.shape[0])

    result: Dict[str, dict] = {}
    by_d: Dict[float, Optional[dict]] = {}
    for d in a.distances:
        if not a.no_prompt:
            input(f"Stand {d} m from the camera, face toward it, then press Enter... ")
        for _ in range(5):
            cap.read()
        s = summarise(record(largest_ratio, a.seconds))
        by_d[d] = s
        result[str(d)] = s
        print(f"{d:>4} m: {s if s else 'NO FACE DETECTED'}")
    cap.release()
    eng = a.engage_distance if a.engage_distance is not None else a.distances[1]
    sug = suggest(by_d, eng)
    print("suggestion (not applied):", sug)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"measured_by": "tape", "camera": a.camera, "detector": a.detector, "distances": result, "suggestion": sug,
                       "smoke_test_distances_unmeasured": bool(a.no_prompt)}, f, indent=1)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
