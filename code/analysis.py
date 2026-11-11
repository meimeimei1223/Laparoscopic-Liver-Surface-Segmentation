"""Label statistics (Tables 1, 4), agreement of zone borders with the official L3D-2K lines (Table 5)
and intra-rater repeatability (Table 6).

    python analysis.py --labels zone_labels.json --redraw zone_labels_redraw100.json --l3d2k L3D-2K/ --out results/

Every value is averaged image -> patient -> over patients; 95% CIs come from a patient bootstrap (10,000 draws).
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from rle_tools import decode, load_labels

N_BOOT, SEED, END_TOL = 10_000, 1234, 1.5
RES = {720: "720p", 1080: "1080p", 2160: "4K"}
EIGHT = np.ones((3, 3), bool)
LINE_CLASS = {"ridge": "r", "ligament": "l"}   # first letter of the L3D-2K shape label
patient = lambda s: "_".join(s.split("_")[:2])  # noqa: E731
near = lambda m: ndimage.binary_dilation(m, structure=EIGHT)  # noqa: E731


def agg(pairs, rng, sorted_patients=True):
    """[(patient, value)] -> mean of patient means and its patient-bootstrap 95% CI."""
    by = {}
    for p, v in pairs:
        if v == v:
            by.setdefault(p, []).append(v)
    pats = sorted(by) if sorted_patients else list(by)
    pm = np.array([np.mean(by[p]) for p in pats])
    out = {"mean": float(pm.mean()), "n_images": sum(map(len, by.values())), "n_patients": len(pats)}
    if len(pats) > 1:
        m = pm[rng.integers(0, len(pats), size=(N_BOOT, len(pats)))].mean(1)
        out["ci95"] = [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    return out


# ---------------------------------------------------------------- Tables 1 and 4
def label_stats(labels):
    rows = [(r, np.array([int(r["counts"][str(z)]) for z in range(7)])) for r in labels.values()]
    t1 = {s: {"images": sum(r["split"] == s for r, _ in rows), "patients": len({r["patient"] for r, _ in rows if r["split"] == s}),
              **{v: sum(r["split"] == s and RES[r["height"]] == v for r, _ in rows) for v in RES.values()}} for s in ("train", "val", "test")}
    t4 = []
    for z in range(7):
        has = [(r, c) for r, c in rows if c[z] > 0]
        share = [100 * c[0] / (r["height"] * r["width"]) if z == 0 else 100 * c[z] / c[1:7].sum() for r, c in has]
        t4.append({"zone": z, "images": len(has), "images_pct": round(100 * len(has) / len(rows), 2), "patients": len({r["patient"] for r, _ in has}),
                   "area_pct_median": float(np.median(share)), "area_pct_q1": float(np.percentile(share, 25)),
                   "area_pct_q3": float(np.percentile(share, 75))})
    return {"table1": t1, "table4": t4}


# ---------------------------------------------------------------- Table 5
def centerline(shapes, letter, H, W):
    """Rasterized official line (1 px) and its end points, as (row, col)."""
    img, ends = np.zeros((H, W), np.uint8), []
    for sh in shapes:
        if not sh["label"].startswith(letter):
            continue
        p = sh["points"]
        for i in range(1, len(p)):
            cv2.line(img, tuple(map(int, p[i - 1])), tuple(map(int, p[i])), 1, 1)
        if len(p) >= 2:
            ends += [(min(max(int(q[1]), 0), H - 1), min(max(int(q[0]), 0), W - 1)) for q in (p[0], p[-1])]
    return np.argwhere(img > 0), np.array(ends, int).reshape(-1, 2)


def border_distance(border, line, scale):
    """Distances (1080p px) from border pixels to the nearest line pixel; points whose nearest pixel is a line end are dropped."""
    lp, ends = line
    is_end = np.zeros(len(lp), bool)
    if len(ends):
        is_end[cKDTree(ends).query_ball_point(lp, END_TOL, return_length=True).nonzero()[0]] = True
    d, i = cKDTree(lp).query(np.argwhere(border), workers=1)
    d = d[~is_end[i]] * scale
    return {"median": float(np.median(d)), "lt15": float((d < 15).mean()), "lt30": float((d < 30).mean())} if len(d) else None


def validity_image(job):
    rec, root = job
    r = decode(rec)
    H, W = r.shape
    right = (r == 1) & near(np.isin(r, (2, 3)))
    left = (r == 4) & near(r == 5)
    ap, lra = right | left, (r == 1) & near(r == 4)
    # the larger ventral zone must show its own dorsal zone
    dominant = right.any() if (r == 1).sum() >= (r == 4).sum() else left.any()
    shapes = json.loads((Path(root) / rec["split"] / "labels" / f"{rec['name']}.json").read_text(encoding="utf-8"))["shapes"]
    out = {"stem": rec["name"], "patient": rec["patient"]}
    for name, border, ok in (("ridge", ap, dominant), ("ligament", lra, True)):
        line = centerline(shapes, LINE_CLASS[name], H, W)
        if ok and border.any() and len(line[0]):
            out[name] = border_distance(border, line, 1080.0 / H)
    return out


def validity(labels, root, workers):
    recs = sorted(labels.values(), key=lambda r: r["name"])
    with Pool(workers) as pool:
        per = pool.map(validity_image, [(r, root) for r in recs], chunksize=10)
    t5 = {}
    for line in ("ridge", "ligament"):
        vals = [(x["patient"], x[line]) for x in per if x.get(line)]
        # one fresh generator per value; patients in order of first appearance (images sorted by name)
        t5[line] = {k: agg([(p, v[k]) for p, v in vals], np.random.default_rng(SEED), sorted_patients=False) for k in ("median", "lt15", "lt30")}
    return t5, per


# ---------------------------------------------------------------- Table 6
def mean_dice(x, y, zones):
    """Mean Dice over zones present in either map; a zone present in only one counts 0."""
    v = []
    for z in zones:
        a, b = x == z, y == z
        if a.any() or b.any():
            v.append(2.0 * (a & b).sum() / (a.sum() + b.sum()) if a.any() and b.any() else 0.0)
    return float(np.mean(v)) if v else float("nan")


def dice(a, b):
    return 2.0 * int((a & b).sum()) / (int(a.sum()) + int(b.sum()))


def repeatability(labels, redraw):
    per = []
    for rec in sorted(redraw.values(), key=lambda r: r["order"]):   # redraw order (1-100)
        x, y = decode(rec), decode(labels[rec["name"]])
        keep = (x != 6) & (y != 6)
        per.append({"stem": rec["name"], "patient": rec["patient"], "macro_dice16": mean_dice(x, y, range(1, 7)),
                    "macro_dice15": mean_dice(x, y, range(1, 6)),
                    "macro_dice15_no_caudate": mean_dice(x[keep], y[keep], range(1, 6)), "liver_dice": dice(x > 0, y > 0),
                    **{f"zone{z}_dice": dice(x == z, y == z) if (x == z).any() and (y == z).any() else float("nan") for z in range(1, 7)}})
    n = len({p["patient"] for p in per})
    rng = np.random.default_rng(SEED)
    t6 = {"macro_dice16": agg([(p["patient"], p["macro_dice16"]) for p in per], rng)}
    # Four draws (two here, two below) are skipped so that the random stream matches the one used for the reported intervals.
    for _ in range(2):
        rng.integers(0, n, size=(N_BOOT, n))
    t6["macro_dice15"] = agg([(p["patient"], p["macro_dice15"]) for p in per], rng)
    for _ in range(2):
        rng.integers(0, n, size=(N_BOOT, n))
    t6["macro_dice15_no_caudate"] = agg([(p["patient"], p["macro_dice15_no_caudate"]) for p in per], rng)
    rng = np.random.default_rng(SEED)
    for z in range(1, 7):
        t6[f"zone{z}_dice"] = agg([(p["patient"], p[f"zone{z}_dice"]) for p in per], rng)
    t6["liver_dice"] = agg([(p["patient"], p["liver_dice"]) for p in per], rng)
    return t6, per


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--redraw", required=True)
    ap.add_argument("--l3d2k", required=True, help="L3D-2K root with <split>/labels/<name>.json (official lines)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(a.labels)
    res = label_stats(labels)
    res["table5"], per5 = validity(labels, a.l3d2k, a.workers)
    res["table6"], per6 = repeatability(labels, load_labels(a.redraw))
    write_csv(out / "table5_per_image.csv",
              [{"stem": x["stem"], "patient": x["patient"], "line": k, **x[k]} for x in per5 for k in LINE_CLASS if x.get(k)])
    write_csv(out / "table6_per_image.csv", per6)
    (out / "tables.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
