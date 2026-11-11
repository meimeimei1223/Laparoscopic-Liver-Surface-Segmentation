# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Meidai Kasai
"""Convert zone-label JSON (run-length encoded) to and from PNG label maps, and verify it.

    python rle_tools.py to-png   zone_labels.json out/                    # out/<split>/<name>.png, 8-bit grayscale
    python rle_tools.py from-png pngs/ zone_labels.json [--images L3D-2K/] # pngs/<split>/<name>.png -> JSON
    python rle_tools.py verify   zone_labels.json [--png-dir pngs/] [--images L3D-2K/]

verify checks counts and mask_hash; with --png-dir also the pixels against PNGs, and with --images
the image_sha256 against the L3D-2K images (<root>/<split>/images/<name>.jpg).
from-png writes image_sha256 when --images is given.

RLE: [[value, length], ...], row-major from the top-left pixel.
mask_hash: SHA-256 of str(shape) followed by the uint8 pixel bytes (channel 0 of a multi-channel PNG).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

LABEL_VALUES = {"0": "background", "1": "ventral right", "2": "dorsal right", "3": "dorsal central",
                "4": "ventral left", "5": "dorsal left", "6": "caudate lobe"}


def encode(a):
    f = a.ravel()
    idx = np.flatnonzero(f[1:] != f[:-1]) + 1
    starts, ends = np.r_[0, idx], np.r_[idx, f.size]
    return [[int(f[s]), int(e - s)] for s, e in zip(starts, ends)]


def decode(rec):
    r = np.asarray(rec["rle"], dtype=np.int64)
    assert r[:, 1].sum() == rec["height"] * rec["width"], rec["name"]
    return np.repeat(r[:, 0].astype(np.uint8), r[:, 1]).reshape(rec["height"], rec["width"])


def mask_hash(a):
    return hashlib.sha256(str(a.shape).encode() + a.astype(np.uint8).tobytes()).hexdigest()


def load_png(p):
    a = np.array(Image.open(p))
    return (a[..., 0] if a.ndim == 3 else a).astype(np.uint8)


def load_labels(path):
    """{name: record} in file order."""
    return {r["name"]: r for r in json.loads(Path(path).read_text(encoding="utf-8"))["images"]}


def counts(a):
    return {str(i): int(n) for i, n in enumerate(np.bincount(a.ravel(), minlength=7))}


def image_sha256(root, rec):
    return hashlib.sha256((Path(root) / rec["split"] / "images" / f"{rec['name']}.jpg").read_bytes()).hexdigest()


def write_labels(path, recs):
    """Same layout as the released files: header, then one image per line."""
    head = json.dumps({"version": "1.0", "label_values": LABEL_VALUES}, indent=1)[:-2]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(head + ',\n "images": [\n' + ",\n".join(json.dumps(r, separators=(",", ":")) for r in recs) + "\n ]\n}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["to-png", "from-png", "verify"])
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--png-dir")
    ap.add_argument("--images", help="L3D-2K root with <split>/images/<name>.jpg")
    a = ap.parse_args()
    if a.cmd == "to-png":
        for rec in load_labels(a.a).values():
            out = Path(a.b) / rec["split"] / f"{rec['name']}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(decode(rec), mode="L").save(out)
    elif a.cmd == "from-png":
        recs = []
        for p in sorted(Path(a.a).glob("*/*.png"), key=lambda p: (p.parent.name, p.stem)):
            x = load_png(p)
            r = {"name": p.stem, "split": p.parent.name, "patient": "_".join(p.stem.split("_")[:2]),
                 "height": x.shape[0], "width": x.shape[1]}
            if a.images:
                r["image_sha256"] = image_sha256(a.images, r)
            recs.append({**r, "rle": encode(x), "mask_hash": mask_hash(x), "counts": counts(x)})
        write_labels(a.b, recs)
    else:
        bad = []
        for rec in load_labels(a.a).values():
            x = decode(rec)
            if counts(x) != rec["counts"] or mask_hash(x) != rec["mask_hash"]:
                bad.append(rec["name"])
            elif a.png_dir and not np.array_equal(x, load_png(Path(a.png_dir) / rec["split"] / f"{rec['name']}.png")):
                bad.append(rec["name"])
            elif a.images and image_sha256(a.images, rec) != rec.get("image_sha256"):
                bad.append(rec["name"])
        print(f"{len(load_labels(a.a))} images, {len(bad)} mismatches {bad[:10]}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
