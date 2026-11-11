"""Pack L3D-2K images into images.zip for the annotation tool.

    python pack_images.py L3D-2K/                      # all splits
    python pack_images.py L3D-2K/ --split test         # one split
    python pack_images.py L3D-2K/ --names list.txt     # only the image names listed (one per line, without .jpg)

The zip holds images/<name>.jpg and manifest.json (the list of names). It is written next to this script.
"""
import argparse
import json
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="L3D-2K root with <split>/images/<name>.jpg")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--names")
    a = ap.parse_args()
    splits = [a.split] if a.split else ["train", "val", "test"]
    files = {p.name: p for s in splits for p in sorted((Path(a.root) / s / "images").glob("*.jpg"))}
    if a.names:
        want = {l.strip() + ".jpg" for l in Path(a.names).read_text(encoding="utf-8").splitlines() if l.strip()}
        missing = sorted(want - set(files))
        if missing:
            print(f"not found: {len(missing)} (e.g. {missing[:3]})")
        files = {n: p for n, p in files.items() if n in want}
    out = Path(__file__).resolve().parent / "images.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:      # JPEG is already compressed
        z.writestr("manifest.json", json.dumps(sorted(files)))
        for n in sorted(files):
            z.write(files[n], arcname=f"images/{n}")
    print(f"packed {len(files)} images -> {out} ({out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
