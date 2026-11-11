# Annotation tool

A static web page for drawing zone labels on L3D-2K images. It runs in the browser and needs no GPU, no model and no Python packages. This is the tool used to draw the released labels.

## Start

1. Get L3D-2K from its distribution.
2. Pack the images you want to label (Python 3, standard library only):
   ```
   python pack_images.py <L3D-2K root>                 # all 2,000 images
   python pack_images.py <L3D-2K root> --split test    # or one split
   ```
   This writes `images.zip` next to `index.html`.
3. Serve the folder and open it in Chrome or Edge: double-click `serve.bat` (Windows) or run `sh serve.sh`. Both start `python -m http.server 8000 --bind 127.0.0.1` (reachable from this computer only) and open `http://localhost:8000`.
4. Click **Choose mask folder** and pick a folder for the masks. Each image is saved there as `<name>.png` when you press Ctrl+S.
   - Writing to a folder needs Chrome or Edge.
   - In other browsers, masks are downloaded instead.
   - If you open `index.html` directly, without the server, you select `images.zip` by hand.

Masks are 8-bit PNGs with pixel value = class:

| Value | Class |
|---|---|
| 0 | background |
| 1 | ventral right |
| 2 | dorsal right |
| 3 | dorsal central |
| 4 | ventral left |
| 5 | dorsal left |
| 6 | caudate lobe |

To convert a folder of masks to the JSON format, place them as `<folder>/<split>/<name>.png` and run `python ../../code/rle_tools.py from-png <folder> my_labels.json`.

## Controls

| Action | Input |
|---|---|
| Polygon | click to add a vertex, drag to trace, right-click to undo a vertex |
| Close polygon | double-click or Enter (preview), Enter again to commit |
| Brush | left button paints, right button erases; `[` `]` change the size |
| Class | keys 0–6 |
| Clear | Shift+C current class, Ctrl+Shift+C everything |
| View | wheel zooms, Space + drag pans, F fits |
| Save | Ctrl+S |

**Edge snap** (button, on by default): polygon vertices snap to colour edges found in the image (Sobel), within the snap radius. It helps on visible borders, such as the liver outline. It does not help on borders between surface orientations, such as ventral versus dorsal, which have no colour edge.

## Third-party file

`jszip.min.js` is JSZip 3.10.1 (MIT). See `../THIRD_PARTY_NOTICES.md`.
