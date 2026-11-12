# Laparoscopic-Liver-Surface-Segmentation

Liver surface zones defined for computer vision: labels for semantic segmentation of the L3D-2K laparoscopic liver surgery dataset.

Code and annotation tool for the liver surface zone labels of the L3D-2K laparoscopic liver dataset.

The labels split the visible liver surface into six zones:

1. ventral right
2. dorsal right
3. dorsal central
4. ventral left
5. dorsal left
6. caudate lobe

They cover all 2,000 L3D-2K images, and a separate file holds a redraw of 100 images. This repository holds:

- `code/`: scripts that reproduce the paper's tables from the label files and L3D-2K
- `tool/`: the browser annotation tool used to draw the labels

## Data

- **Labels:** download the label files (`zone_labels.json`, `zone_labels_redraw100.json`, CC BY-NC 4.0) from Zenodo: https://doi.org/10.5281/zenodo.24125640.
- **Images and lines:** get the L3D-2K images and official lines from the L3D-2K distribution, which is linked from the README of https://github.com/cuiruize/D2GPLand-Plus. The scripts expect `<L3D-2K root>/<split>/images/<name>.jpg` and `<L3D-2K root>/<split>/labels/<name>.json`.

## Code

```
pip install -r code/requirements.txt          # Python 3.10
cd code
```

| Script | What it does | Paper |
|---|---|---|
| `rle_tools.py` | JSON <-> PNG label maps; checks counts, `mask_hash` and, with `--images`, `image_sha256` | — |
| `analysis.py` | label statistics, agreement of zone boundaries with the official lines, intra-rater repeatability | Tables 1, 4, 5, 6 |
| `baseline.py` | trains and evaluates the four baseline models | Tables 3, 7 |

```
python rle_tools.py to-png zone_labels.json pngs/          # PNG label maps (pngs/<split>/<name>.png)
python rle_tools.py verify zone_labels.json --images <L3D-2K root>

python analysis.py --labels zone_labels.json --redraw zone_labels_redraw100.json --l3d2k <L3D-2K root> --out results/

for m in dino segf dlv3p unet; do for s in 0 1 2; do
  python baseline.py train --labels zone_labels.json --l3d2k <L3D-2K root> --model $m --seed $s --out runs/
done; done
python baseline.py evaluate --labels zone_labels.json --l3d2k <L3D-2K root> --runs runs/ --split test --out test.json
```

`evaluate` looks for `runs/<prefix><model>_s<seed>/best.pt`; `--prefix` (empty by default) is for run folders named with a prefix.

### Models

| Name | Model | Encoder weights |
|---|---|---|
| `dino` | DINOv3 ViT-S/16 with a pyramid decoder. The last 8 blocks are trained | `facebook/dinov3-vits16-pretrain-lvd1689m`. Gated on Hugging Face: accept the licence and log in first |
| `segf` | SegFormer, MiT-B2 encoder | ImageNet |
| `dlv3p` | DeepLabV3+, ResNet-50 encoder | ImageNet |
| `unet` | U-Net, ResNet-34 encoder | ImageNet |

### Settings

**Input and labels**

- Input: 512×896 RGB, letterboxed. The padding is excluded from the loss.
- Output: background and zones 1–5. Pixels of the caudate lobe (6) are excluded from the loss and from scoring.

**Training**

- Augmentation: affine, colour jitter and Gaussian blur. No horizontal flip, because it would make the right–left arrangement of the zones inconsistent with the anatomy.
- Loss: 0.5 cross-entropy + 0.5 soft Dice over zones 1–5.
- Optimiser: AdamW (weight decay 1e-4). The learning rate is 3e-4 for the decoder and head and 3e-5 for the encoder.
- Schedule: cosine over 50 epochs, batch 4, mixed precision, gradient clipping at 1.0.
- Checkpoint: the epoch with the highest validation macro IoU is kept.

**Scoring**

- Scoring is at the original resolution. The padding is removed, the logits are bilinear-resized to the original size, and then argmax is taken.
- Zone IoU is averaged within each image, then within each patient, then across patients.
- 95% CIs come from a patient bootstrap with 10,000 draws.
- GPU training is not bit-for-bit reproducible. In the paper, the standard deviation of test macro IoU over three seeds was at most 0.020.

## Annotation tool

`tool/web_anno_static/` is a static web page. It needs no GPU and no Python packages:

1. Pack images with `pack_images.py`.
2. Serve the folder with `serve.bat` or `serve.sh`.
3. Draw polygons or brush strokes. Masks are saved as PNG files (pixel value = class).

Polygon vertices can snap to colour edges (Sobel); this can be switched on and off. See `tool/web_anno_static/README.md`.

## Licence

- Code and tool: MIT (`LICENSE`).
- `tool/web_anno_static/jszip.min.js`: JSZip, MIT (`tool/THIRD_PARTY_NOTICES.md`).
- Labels (on Zenodo): CC BY-NC 4.0.
- The L3D-2K images and lines are not covered by these licences.

## Citation

Kasai M. Liver surface zone labels for laparoscopic liver surgery images. (manuscript in preparation)

Please also cite L3D-2K: Cui R, Si W, Li Z, Wang K, Pei J, Heng PA, Qin J. Depth-induced prompt learning for laparoscopic liver landmark detection. Med Image Anal 2026;109:103940. doi:10.1016/j.media.2026.103940

## Topics

computer-vision, semantic-segmentation, liver-segmentation, laparoscopy, surgical-data-science, medical-image-segmentation, dataset, l3d-2k
