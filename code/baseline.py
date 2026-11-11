"""Baseline zone-recognition models: train and evaluate.

    python baseline.py train    --labels zone_labels.json --l3d2k L3D-2K/ --model dino --seed 0 --out runs/
    python baseline.py evaluate --labels zone_labels.json --l3d2k L3D-2K/ --runs runs/ --split test --out test.json

Models: dino (DINOv3 ViT-S/16, last 8 blocks trained, pyramid decoder), segf (SegFormer MiT-B2),
dlv3p (DeepLabV3+ ResNet-50), unet (U-Net ResNet-34). Output: background + zones 1-5; caudate (6) is ignored.
Scoring at the original resolution: zone IoU per image -> image macro IoU -> patient mean -> mean over patients.
"""
import argparse
import json
import random
import warnings
from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.amp import GradScaler, autocast

from rle_tools import decode, load_labels

IGNORE, NCLS, SIZE, ZONES = 255, 6, (512, 896), (1, 2, 3, 4, 5)
MODELS, SEEDS, N_BOOT = ("dino", "segf", "dlv3p", "unet"), (0, 1, 2), 10_000
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
SMP = {"segf": (smp.Segformer, "mit_b2"), "dlv3p": (smp.DeepLabV3Plus, "resnet50"), "unet": (smp.Unet, "resnet34")}
patient = lambda s: "_".join(s.split("_")[:2])  # noqa: E731


# ---------------------------------------------------------------- models
def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.GELU())


class DinoEncoder(nn.Module):
    def __init__(self, unfreeze=8):
        super().__init__()
        from transformers import AutoModel
        self.net = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        self.taps = (2, 5, 8, 11)          # 4 of 12 blocks, shallow -> deep
        for p in self.net.parameters():
            p.requires_grad_(False)
        for m in self.net.model.layer[-unfreeze:]:   # the final LayerNorm stays frozen: the taps are taken before it
            for p in m.parameters():
                p.requires_grad_(True)

    def trainable_parameters(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    def forward(self, x):
        gh, gw = x.shape[2] // 16, x.shape[3] // 16
        hs = self.net(pixel_values=x, output_hidden_states=True).hidden_states
        return [hs[t + 1][:, -gh * gw:].transpose(1, 2).reshape(x.shape[0], -1, gh, gw) for t in self.taps]


class PyramidDecoder(nn.Module):
    def __init__(self, cin=384, dim=256, out=NCLS):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Sequential(nn.Conv2d(cin, dim, 1, bias=False), nn.GroupNorm(8, dim), nn.GELU()) for _ in range(4)])
        self.fuse = nn.ModuleList([block(dim * 2, dim) for _ in range(3)])
        self.head = nn.Sequential(block(dim, 128), nn.Conv2d(128, out, 1))

    def forward(self, grids, size):
        up = lambda t, **k: F.interpolate(t, mode="bilinear", align_corners=False, **k)  # noqa: E731
        lv = [up(self.lateral[i](g.float()), scale_factor=u) if u > 1 else self.lateral[i](g.float())
              for i, (g, u) in enumerate(zip(grids, (8, 4, 2, 1)))]
        x = lv[3]
        for i, blk in enumerate(self.fuse):
            x = blk(torch.cat([up(x, size=lv[2 - i].shape[-2:]), lv[2 - i]], 1))
        x = up(x, size=(size[0] // 4, size[1] // 4))
        return up(self.head(x), size=size)


class Dino(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = DinoEncoder(), PyramidDecoder()

    def forward(self, x):
        return self.decoder(self.encoder(x), x.shape[-2:])


def build(name):
    if name == "dino":
        return Dino()
    cls, enc = SMP[name]
    return cls(encoder_name=enc, encoder_weights="imagenet", in_channels=3, classes=NCLS, activation=None)


def param_groups(model, name):
    enc = model.encoder.trainable_parameters() if name == "dino" else [p for p in model.encoder.parameters() if p.requires_grad]
    ids = {id(p) for p in enc}
    return [{"params": [p for p in model.parameters() if p.requires_grad and id(p) not in ids], "lr": 3e-4},
            {"params": enc, "lr": 3e-5}]


# ---------------------------------------------------------------- data
def letterbox(h, w):
    return [A.LongestMaxSize(max_size=max(h, w)), A.PadIfNeeded(min_height=h, min_width=w, border_mode=0, fill=0, fill_mask=IGNORE)]


def train_aug(h, w):
    return letterbox(h, w) + [A.RandomCrop(height=h, width=w),
                              A.Affine(translate_percent=(-0.05, 0.05), scale=(0.85, 1.15), rotate=(-12, 12), fill=0, fill_mask=IGNORE, p=0.5),
                              A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.4),
                              A.GaussianBlur(blur_limit=(3, 5), p=0.15)]


def eval_aug(h, w):
    return letterbox(h, w) + [A.CenterCrop(height=h, width=w)]


def normalize(img):
    return torch.from_numpy(np.ascontiguousarray(((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)))


def rgb(root, split, stem):
    return np.array(Image.open(Path(root) / split / "images" / f"{stem}.jpg").convert("RGB"))


def target(label):
    t = label.astype(np.int64)
    t[t == 6] = IGNORE
    return t


class ZoneData(torch.utils.data.Dataset):
    def __init__(self, recs, root, train, seed=None):
        self.recs, self.root = recs, root
        self.aug = A.Compose(train_aug(*SIZE) if train else eval_aug(*SIZE), seed=seed)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        o = self.aug(image=rgb(self.root, r["split"], r["name"]), mask=target(decode(r)))
        return {"image": normalize(o["image"]), "mask": torch.from_numpy(o["mask"].astype(np.int64)), "stem": r["name"]}


def seed_worker(_):
    base = torch.initial_seed() % 2 ** 31
    random.seed(base)
    np.random.seed(base % 2 ** 32)
    torch.manual_seed(base)
    torch.utils.data.get_worker_info().dataset.aug.set_random_seed(base)


def split_recs(labels, split):
    return sorted((r for r in labels.values() if r["split"] == split), key=lambda r: r["name"])


# ---------------------------------------------------------------- loss and metrics
def loss_fn(logits, t):
    """0.5 cross-entropy + 0.5 soft Dice over zones 1-5 (batch-pooled); ignored pixels excluded from both."""
    valid = t != IGNORE
    ce = F.cross_entropy(logits, t, ignore_index=IGNORE)
    probs = F.softmax(logits, dim=1)
    oh = F.one_hot(torch.where(valid, t, torch.zeros_like(t)), NCLS).permute(0, 3, 1, 2).to(probs.dtype)
    v = valid.unsqueeze(1).to(probs.dtype)
    p, g = probs[:, 1:] * v, oh[:, 1:] * v
    dice = 1.0 - ((2 * (p * g).sum((0, 2, 3)) + 1e-6) / (p.sum((0, 2, 3)) + g.sum((0, 2, 3)) + 1e-6)).mean()
    return 0.5 * ce + 0.5 * dice


def image_cms(pred, gt):
    """Per-image confusion matrices [B, 6, 6] (rows = label), ignored pixels excluded."""
    b = pred.shape[0]
    idx = gt.clamp(0, NCLS - 1) * NCLS + pred.clamp(0, NCLS - 1) + torch.arange(b, device=pred.device).view(b, 1, 1) * NCLS * NCLS
    return torch.bincount(idx[gt != IGNORE], minlength=b * NCLS * NCLS).view(b, NCLS, NCLS)


def summarize(cms, stems):
    """Zone IoU per image (NaN if the zone is in neither) -> patient means -> means over patients."""
    c = cms.astype(np.float64)
    tp, g, p = np.einsum("nii->ni", c)[:, 1:], c.sum(2)[:, 1:], c.sum(1)[:, 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(g + p - tp > 0, tp / (g + p - tp), np.nan)
    pats = sorted({patient(s) for s in stems})
    pid = np.array([pats.index(patient(s)) for s in stems])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        img = np.nanmean(iou, 1)
        pm = np.array([np.nanmean(img[pid == k]) for k in range(len(pats))])
        pz = np.stack([np.nanmean(iou[pid == k], 0) for k in range(len(pats))])
        zone = np.nanmean(pz, 0)
    return {"macro_iou": float(np.nanmean(pm)), "patient_macro_iou": dict(zip(pats, pm.tolist())),
            "patient_zone_iou": {q: dict(zip(map(str, ZONES), z.tolist())) for q, z in zip(pats, pz)},
            "zone_iou": dict(zip(map(str, ZONES), zone.tolist()))}


def boot_ci(vals, rng):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    m = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# ---------------------------------------------------------------- commands
def train(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)  # noqa: E702
    out = Path(a.out) / f"{a.model}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=False)
    labels = load_labels(a.labels)
    ds_tr, ds_va = ZoneData(split_recs(labels, "train"), a.l3d2k, True, a.seed), ZoneData(split_recs(labels, "val"), a.l3d2k, False)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    gen = torch.Generator()
    gen.manual_seed(a.seed)
    kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=4, shuffle=True, drop_last=True, generator=gen, worker_init_fn=seed_worker, **kw)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=4, **kw)
    model = build(a.model).to(dev)
    opt = torch.optim.AdamW(param_groups(model, a.model), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    amp = dev.type == "cuda"
    scaler, best, hist = GradScaler(enabled=amp), -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        losses = []
        for step, b in enumerate(dl_tr):
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp):
                loss = loss_fn(model(b["image"].to(dev)), b["mask"].to(dev))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
            if a.max_iters and step + 1 >= a.max_iters:     # quick check only
                torch.save({"losses": losses, "model": model.state_dict()}, out / "check.pt")
                return
        sched.step()
        model.eval()
        cms, stems = [], []
        with torch.no_grad():
            for b in dl_va:
                with autocast(device_type="cuda", enabled=amp):
                    pred = model(b["image"].to(dev)).argmax(1)
                cms.append(image_cms(pred, b["mask"].to(dev)).cpu())
                stems += list(b["stem"])
        val = summarize(torch.cat(cms).numpy(), stems)["macro_iou"]
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_macro_iou": val})
        print(hist[-1], flush=True)
        ck = {"arch": a.model, "seed": a.seed, "epoch": ep, "val_macro_iou": val, "model": model.state_dict()}
        if val > best:
            best = val
            torch.save(ck, out / "best.pt")
        (out / "history.json").write_text(json.dumps(hist, indent=1))


@torch.no_grad()
def predict(model, img, dev):
    h, w = img.shape[:2]
    x = normalize(A.Compose(eval_aug(*SIZE))(image=img)["image"])[None].to(dev)
    with autocast(device_type="cuda", enabled=dev.type == "cuda"):
        logits = model(x)[0]
    m = A.Compose(eval_aug(*SIZE))(image=np.zeros((h, w, 3), np.uint8), mask=np.ones((h, w), np.uint8))["mask"] == 1
    ys, xs = np.nonzero(m)
    crop = logits[:, ys.min():ys.max() + 1, xs.min():xs.max() + 1].float()   # drop the letterbox padding
    return F.interpolate(crop[None], size=(h, w), mode="bilinear", align_corners=False)[0].argmax(0).cpu().numpy()


def evaluate(a):
    recs = split_recs(load_labels(a.labels), a.split)
    stems = [r["name"] for r in recs]
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    runs = {}
    for m in MODELS:
        for s in a.seeds:
            name = f"{a.prefix}{m}_s{s}"
            model = build(m)
            model.load_state_dict(torch.load(Path(a.runs) / name / "best.pt", map_location="cpu", weights_only=False)["model"])
            model = model.to(dev).eval()
            preds = (predict(model, rgb(a.l3d2k, a.split, r["name"]), dev) for r in recs)
            cms = np.stack([image_cms(torch.from_numpy(p.astype(np.int64))[None], torch.from_numpy(target(decode(r)))[None])[0].numpy()
                            for p, r in zip(preds, recs)])
            runs[name] = summarize(cms, stems)
            np.save(Path(a.out).with_name(f"{Path(a.out).stem}_{name}_cm.npy"), cms)
            print(name, round(runs[name]["macro_iou"], 4), flush=True)
    agg = {}
    for mi, m in enumerate(MODELS):
        rs = [runs[f"{a.prefix}{m}_s{s}"] for s in a.seeds]
        rng = np.random.default_rng(100 * mi)   # seeds and call order match those used for the reported values
        seedavg = lambda ds: {q: float(np.mean([d[q] for d in ds if q in d and np.isfinite(d[q])] or [np.nan]))  # noqa: E731
                              for q in sorted(set().union(*ds))}
        pm = seedavg([r["patient_macro_iou"] for r in rs])
        vd = seedavg([{q: np.nanmean([z[k] for k in "14"]) - np.nanmean([z[k] for k in "235"])
                       for q, z in r["patient_zone_iou"].items()} for r in rs])
        v = np.array([r["macro_iou"] for r in rs])
        agg[m] = {"macro_iou": float(np.nanmean(list(pm.values()))), "ci95": boot_ci(list(pm.values()), rng),
                  "seed_sd": float(v.std(ddof=1)) if len(v) > 1 else None, "per_seed": v.tolist()}
        agg[m]["ventral_minus_dorsal"] = {"value": float(np.nanmean(list(vd.values()))), "ci95": boot_ci(list(vd.values()), rng)}
        agg[m]["zone_iou"] = {}
        for z in map(str, ZONES):
            pz = seedavg([{q: d[z] for q, d in r["patient_zone_iou"].items()} for r in rs])
            agg[m]["zone_iou"][z] = {"value": float(np.nanmean(list(pz.values()))), "ci95": boot_ci(list(pz.values()), rng)}
    Path(a.out).write_text(json.dumps({"split": a.split, "runs": runs, "aggregate": agg}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "evaluate"])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--l3d2k", required=True, help="L3D-2K root with <split>/images/<name>.jpg")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="evaluate: seeds to score")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-iters", type=int, default=0, help="stop after N training steps (code check)")
    ap.add_argument("--runs", help="folder with <prefix><model>_s<seed>/best.pt")
    ap.add_argument("--prefix", default="", help="evaluate: run folders are named <prefix><model>_s<seed>")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
