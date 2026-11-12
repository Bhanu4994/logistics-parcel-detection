import os, shutil, glob, collections, sys

SRC = sys.argv[1]
DST = sys.argv[2]
NAMES = ["Box", "Parcel"]

def to_bbox(vals):
    if len(vals) == 4:
        x, y, w, h = vals
    else:
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    clip = lambda v: max(0.0, min(1.0, v))
    return clip(x), clip(y), clip(w), clip(h)

stats = collections.Counter()
for split in ["train", "valid", "test"]:
    os.makedirs(f"{DST}/{split}/images", exist_ok=True)
    os.makedirs(f"{DST}/{split}/labels", exist_ok=True)
    for lf in glob.glob(f"{SRC}/{split}/labels/*.txt"):
        out = []
        for line in open(lf).read().strip().splitlines():
            p = line.split()
            if len(p) < 5: continue
            c = int(p[0]); x, y, w, h = to_bbox(list(map(float, p[1:])))
            if w < 0.005 or h < 0.005:
                stats["dropped_tiny"] += 1; continue
            out.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            stats[f"{split}_{NAMES[c]}"] += 1
        name = os.path.basename(lf)
        open(f"{DST}/{split}/labels/{name}", "w").write("\n".join(out))
        img = name[:-4] + ".jpg"
        shutil.copy(f"{SRC}/{split}/images/{img}", f"{DST}/{split}/images/{img}")
        stats[f"{split}_images"] += 1

abs_dst = os.path.abspath(DST)
open(f"{DST}/data.yaml", "w").write(
    f"path: {abs_dst}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
    f"nc: {len(NAMES)}\nnames: {NAMES}\n")
for k in sorted(stats): print(k, stats[k])
