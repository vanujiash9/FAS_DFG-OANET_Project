# src/scripts/split_oanet_pt_fixed.py
import random
from pathlib import Path
import shutil

random.seed(42)

# ==== Thư mục nguồn ====
OA_LIVE_FOLDERS = [
    Path("data/processed/oanet_dataset/live/ffhq"),
    Path("data/processed/oanet_dataset/live/celeba")
]
OA_SPOOF_FOLDER = Path("data/processed/oanet_dataset/spoof_pt")

# ==== Thư mục đích ====
OUTPUT_ROOT = Path("data/splits_pt")
SPLITS = {"train":0.7, "val":0.15, "test":0.15}

# ==== Helpers ====
def make_dir(p):
    p.mkdir(parents=True, exist_ok=True)

def get_pt_files(folder):
    return list(folder.rglob("*.pt"))

def split_list(lst, ratios):
    n = len(lst)
    train_end = int(ratios["train"]*n)
    val_end = train_end + int(ratios["val"]*n)
    test_end = val_end + int(ratios["test"]*n)
    return lst[:train_end], lst[train_end:val_end], lst[val_end:test_end]

def copy_files(files, dst_dir):
    make_dir(dst_dir)
    for f in files:
        shutil.copy(f, dst_dir / f.name)

# ==== Remove old splits ====
if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)
make_dir(OUTPUT_ROOT)

# ==== Load file .pt ====
live_files = []
for folder in OA_LIVE_FOLDERS:
    live_files.extend(get_pt_files(folder))

spoof_files = get_pt_files(OA_SPOOF_FOLDER)

# ==== Cân bằng số lượng ====
min_len = min(len(live_files), len(spoof_files))
live_files = live_files[:min_len]
spoof_files = spoof_files[:min_len]

random.shuffle(live_files)
random.shuffle(spoof_files)

live_train, live_val, live_test = split_list(live_files, SPLITS)
spoof_train, spoof_val, spoof_test = split_list(spoof_files, SPLITS)

# ==== Copy file ====
for split_name, live_split, spoof_split in zip(
    ["train","val","test"],
    [live_train, live_val, live_test],
    [spoof_train, spoof_val, spoof_test]
):
    copy_files(live_split, OUTPUT_ROOT / "oan" / split_name / "live")
    copy_files(spoof_split, OUTPUT_ROOT / "oan" / split_name / "spoof")

print(f"[DONE] OA-Net .pt split -> {OUTPUT_ROOT/'oan'}")
print(f"Train: {len(live_train)} live, {len(spoof_train)} spoof")
print(f"Val: {len(live_val)} live, {len(spoof_val)} spoof")
print(f"Test: {len(live_test)} live, {len(spoof_test)} spoof")
