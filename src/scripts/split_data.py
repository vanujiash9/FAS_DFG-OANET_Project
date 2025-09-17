# src/scripts/split_data_updated.py
import os
import json
import random
from pathlib import Path
import shutil

# ================= CONFIG =================
OANET_ROOT = Path("data/processed/oanet_dataset")
DFG_ROOT = Path("data/processed/dfg")
OUTPUT_ROOT = Path("data/splits")
SPLITS = {"train": 0.7, "val": 0.3, "test": 0.15}  # DFG: chỉ train/val
random.seed(42)

# ================= HELPER =================
def make_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def get_all_files(folder, exts=[".jpg", ".png", ".pt"]):
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]

def split_list(lst, ratios):
    n = len(lst)
    train_end = int(ratios["train"] * n)
    val_end = train_end + int(ratios.get("val",0) * n)
    test_end = val_end + int(ratios.get("test",0) * n)
    return lst[:train_end], lst[train_end:val_end], lst[val_end:test_end]

# ================= REMOVE OLD SPLITS =================
for f in ["dfg_splits.json", "oan_splits.json"]:
    old_file = OUTPUT_ROOT / f
    if old_file.exists():
        old_file.unlink()
        print(f"[INFO] Deleted old split file: {old_file}")

make_dir(OUTPUT_ROOT)

# ================= SPLIT DFG =================
print("[INFO] Splitting DFG dataset (ffhq + celeba, train/val only)")
dfg_files = []
for folder in ["ffhq", "celeba"]:
    folder_path = DFG_ROOT / folder
    dfg_files.extend(get_all_files(folder_path))

random.shuffle(dfg_files)
train_files, val_files, _ = split_list(dfg_files, {"train": 0.7, "val": 0.3, "test":0})
dfg_splits = {
    "train": [str(f) for f in train_files],
    "val": [str(f) for f in val_files]
}

# ================= SPLIT OA-NET =================
print("[INFO] Splitting OA-Net dataset (train/val/test, balance live/spoof)")
live_files = []
for folder in ["ffhq", "celeba"]:
    live_files.extend(get_all_files(OANET_ROOT / "live" / folder))

spoof_files = []
for spoof_type_dir in (OANET_ROOT / "spoof").iterdir():
    if spoof_type_dir.is_dir():
        spoof_files.extend(get_all_files(spoof_type_dir))

random.shuffle(live_files)
random.shuffle(spoof_files)

# Cân bằng số lượng
min_len = min(len(live_files), len(spoof_files))
live_files = live_files[:min_len]
spoof_files = spoof_files[:min_len]

live_train, live_val, live_test = split_list(live_files, SPLITS)
spoof_train, spoof_val, spoof_test = split_list(spoof_files, SPLITS)

oan_splits = {
    "train": {"live": [str(f) for f in live_train], "spoof": [str(f) for f in spoof_train]},
    "val": {"live": [str(f) for f in live_val], "spoof": [str(f) for f in spoof_val]},
    "test": {"live": [str(f) for f in live_test], "spoof": [str(f) for f in spoof_test]},
}

# ================= SAVE JSON =================
with open(OUTPUT_ROOT / "dfg_splits.json", "w") as f:
    json.dump(dfg_splits, f, indent=2)
with open(OUTPUT_ROOT / "oan_splits.json", "w") as f:
    json.dump(oan_splits, f, indent=2)

print(f"[DONE] DFG splits saved: {OUTPUT_ROOT / 'dfg_splits.json'}")
print(f"[DONE] OA-Net splits saved: {OUTPUT_ROOT / 'oan_splits.json'}")
