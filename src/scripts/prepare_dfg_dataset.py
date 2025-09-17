# src/scripts/prepare_dfg_dataset.py
import torch
from pathlib import Path

def check_pt_files(pt_dir: Path):
    """
    Lọc các file .pt hợp lệ: dict có key 'img' hoặc tensor.
    Trả về danh sách Path.
    """
    valid_files = []
    for pt_file in pt_dir.rglob("*.pt"):
        try:
            data = torch.load(pt_file)
            if isinstance(data, dict) and 'img' in data:
                valid_files.append(pt_file)
            elif isinstance(data, torch.Tensor):
                valid_files.append(pt_file)
            else:
                print(f"[WARN] Bỏ qua file không hợp lệ: {pt_file}")
        except Exception as e:
            print(f"[ERROR] Không load được {pt_file}: {e}")
    return valid_files

def main():
    dfg_train_dir = Path("data/splits_pt/dfg/train")
    dfg_val_dir = Path("data/splits_pt/dfg/val")

    valid_train = check_pt_files(dfg_train_dir)
    valid_val = check_pt_files(dfg_val_dir)

    print(f"[INFO] Train hợp lệ: {len(valid_train)} files")
    print(f"[INFO] Val hợp lệ: {len(valid_val)} files")

    # Lưu lại danh sách nếu muốn dùng cho DataLoader
    torch.save(valid_train, "data/splits_pt/dfg/train_files.pt")
    torch.save(valid_val, "data/splits_pt/dfg/val_files.pt")

if __name__ == "__main__":
    main()
