import torch
import matplotlib.pyplot as plt
import os

cue_path = "data/processed/anomalous_cues/celeba_spoof_spoof/cue_525955.pt"
cue = torch.load(cue_path)

# Nếu là (3, H, W), chuyển về (H, W, 3)
if len(cue.shape) == 3 and cue.shape[0] == 3:
    # Sử dụng imshow với ảnh màu (H, W, 3 chuẩn)
    cue_np = cue.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow((cue_np - cue_np.min()) / (cue_np.max() - cue_np.min()))
elif len(cue.shape) == 2:
    # Ảnh xám
    cue_np = cue.cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(cue_np, cmap='jet')
else:
    # Nếu batch, squeeze
    cue_np = cue.squeeze().cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(cue_np, cmap='jet')

plt.axis('off')
out_path = os.path.splitext(cue_path)[0] + '_viz.png'
plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
print(f"Đã lưu ảnh trực quan hóa sang: {out_path}")
