import torch.nn as nn

def dfg_loss_fn():
    """Hàm loss cho DFG (Mean Squared Error)."""
    return nn.MSELoss()

def oanet_loss_fn():
    """Hàm loss cho OA-Net (Binary Cross Entropy with Logits)."""
    # SỬA LỖI: Thêm từ khóa 'return' ở đây
    return nn.BCEWithLogitsLoss()