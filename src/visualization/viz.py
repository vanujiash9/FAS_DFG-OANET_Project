import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg # Cập nhật import

def plot_data_distribution(dfg_live_count, oa_net_live_count, oa_net_spoof_count, spoof_types_distribution=None):
    labels = ['dfg live (ffhq + celeba)', 'oanet live (celeba)', 'oanet spoof (celeba)']
    counts = [dfg_live_count, oa_net_live_count, oa_net_spoof_count]

    plt.figure(figsize=(12, 7))
    plt.bar(labels, counts, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel('số lượng ảnh')
    plt.title('phân bố dữ liệu tổng thể')
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.oanet.VISUALIZATIONS_DIR, "data_distribution.png"))
    plt.show()

    if spoof_types_distribution:
        spoof_labels = list(spoof_types_distribution.keys())
        spoof_counts = list(spoof_types_distribution.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(spoof_labels, spoof_counts, color='purple')
        plt.ylabel('số lượng ảnh')
        plt.title('phân bố các loại tấn công giả mạo (spoof types)')
        for i, count in enumerate(spoof_counts):
            plt.text(i, count, str(count), ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.oanet.VISUALIZATIONS_DIR, "spoof_type_distribution.png"))
        plt.show()

def plot_split_distribution(train_len, val_len, test_len):
    labels = ['train', 'validation', 'test']
    sizes = [train_len, val_len, test_len]
    colors = ['#ff9999','#66b3ff','#99ff99']
    explode = (0.05, 0.05, 0.05)

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title('phân chia tập huấn luyện/kiểm định/kiểm tra cho oanet')
    plt.savefig(os.path.join(cfg.oanet.VISUALIZATIONS_DIR, "train_val_test_split.png"))
    plt.show()

def plot_dfg_training_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, label='dfg training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (mse)')
    plt.title('dfg training loss over epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cfg.dfg.VISUALIZATIONS_DIR, "dfg_training_loss.png"))
    plt.show()

def save_dfg_cue_visualization(original_images, reconstructed_images, cues, prefix_filename, sample_idx_in_batch, num_samples=1):
    os.makedirs(cfg.dfg.DFG_CUES_VIZ_DIR, exist_ok=True)
    
    for i in range(min(num_samples, original_images.shape[0])):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_images[i].cpu().permute(1, 2, 0).numpy())
        axes[0].set_title("original image")
        axes[0].axis('off')

        axes[1].imshow(reconstructed_images[i].cpu().permute(1, 2, 0).numpy())
        axes[1].set_title("reconstructed (dfg output)")
        axes[1].axis('off')

        cue_img = cues[i].mean(dim=0).cpu().numpy()
        axes[2].imshow(cue_img, cmap='hot')
        axes[2].set_title("anomalous cue")
        axes[2].axis('off')
        
        plt.tight_layout()
        # Ensure unique naming: prefix_filename can contain epoch/batch info
        save_path = os.path.join(cfg.dfg.DFG_CUES_VIZ_DIR, f"{prefix_filename}_sample_{sample_idx_in_batch:03d}.png")
        plt.savefig(save_path)
        plt.close(fig)

def plot_oanet_train_val_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='train loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (bcewithlogits)')
    plt.title('oanet training and validation loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cfg.oanet.VISUALIZATIONS_DIR, "oanet_train_val_loss.png"))
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=['live', 'spoof'], save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('confusion matrix')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve(y_true, y_prob, save_path=None):
    if len(np.unique(y_true)) < 2:
        print("roc curve cannot be plotted for a single class.")
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'roc curve (auc = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('receiver operating characteristic (roc) curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.show()