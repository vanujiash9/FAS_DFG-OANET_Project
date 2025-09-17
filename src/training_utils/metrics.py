import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_classification_metrics(y_true, y_pred_binary, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "f1_score": f1_score(y_true, y_pred_binary, zero_division=0),
    }
    
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics["auc"] = auc(fpr, tpr)
    else:
        metrics["auc"] = np.nan

    return metrics

def compute_fixed_threshold_anti_spoofing_metrics(y_true, y_score, threshold=0.5):
    y_pred_binary = (np.array(y_score) >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total_live = tn + fp
    apcer = fp / total_live if total_live > 0 else 0.0

    total_spoof = tp + fn
    bpcer = fn / total_spoof if total_spoof > 0 else 0.0

    acer = (apcer + bpcer) / 2.0

    return {
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "threshold": threshold
    }

def compute_eer_and_related_metrics(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    eer_threshold = None
    eer = 1.0
    
    if len(thresholds) > 1:
        try:
            interp_fpr = interp1d(thresholds, fpr)
            interp_fnr = interp1d(thresholds, fnr)
            
            a = thresholds[-1]
            b = thresholds[0]
            
            if (interp_fpr(a) - interp_fnr(a)) * (interp_fpr(b) - interp_fnr(b)) <= 0:
                eer_threshold = brentq(lambda x: interp_fpr(x) - interp_fnr(x), a, b)
                eer = interp_fpr(eer_threshold)
            else:
                min_diff_idx = np.argmin(np.abs(fpr - fnr))
                eer = (fpr[min_diff_idx] + fnr[min_diff_idx]) / 2
                eer_threshold = thresholds[min_diff_idx]

        except ValueError:
            min_diff_idx = np.argmin(np.abs(fpr - fnr))
            eer = (fpr[min_diff_idx] + fnr[min_diff_idx]) / 2
            eer_threshold = thresholds[min_diff_idx]
            
    else:
        eer = (fpr[0] + fnr[0]) / 2
        eer_threshold = thresholds[0]

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "apcer_at_eer": eer,
        "bpcer_at_eer": eer,
        "acer_at_eer": eer,
        "hter_at_eer": eer
    }

def calculate_psnr_ssim(img1_np, img2_np, data_range=1.0):
    psnr = peak_signal_noise_ratio(img1_np, img2_np, data_range=data_range)
    ssim = structural_similarity(img1_np, img2_np, data_range=data_range, channel_axis=2)
    return psnr, ssim