from sklearn.metrics import auc, roc_curve, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
import numpy as np
from skimage import measure

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc

def cal_pro_score_gpu(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = torch.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in torch.arange(min_th, max_th, delta, device='cuda'):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask.cpu().numpy())):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = torch.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(torch.tensor(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = torch.tensor(pros), torch.tensor(fprs), torch.tensor(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    # pro_auc = auc(fprs, pros[idxes])
    pro_auc =torch.trapz(pros[idxes], fprs).item()
    return pro_auc

def image_level_metrics(results, obj, metric):
    gt = results[obj]['gt_sp']# 真正的标签
    pr = results[obj]['pr_sp']# 模型输出概率
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'image-auroc':
        try:
            performance = roc_auc_score(gt, pr)
        except:
            print("Warning: ROC AUC score calculation failed. Setting performance to 0.")
            performance = 0
    elif metric == 'image-ap':
        try:
            performance = average_precision_score(gt, pr)
        except:
            print("Warning: Average precision score calculation failed. Setting performance to 0.")
            performance = 0

    # 最佳阈值
    try:
        fpr, tpr, thresholds = roc_curve(gt, pr)
        j_scores = tpr - fpr
        j_ordered = sorted(zip(j_scores, thresholds))
        best_threshold = j_ordered[-1][1]
    except:
        print("Warning: ROC curve calculation failed. Setting best threshold to 0.")
        best_threshold = 0
    return performance, best_threshold
    # table.append(str(np.round(performance * 100, decimals=1)))

from torchmetrics.classification import BinaryAUROC, BinaryPrecisionRecallCurve
import torch
def pixel_level_metrics(results, obj, metric):
    gt = results[obj]['imgs_masks']
    pr = results[obj]['anomaly_maps']
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'pixel-auroc':
        try:
            performance = roc_auc_score(gt.ravel(), pr.ravel())
        except:
            print("Warning: ROC AUC score calculation failed. Setting performance to 0.")
            performance = 0
    elif metric == 'pixel-aupro':
        if len(gt.shape) == 4:
            gt = gt.squeeze(1)
        if len(pr.shape) == 4:
            pr = pr.squeeze(1)
        try:
            performance = cal_pro_score(gt, pr)
        except:
            print("Warning: Pro score calculation failed. Setting performance to 0.")
            performance = 0
    return performance
    