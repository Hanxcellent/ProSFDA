import cv2
import os
from myutils.utils import normalize
import numpy as np

def visualizer(pathes, anomaly_map, img_size, save_path, cls_name, is_anomaly=True, mask_path=None):
    for idx, path in enumerate(pathes):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        if is_anomaly:
            mask = normalize(anomaly_map[idx]) # 归一化 
        else:
            mask = np.zeros_like(anomaly_map[idx])
        vis = apply_ad_scoremap(vis, mask) # 得到异常分数热力图
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        save_vis = os.path.join(save_path, 'imgs')
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(os.path.join(save_vis, filename), vis)

        # 变换后的输入
        save_vis= os.path.join(save_vis, 'sample')
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        vis = cv2.resize(cv2.imread(path), (img_size, img_size))
        cv2.imwrite(os.path.join(save_vis,  filename), vis)    
        #变换后的mask     
        if mask_path is not None and mask_path != ['']:
            save_vis= os.path.join(save_vis, 'mask')
            if not os.path.exists(save_vis):
                os.makedirs(save_vis)
            vis = cv2.resize(cv2.imread(mask_path[0]), (img_size, img_size))
            cv2.imwrite(os.path.join(save_vis,  filename), vis)   

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)# 将得分图转换为彩色热力图。靠近1暖色调，靠近0冷
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8) # 热力图和原图按比例合成一张图
