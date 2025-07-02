# from pandas.tests.frame.methods.test_replace import mix_ab
# from pandas.tests.frame.methods.test_replace import mix_ab
from datetime import datetime

from scipy.ndimage import gaussian_filter
from tabulate import tabulate
from torch.utils.data.distributed import DistributedSampler

import anomalyclip
from myutils.metrics import image_level_metrics, pixel_level_metrics
from myutils.visualization import visualizer
from prompt_ensemble import AnomalyCLIP_PromptLearner
from src.data.dataset import *
from src.models import network
from src.utils.loss import FocalLoss, BinaryDiceLoss
from .methods import *

logger = logging.getLogger(__name__)




def train_global_source(cfg):
    """
    train the global source model
    input:
        cfg: config file
    output:
        None
    """
    start_time = cfg.LOG_TIME
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"++++++++++++++++++++++++Training global source model++++++++++++++++++++++++")
    logger.info("Loading global source model...")
    anoclip_params = {"Prompt_length"                  : 12,
                      "learnabel_text_embedding_depth" : 9,
                      "learnabel_text_embedding_length": 4}
    model, _ = anomalyclip.load(cfg.ANOCLIP.BACKBONE, device=device,
                                    design_details=anoclip_params)
    model.eval()
    root = os.path.join(cfg.DIR.DATASET, cfg.SETTING.DATASET)
    preprocess, target_transform = get_transform(cfg)
    train_data = AnomalyDataset(root=root, transform=preprocess, target_transform=target_transform, dataset_name=cfg.SETTING.DATASET, mode="train", json_name="meta_all.json")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), anoclip_params)
    if cfg.TRAIN.CKPT_PATH is not "":
        logger.info(f"Loading global source model from {cfg.TRAIN.CKPT_PATH}...")
        checkpoint = torch.load(cfg.TRAIN.CKPT_PATH, map_location=device)
        prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=cfg.OPTIM.LR, betas=(0.5, 0.999))
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    lamda = 8

    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(cfg.TRAIN.GLOBAL_EPOCH)):
        model.eval()
        prompt_learner.train()
        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                # Apply DPAM to the layer from 6 to 24
                # DPAM_layer represents the number of layer refined by DPAM from top to bottom
                # DPAM_layer = 1, no DPAM is used
                # DPAM_layer = 20 as default
                image_features, patch_features = model.encode_image(image, cfg.ANOCLIP.FEATURE_LIST, DPAM_layer=20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            ####################################
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Apply DPAM surgery
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...] / 0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().cuda())
            image_loss_list.append(image_loss.item())
            #########################################################################
            similarity_map_list = []
            # similarity_map_list.append(similarity_map)
            for idx, patch_feature in enumerate(patch_features):
                if idx >= cfg.ANOCLIP.FEATURE_MAP_LAYER[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = anomalyclip.compute_similarity(patch_feature, text_features[0])
                    similarity_map = anomalyclip.get_similarity_map(similarity[:, 1:, :], cfg.DATA.CROP_SIZE).permute(
                        0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1 - gt)

            loss = lamda * loss
            optimizer.zero_grad()
            (loss + image_loss).backward()
            optimizer.step()
            loss_list.append(loss.item())
        # logs
        if (epoch + 1) % cfg.TRAIN.REC_INTERVAL == 0:
            log_str = f'Task: total; Iter:{epoch}/{cfg.TRAIN.GLOBAL_EPOCH}; loss ={np.mean(loss_list)}'
            logger.info(log_str)

        # save model
        if (epoch + 1) % cfg.TRAIN.REC_INTERVAL == 0:
            if cfg.TRAIN.IS_SAVE:
                torch.save({"prompt_learner": prompt_learner.state_dict()},
                           f"{cfg.output_path}/global_tp_epoch_{epoch+1:03d}.pt")
                torch.save({"prompt_learner": prompt_learner.state_dict()},
                           f"{cfg.output_path}/latest.pt")

def test_global_source(cfg):
    """
    train the global source model
    input:
        cfg: config file
    output:
        None
    """
    start_time = cfg.LOG_TIME
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"++++++++++++++++++++++++Testing global source model++++++++++++++++++++++++")
    logger.info("Loading global source model...")
    anoclip_params = {"Prompt_length"                  : 12,
                      "learnabel_text_embedding_depth" : 9,
                      "learnabel_text_embedding_length": 4}
    model, _ = anomalyclip.load(cfg.ANOCLIP.BACKBONE, device=device,
                                    design_details=anoclip_params)
    model.eval()

    root = os.path.join(cfg.DIR.DATASET, cfg.SETTING.DATASET)
    preprocess, target_transform = get_transform(cfg)
    test_data = AnomalyDataset(root=root, transform=preprocess, target_transform=target_transform, dataset_name=cfg.SETTING.DATASET, mode="test", json_name="meta_all.json")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {}
    metrics = {}
    img_names = []
    anomaly_map_save = []
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), anoclip_params)
    checkpoint = torch.load(cfg.TEST.CKPT_PATH)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    if cfg.ANOCLIP.BACKBONE == "ViT-B/16":
        dpam_layer=10
    elif cfg.ANOCLIP.BACKBONE == "ViT-L/14@336px":
        dpam_layer=20
    model.visual.DAPM_replace(DPAM_layer = dpam_layer)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)  # [1,2,768]
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    model.to(device)
    for idx, items in enumerate(tqdm(test_dataloader)):  # tqdm是进度条
        # print("items:",items['anomaly'],items['anomaly'].item())
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())  # 0代表正常，1代表异常

        # print("items['anomaly']",items['anomaly'])

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, cfg.ANOCLIP.FEATURE_LIST, DPAM_layer=dpam_layer)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [1,768]

            text_probs = image_features @ text_features.permute(0, 2, 1)  # permute后为[1, 768, 2].后两个维度进行矩阵乘法
            text_probs = (text_probs / 0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]  # 属于异常的概率
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= cfg.ANOCLIP.FEATURE_MAP_LAYER[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)  # [1, 1297, 768]
                    similarity, _ = anomalyclip.compute_similarity(patch_feature, text_features[0])  # 和正常特征对比
                    similarity_map = anomalyclip.get_similarity_map(similarity[:, 1:, :], cfg.DATA.RESIZE_SIZE)
                    anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0  # 减去相似性得到异常分数
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)

            anomaly_map = anomaly_map.sum(dim=0)
            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack(
                    [torch.from_numpy(gaussian_filter(i, sigma=4)) for i in anomaly_map.detach().cpu()], dim=0)
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            # if is_saved==True:# 可视化+保存
            #     visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name)
            file_name = items['img_path'][0].split('/')[-1]
            img_names.append(file_name)
            anomaly_map_save.append(anomaly_map.detach().cpu().numpy())

    # 计算检测指标
    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    for obj in obj_list:
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if cfg.TEST.METRIC == 'image-level':
            image_auroc, best_threshold = image_level_metrics(results, obj, "image-auroc")
            image_ap, best_threshold = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        elif cfg.TEST.METRIC == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif cfg.TEST.METRIC == 'image-pixel-level':
            image_auroc, best_threshold = image_level_metrics(results, obj, "image-auroc")
            image_ap, best_threshold = image_level_metrics(results, obj, "image-ap")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        table_ls.append(table)

    # 保存异常热力图
    best_threshold = best_threshold - 0.00001
    logger.info("\nbest_threshold:%f", best_threshold)
    if cfg.TEST.IS_SAVE:  # 可视化+保存
        for idx, items in enumerate(test_dataloader):
            cls_name = items['cls_name']
            anomaly_map_ = anomaly_map_save[idx]
            if results[cls_name[0]]['pr_sp'][idx] > best_threshold:  # 异常
                visualizer(items['img_path'], anomaly_map_, cfg.DATA.RESIZE_SIZE, cfg.output_path, cls_name, mask_path=items['mask_path'])
            else:  # 正常
                visualizer(items['img_path'], anomaly_map_, cfg.DATA.RESIZE_SIZE, cfg.output_path, cls_name, is_anomaly=False)

    # 最后计算检测结果的时候用
    for obj in obj_list:
        anomal_pro_array = np.array(results[obj]['pr_sp'])
        anomal_gt_array = np.array(results[obj]['gt_sp'])

    # 打印检测指标
    if cfg.TEST.METRIC == 'image-level':
        # logger
        table_ls.append(['mean',
                         str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                         str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    elif cfg.TEST.METRIC == 'pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                         str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                         ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    elif cfg.TEST.METRIC == 'image-pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                         str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)),
                         str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                         str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'],
                           tablefmt="pipe")
    logger.info("\n%s", results)

    # 检测结果，image-level
    all_indices = np.arange(len(anomal_pro_array))
    anomaly_indices = np.where(anomal_pro_array > best_threshold)[0]
    normal_indices = np.setdiff1d(all_indices, anomaly_indices)
    false_detect_ano_to_nor = anomaly_indices[np.where(anomal_gt_array[anomaly_indices] == 0)[0]]
    false_detect_nor_to_ano = normal_indices[np.where(anomal_gt_array[normal_indices] == 1)[0]]
    img_names = np.array(img_names)
    logger.info("anomaly imgs:%s", img_names[anomaly_indices])
    logger.info("normal imgs:%s", img_names[normal_indices])
    if false_detect_nor_to_ano.size == 0:
        logger.info("not false detect ano to nor")
    else:
        logger.info("false detect ano to nor:%s", img_names[false_detect_ano_to_nor])
    if false_detect_nor_to_ano.size == 0:
        logger.info("not false detect nor to ano")
    else:
        logger.info("false detect nor to ano:%s", img_names[false_detect_nor_to_ano])
    logger.info(f"Everything saved at {cfg.output_path}")