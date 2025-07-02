# from pandas.tests.frame.methods.test_replace import mix_ab
from datetime import datetime
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import torch.nn.functional as F
from conf import cfg, load_cfg_from_args
from prompt_ensemble import AnomalyCLIP_PromptLearner
from .methods import *
from src.models import network
from src.utils import IID_losses
from src.utils.utils import CrossEntropyLabelSmooth
import anomalyclip
from src.data.dataset import *
from src.utils.loss import FocalLoss, BinaryDiceLoss
import time

logger = logging.getLogger(__name__)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_global_source(cfg):
    """
    train the global source model
    input:
        cfg: config file
    output:
        None
    """
    start_time = datetime.now().strftime('%y%m%d%H%M')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"++++++++++++++++++++++++Training global source model++++++++++++++++++++++++")
    logger.info("Loading global source model...")
    anoclip_params = {"Prompt_length"                  : 12,
                      "learnabel_text_embedding_depth" : 9,
                      "learnabel_text_embedding_length": 4}
    model, _ = anomalyclip.load(cfg.ANOCLIP.BACKBONE, device=device,
                                    design_details=anoclip_params)
    model.eval()
    root = cfg.DIR.DATASET + "/MVTecAD/"
    preprocess, target_transform = get_transform(cfg)
    train_data = DefaultDataset(root=root, transform=preprocess, target_transform=target_transform, mode=cfg.SETTING.MODE)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), anoclip_params)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=cfg.OPTIM.LR, betas=(0.5, 0.999))
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    lamda = 4

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
        if (epoch + 1) % 1 == 0:
            log_str = f'Task: total; Iter:{epoch}/{cfg.TRAIN.GLOBAL_EPOCH}; loss ={np.mean(loss_list)}'
            logger.info(log_str)

        # save model
        if (epoch + 1) % 1 == 0:
            if cfg.TRAIN.IS_SAVE:
                save_path = f"{cfg.output_path}/../total/global_tp/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save({"prompt_learner": prompt_learner.state_dict()}, f"{start_time}_global_tp_epoch_{epoch:03d}.pt")



def train_local_source(cfg):
    """
    train the global source model
    input:
        cfg: config file
    output:
        None
    """
    start_time = datetime.now().strftime('%y%m%d%H%M')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"++++++++++++++++++++++++Training local source model++++++++++++++++++++++++")

    model = network.ResAnomaly(res_name=cfg.MODEL.LOCAL)
    model = model.to(device)
    model.train()
    root = cfg.DIR.DATASET + "/MVTecAD/"
    preprocess, target_transform = get_transform(cfg)
    train_data = AnomalyDataset(root=root, transform=preprocess, target_transform=target_transform,
                         dataset_name=cfg.SETTING.DATASET)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=cfg.OPTIM.LR, betas=(0.5, 0.999))
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    lamda = 4

    for epoch in range(cfg.TRAIN.LOCAL_EPOCH):
        model.train()
        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0


            probs, similarity_map = model(image)
            similarity_map = similarity_map / similarity_map.norm(dim=-1, keepdim=True)

            ####################################
            image_loss = F.cross_entropy(probs.squeeze(), label.long().cuda())
            image_loss_list.append(image_loss.item())
            #########################################################################
            similarity_map = F.interpolate(similarity_map, size=cfg.DATA.CROP_SIZE, mode='bilinear')


            # focal_loss = loss_focal(similarity_map, gt)
            focal_loss = 0
            dice_loss = loss_dice(similarity_map[:, 1, :, :], gt) + loss_dice(similarity_map[:, 0, :, :], 1 - gt)
            loss = lamda * (focal_loss + dice_loss) + image_loss
            print(f"{image_loss: .4f} {focal_loss: .4f} {dice_loss: .4f} {loss: .4f}")
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            loss_list.append(loss.item())
        # logs
        if (epoch + 1) % 1 == 0:
            log_str = f'Task: total; Iter:{epoch}/{cfg.TRAIN.LOCAL_EPOCH}; loss ={np.mean(loss_list)}'
            logger.info(log_str)

        # save model
        if (epoch + 1) % 1 == 0:
            if cfg.TRAIN.IS_SAVE:
                save_path = f"{cfg.output_path}/../total/local_backbone"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save({"prompt_learner": model.state_dict()}, f"{save_path}/{start_time}_local_backbone_epoch_{epoch:03d}.pt")



if __name__ == '__main__':
    cfg.type = cfg.domain

    # difo
    cfg.savename = cfg.TRAIN.METHOD

    start = time.time()

    torch.manual_seed(cfg.SETTING.SEED)
    torch.cuda.manual_seed(cfg.SETTING.SEED)
    np.random.seed(cfg.SETTING.SEED)
    random.seed(cfg.SETTING.SEED)

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    train_global_source(cfg)
