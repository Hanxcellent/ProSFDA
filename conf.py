"""Configuration file (powered by YACS)."""

import argparse
import logging
import os
from datetime import datetime

import torch
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode

# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C
# ---------------------------------- Directory options --------------------------- #
_C.DIR = CfgNode()
# Dataset directory
_C.DIR.DATASET = "/remote-home/iot_hanxiang/dataset/"
_C.DIR.MODELS = "/remote-home/iot_hanxiang/models/"
# Pretrained weight directory
_C.DIR.CKPT = "./ckpts/"
# Output directory
_C.DIR.OUTPUT = "./output/"
# # Path to a specific checkpoint
# ---------------------------------- Misc options --------------------------- #
# Setting - see README.md for more information
_C.GPU_ID = '0'
_C.NUM_WORKERS = 4
# ----------------------------- Base model options ------------------------------- #
_C.MODEL = CfgNode()
_C.MODEL.LOCAL = "resnet50"  # _C.MODEL.ARCH = 'resnet50'
_C.MODEL.GLOBAL = "anomalyclip"
_C.MODEL.WEIGHTS = 'IMAGENET1K_V1'
# ----------------------------- AnoCLIP options ------------------------------- #
_C.ANOCLIP = CfgNode()
_C.ANOCLIP.BACKBONE = "ViT-L/14@336px"
_C.ANOCLIP.FEATURE_LIST = [6, 12, 18, 24]
_C.ANOCLIP.FEATURE_MAP_LAYER =[0, 1, 2, 3]
_C.ANOCLIP.N_CTX = 12
_C.ANOCLIP.DEPTH = 9
_C.ANOCLIP.T_N_CTX =4
# ----------------------------- Visual prompt options -------------------------- #
_C.VP = CfgNode()
_C.VP.PAD_WIDTH = 20
_C.VP.ALPHA = 0.3
_C.VP.GTYPE = "overlay"
_C.VP.LTYPE = "adapter"
# ----------------------------- Setting options -------------------------- #
_C.SETTING = CfgNode()
_C.SETTING.MODE = "train" # (train, transfer, test)
if _C.SETTING.MODE == "transfer":
    _C.SETTING.SOURCE = 0
    _C.SETTING.TARGET = 1
_C.SETTING.DATASET = 'mypaper-large-only' # (zjuleaper, mvtecad, pcb)
_C.SETTING.SEED = 2021
# ----------------------------- Data options -------------------------- #
_C.DATA = CfgNode()
_C.DATA.RESIZE_SIZE = 336
_C.DATA.CROP_SIZE = 336
_C.DATA.GEN_JSON = True
# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()
_C.OPTIM.METHOD = "SGD"
_C.OPTIM.LR = 2e-3
_C.OPTIM.MOMENTUM = 0.9
# Nesterov momentum
_C.OPTIM.NESTEROV = True
# L2 regularization
_C.OPTIM.WD = 5e-4
_C.OPTIM.LR_DECAY1 = 0.1
_C.OPTIM.LR_DECAY2 = 1
_C.OPTIM.LR_DECAY3 = 0.01
# ------------------------------- Train options ------------------------- #
_C.TRAIN = CfgNode()
# Train step
_C.TRAIN.METHOD = "difo"
_C.TRAIN.CKPT_PATH = ""
_C.TRAIN.STEP = "global-step" # (global-step, local-step, two-step)
# Batch size
_C.TRAIN.BATCH_SIZE = 128
# Max global epoch
_C.TRAIN.GLOBAL_EPOCH = 10
# Max local epoch
_C.TRAIN.LOCAL_EPOCH = 10
# Communication epoch
_C.TRAIN.COMMUNICATION_EPOCH = 1
# Num of validate interval
_C.TRAIN.REC_INTERVAL = 10
# Save model
_C.TRAIN.IS_SAVE = True
# ------------------------------- Test options ------------------------- #
_C.TEST = CfgNode()
_C.TEST.BATCH_SIZE = 1
_C.TEST.MODEL = 'local'
_C.TEST.CKPT_PATH = "./output/mypaper/TotalTrain/gsource/2506091713_global_tp_epoch_099.pt"
_C.TEST.METRIC = "image-pixel-level"
_C.TEST.IS_SAVE = True
# ------------------------------- Transfer options ---------- #
_C.TRANSFER = CfgNode()
_C.TRANSFER.METHOD = "difo"
_C.TRANSFER.STEP = "global-step" # (global-step, local-step, two-step)
_C.TRANSFER.BATCH_SIZE = 128
_C.TRANSFER.GLOBAL_EPOCH = 10
_C.TRANSFER.LOCAL_EPOCH = 10
_C.TRANSFER.COMMUNICATION_EPOCH = 1
_C.TRANSFER.NUM_INTERVAL = 10
_C.TRANSFER.IS_SAVE = True
# ------------------------------- Logging options ------------------------- #
_C.LOG_TIME = ""
_C.LOG_DEST = ""
# ------------------------------- Other options ---------- #
_C.OTHER = CfgNode()
_C.OTHER.EVAL_METRIC = "acc"
_C.OTHER.EVAL_MODE = "val"
_C.OTHER.EVAL_FREQ = 1
_C.OTHER.EVAL_CKPT = "best"
_C.OTHER.EVAL_CKPT_METRIC = "acc"
# --------------------------------- SOURCE options ---------------------------- #
_C.SOURCE = CfgNode()
_C.SOURCE.EPSILON = 1e-5
_C.SOURCE.TRTE = 'val'
# --------------------------------- DIFO options ----------------------------- #
_C.DIFO = CfgNode()
_C.DIFO.EPSILON = 1e-5
_C.DIFO.CTX_INIT = 'a_photo_of_a' #initialize context
_C.DIFO.N_CTX = 4
_C.DIFO.TTA_STEPS = 1
# _C.DIFO.LOAD = None
_C.DIFO.THRESHOLD = 0.8
# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()
_C.CUDNN.BENCHMARK = True
# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def merge_from_file(cfg_file):
    """Loads a yaml config file and merges it into the global config."""
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)

def load_cfg_from_args():
    """Load config from command line args and set any specified options."""
    cfg.LOG_TIME = datetime.now().strftime("%y%m%d%H%M%S")
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--cfg", dest="cfg_file",default="cfgs/office-home/difo.yaml", type=str,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    args = parser.parse_args()

    # Load config from file
    merge_from_file(args.cfg_file)
    # Merge config from command line
    cfg.merge_from_list(args.opts)

    cfg.TRAIN.METHOD = os.path.basename(args.cfg_file)[:-5]
    cfg.LOG_DEST = 'log.txt'

    cfg.bottleneck = 512

    if cfg.SETTING.MODE == 'transfer':
        cfg.name_src = cfg.domain[cfg.SETTING.SOURCE].capitalize()
        cfg.name_tar = cfg.domain[cfg.SETTING.TARGET].capitalize()
        cfg.name = cfg.name_src + cfg.name_tar
        cfg.pretrained_model_path = f"{cfg.DIR.CKPT}/{cfg.SETTING.DATASET}/{cfg.name_src}"
    elif cfg.SETTING.MODE == 'train':
        cfg.name = "TotalTrain"
    else:
        cfg.name = "TotalTest"

    cfg.output_path = f"{cfg.DIR.OUTPUT}/{cfg.SETTING.DATASET}/{cfg.name}/{cfg.TRAIN.METHOD}/{cfg.LOG_TIME}"

    g_pathmgr.mkdirs(cfg.output_path)

    log_dir = f"{cfg.output_path}"
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(f"{log_dir}/{cfg.LOG_DEST}"),
            logging.StreamHandler()
        ])

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda
               ]
    logger.info("Loading config...")
    logger.info("PyTorch Version: torch={}, cuda={}".format(*version))
    logger.info(cfg)


def complete_data_dir_path(root, dataset_name):
    # map dataset name to data directory name
    mapping = {"imagenet": "imagenet2012",
               "imagenet_c": "ImageNet-C",
               "imagenet_r": "imagenet-r",
               "imagenet_k": os.path.join("ImageNet-Sketch", "sketch"),
               "imagenet_a": "imagenet-a",
               "imagenet_d": "imagenet-d",      # do not change
               "imagenet_d109": "imagenet-d",   # do not change
               "domainnet126": "DomainNet-126", # directory containing the 6 splits of "cleaned versions" from http://ai.bu.edu/M3SDA/#dataset
               "office31": "office-31",
               "visda": "visda-2017",
               "cifar10": "",  # do not change the following values
               "cifar10_c": "",
               "cifar100": "",
               "cifar100_c": "",
               "imagenet_v": "imagenetv2-matched-frequency-format-val"
               }
    return os.path.join(root, mapping[dataset_name])


# def get_domain_sequence(ckpt_path):
#     assert ckpt_path.endswith('.pth') or ckpt_path.endswith('.pt')
#     domain = ckpt_path.replace('.pth', '').split(os.sep)[-1].split('_')[1]
#     mapping = {"real": ["clipart", "painting", "sketch"],
#                "clipart": ["sketch", "real", "painting"],
#                "painting": ["real", "sketch", "clipart"],
#                "sketch": ["painting", "clipart", "real"],
#                }
#     return mapping[domain]
#
#
# def adaptation_method_lookup(adaptation):
#     lookup_table = {"source": "Norm",
#                     "norm_test": "Norm",
#                     "norm_alpha": "Norm",
#                     "norm_ema": "Norm",
#                     "ttaug": "TTAug",
#                     "memo": "MEMO",
#                     "lame": "LAME",
#                     "tent": "Tent",
#                     "eata": "EATA",
#                     "sar": "SAR",
#                     "adacontrast": "AdaContrast",
#                     "cotta": "CoTTA",
#                     "rotta": "RoTTA",
#                     "gtta": "GTTA",
#                     "rmt": "RMT",
#                     "roid": "ROID",
#                     "proib": "Proib"
#                     }
#     assert adaptation in lookup_table.keys(), \
#         f"Adaptation method '{adaptation}' is not supported! Choose from: {list(lookup_table.keys())}"
#     return lookup_table[adaptation]
