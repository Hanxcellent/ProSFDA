import os
import random
import time
import logging
import torch
import numpy as np
import torch
import src.methods.mypaper.source as mypaper
from src.data.generate_dataset_json import PaperSolver
# import src.methods.oh.source as SOURCE
# import src.methods.oh.test as TEST
from conf import cfg, load_cfg_from_args

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    load_cfg_from_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    # cfg.type = cfg.domain
    if cfg.DATA.GEN_JSON or \
            not os.path.exists(os.path.join(cfg.DIR.DATASET, cfg.SETTING.DATASET, "meta_all.json")):
        runner =  PaperSolver(root=os.path.join(cfg.DIR.DATASET, cfg.SETTING.DATASET))
        runner.run()
    assert os.path.exists(os.path.join(cfg.DIR.DATASET, cfg.SETTING.DATASET, "meta_all.json"))

    start = time.time()
    torch.manual_seed(cfg.SETTING.SEED)
    torch.cuda.manual_seed(cfg.SETTING.SEED)
    np.random.seed(cfg.SETTING.SEED)
    random.seed(cfg.SETTING.SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    if cfg.SETTING.MODE == "train":
        if cfg.TRAIN.STEP == "global-step":
            logger.info("Start training the global model")
            mypaper.train_global_source(cfg)
        else:
            logger.info("Start training the local model")
            logger.info("CODE NOT IMPLEMENTED YET\n"*3)
            pass
    elif cfg.SETTING.MODE == "test":
        if cfg.TEST.MODEL == "global":
            logger.info("Start testing the global model")
            mypaper.test_global_source(cfg)
        elif cfg.TEST.MODEL == "local":
            logger.info("Start testing the local model")
            logger.info("CODE NOT IMPLEMENTED YET\n"*3)
            pass
    else:
        logger.info("CODE NOT IMPLEMENTED YET\n"*3)
    logger.info(f"Everything finished in {round(time.time()-start, 2)} seconds")


