import torch
import torch.utils.data as data
import os
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from anomalyclip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from anomalyclip.transform import image_transform


def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtecad':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    else:
        obj_list = [dataset_name]

    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class AnomalyDataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test', json_name='meta.json'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(os.path.join(self.root, json_name), 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(img_path)
        if anomaly and os.path.isfile(mask_path):
            img_mask = np.array(Image.open(mask_path).convert('L')) > 0
            img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        else:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')

        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': img_path,
                 "cls_id": self.class_name_map_class_id[cls_name], 'mask_path':mask_path}
def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def get_transform(cfg):
    preprocess = image_transform(cfg.DATA.RESIZE_SIZE, is_train=False, mean = OPENAI_DATASET_MEAN, std = OPENAI_DATASET_STD)
    target_transform = transforms.Compose([
        transforms.Resize((cfg.DATA.RESIZE_SIZE, cfg.DATA.RESIZE_SIZE)),
        transforms.ToTensor()
    ])
    preprocess.transforms[0] = transforms.Resize(size=(cfg.DATA.RESIZE_SIZE, cfg.DATA.RESIZE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC,
                                                    max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE))
    return preprocess, target_transform

class ZLDataset(data.Dataset):
    def __init__(self, root, transform, target_transform, json_path="./ImageSets/Groups/group1.json", mode='train'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(os.path.join(self.root, json_path), 'r'))
        self.data_all = []
        for name in meta_info["normal"][mode]:
            self.data_all.append({"name": name, "label": 0})
        for name in meta_info["defect"][mode]:
            self.data_all.append({"name": name, "label": 1})

        self.length = len(self.data_all)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample = self.data_all[index]
        img_path = os.path.join(self.root, "Images", sample["name"]+".jpg")
        mask_path = os.path.join(self.root, "Annotations", "masks", sample["name"]+".png")
        label = sample["label"]
        img = Image.open(img_path)
        if label == 0:
            mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(mask_path):
                # just for classification not report error
                mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                mask = np.array(Image.open(mask_path).convert('L')) > 0
                mask = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
        # transforms
        if self.transform:
            img = self.transform(img)
        if self.target_transform and mask:
            mask = self.target_transform(mask)
        if not mask:
            mask = []

        return {'img': img, 'mask': mask, "label": label, 'img_path': img_path}

class DefaultDataset(data.Dataset):
    def __init__(self, root, transform, target_transform, json_path=None, mode='train'):
        if json_path is None:
            json_path = os.path.join(root, "meta.json")
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(json_path, 'r'))
        meta_info = meta_info[mode]
        self.data_all.extend(meta_info)
        self.length = len(self.data_all)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path = data["img_path"]
        mask_path = data["mask_path"]
        anomaly = data["anomaly"]
        img = Image.open(img_path)
        if anomaly == 0:
            mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(mask_path):
                # just for classification not report error
                mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                mask = np.array(Image.open(mask_path).convert('L')) > 0
                mask = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
        # transforms
        if self.transform:
            img = self.transform(img)
        if self.target_transform and mask:
            mask = self.target_transform(mask)
        if mask is None:
            mask = []

        return {'img': img, 'img_mask': mask, 'cls_name': None, 'anomaly': anomaly,
                'img_path': img_path, "cls_id": 0}

if __name__ == '__main__':
    preprocess, target_transform = get_transform2()
    train_data = DefaultDataset(root="/remote-home/iot_hanxiang/ProSFDA/datasets/mypaper", transform=preprocess, target_transform=target_transform,
                                mode="train")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=12, shuffle=True)
    for item in train_dataloader:
        print(item)