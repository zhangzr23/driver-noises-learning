import os
import re
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from data.ContrastiveCrop import ContrastiveCrop
from torchvision.transforms import Compose
from utils import CCompose
import numpy as np
import random


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(args, model_idx=None):
    g = torch.Generator()
    g.manual_seed(42)
    train_set, train_loader, eval_train_loader, train_sampler = None, None, None, None

    if args.auc_eval:
        val_set = ImageDataset(istrain=False, root=args.val_root, re_size=args.re_size, crop_size=args.crop_size)
        val_loader = DataLoader(val_set, num_workers=args.num_workers, shuffle=False, batch_size=args.batch_size,
                                pin_memory=True, drop_last=False)
        return val_loader

    if args.one_hundred_eval:
        val_set = one_hundred_dataset(istrain=False, root=args.data_root, re_size=args.re_size, crop_size=args.crop_size)
        val_loader = DataLoader(val_set, num_workers=args.num_workers, shuffle=False, batch_size=args.batch_size,
                                pin_memory=True, drop_last=False)
        return val_loader

    if args.multi_gpu:
        train_set = one_hundred_dataset(istrain=True, root=args.data_root, re_size=args.re_size, crop_size=args.crop_size)
        val_set = one_hundred_dataset(istrain=False, root=args.data_root, re_size=args.re_size, crop_size=args.crop_size)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

        train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size,
                                  drop_last=True, sampler=train_sampler, generator=g,
                                  worker_init_fn=seed_worker, collate_fn=train_set.collate_fn)
        val_loader = DataLoader(val_set, num_workers=args.num_workers, batch_size=args.batch_size,
                                drop_last=False, sampler=val_sampler, generator=g,
                                worker_init_fn=seed_worker, collate_fn=val_set.collate_fn)

        return train_loader, train_set, val_loader, eval_train_loader, train_sampler

    if args.update_bbox:
        train_set = ImageFolderCCrop(trans_ccrop=transform_ccrop(args),
                                     trans_rcrop=transform_rcrop(args), root=args.train_root)
        train_loader = DataLoader(train_set, num_workers=args.num_workers, shuffle=True,
                                  batch_size=args.batch_size, pin_memory=True, drop_last=True)

        eval_train_transform = transforms.Compose([transforms.Resize((args.crop_size, args.crop_size)),
                                                   transforms.ToTensor(),
                                                   normalize,
                                                   ])
        eval_train_set = datasets.ImageFolder(root=args.train_root, transform=eval_train_transform)
        eval_train_loader = DataLoader(eval_train_set, num_workers=args.num_workers, shuffle=False,
                                       batch_size=args.batch_size, pin_memory=True, drop_last=False)
    else:
        train_set = ImageDataset(istrain=True, root=args.train_root, re_size=args.re_size, crop_size=args.crop_size)
        train_loader = DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size,
                                  pin_memory=True, drop_last=True)

    val_set = ImageDataset(istrain=False, root=args.val_root, re_size=args.re_size, crop_size=args.crop_size)
    val_loader = DataLoader(val_set, num_workers=args.num_workers, shuffle=False, batch_size=args.batch_size,
                            pin_memory=True, drop_last=False)

    return train_loader, train_set, val_loader, eval_train_loader, train_sampler


def get_dataset(args):
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, root=args.train_root, re_size=args.re_size, crop_size=args.crop_size,
                                 return_index=True)
        return train_set
    return None


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 istrain: bool,
                 root: str,
                 re_size: int,
                 return_index: bool = False):
        self.return_index = return_index
        self.istrain = istrain

        if istrain:
            self.transforms = transforms.Compose([
                transforms.Resize((re_size, re_size), Image.BILINEAR),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((re_size, re_size), Image.BILINEAR),
                transforms.ToTensor(),
                normalize
            ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)

    def getDataInfo(self, root):
        data_infos = []
        if isinstance(root, str):
            folders = os.listdir(root)
            folders.sort()
            print("[dataset] class number:", len(folders))
            for class_id, folder in enumerate(folders):
                files = os.listdir(os.path.join(root, folder))
                files.sort(key=lambda l: int(re.findall('\d+', l)[0]))
                for file in files:
                    data_path = os.path.join(root, folder, file)
                    data_infos.append({"path": data_path, "label": class_id})

            return data_infos

        elif isinstance(root, list):
            folders = os.listdir(root[0])
            folders.sort()
            print("[dataset] class number:", len(folders))
            for sub_root in root:
                for class_id, folder in enumerate(folders):
                    for file in os.listdir(os.path.join(sub_root, folder)):
                        data_path = os.path.join(sub_root, folder, file)
                        data_infos.append({"path": data_path, "label": class_id})
            return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1]  # BGR to RGB.

        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, label

        # return img, sub_imgs, label, sub_boundarys
        return img, label, index

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        img_1, target1 = tuple(zip(*batch))

        img_1 = torch.stack(img_1, dim=0)
        target1 = torch.as_tensor(target1)

        return img_1, target1


class one_hundred_dataset(torch.utils.data.Dataset):
    def __init__(self,
                 istrain: bool,
                 root: str,
                 re_size: int,
                 return_index: bool = False,
                 init_box=(0., 0., 1., 1.)):
        self.return_index = return_index
        self.istrain = istrain

        if istrain:
            self.transforms = transforms.Compose([
                transforms.Resize((re_size, re_size), Image.BILINEAR),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((re_size, re_size), Image.BILINEAR),
                transforms.ToTensor(),
                normalize
            ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)

    def getDataInfo(self, root):
        modality_dict = {'Day': 'D', 'Night': 'N'}
        status = 'train' if self.istrain else 'test'
        data_infos = []

        modality = root.split('/')[-1]
        for cam_name in os.listdir(root):
            cam_id = cam_name.replace('Cam', '')
            cam_path = os.path.join(root, cam_name)
            txt_name = modality_dict[modality] + str(cam_id) + '_' + status + '.txt'
            txt_path = os.path.join(root.replace(modality, ''), 'Traditional-setting', modality, cam_name, txt_name)
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    img_path = os.path.join(cam_path, line.split('\t')[1])
                    label = int(line.split('\t')[-1])
                    data_infos.append({"path": img_path, "label": label})

        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1]  # BGR to RGB.

        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, label

        # return img, sub_imgs, label, sub_boundarys
        return img, label, index

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        img_1, target1, index = tuple(zip(*batch))

        img_1 = torch.stack(img_1, dim=0)
        target1 = torch.as_tensor(target1)

        return img_1, target1, index


def transform_ccrop(args):
    trans_list = [
        ContrastiveCrop(alpha=0.6, size=args.crop_size, scale=(0.2, 1.0)),
        transforms.ToTensor(),
        normalize
    ]
    transform = CCompose(trans_list)
    return transform


def transform_rcrop(args):
    trans_list = [
        transforms.RandomResizedCrop(size=args.crop_size, scale=(0.2, 1.0)),
        transforms.ToTensor(),
        normalize
    ]
    transform = transforms.Compose(trans_list)
    return transform


class ImageFolderCCrop(datasets.ImageFolder):
    def __init__(self, root, trans_ccrop, trans_rcrop, init_box=(0., 0., 1., 1.), **kwargs):
        super().__init__(root=root, **kwargs)

        # self.boxes = torch.tensor([0., 0., 1., 1.]).repeat(self.__len__(), 1)
        self.boxes = torch.tensor(init_box).repeat(self.__len__(), 1)
        self.transform_ccrop = trans_ccrop
        self.transform_rcrop = trans_rcrop
        self.use_box = False

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.use_box:
            box = self.boxes[index].float().tolist()  # box=[h_min, w_min, h_max, w_max]
            sample = self.transform_ccrop([sample, box])
        else:
            sample = self.transform_rcrop(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
