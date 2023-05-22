import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import os


def config_dataset(config):
#     if "cifar" in config["dataset"]:
#         config["topK"] = -1
#         config["n_class"] = 10
#     elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
#         config["topK"] = 5000
#         config["n_class"] = 21
#     elif config["dataset"] == "nuswide_81_m":
#         config["topK"] = 5000
#         config["n_class"] = 81
#     elif config["dataset"] == "coco":
#         config["topK"] = 5000
#         config["n_class"] = 80
#     elif config["dataset"] == "imagenet":
#         config["topK"] = 1000
#         config["n_class"] = 100
#     elif config["dataset"] == "mirflickr":
#         config["Ktop"] = -1
#         config["n_class"] = 38
#     elif config["dataset"] == "voc2012":
#         config["topK"] = -1
#         config["n_class"] = 20

    config["data_path"] = "/dataset/" + config["dataset"] + "/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "/hdd/sxz/gan/NUS-WIDE/"

    if config["dataset"] == "corel10k":
        config["data_path"] = "/home/sxz/code/data/Corel10k/"

    if config["dataset"] == "corel_en":
        config["data_path"] = "/hdd/sxz/0dataset/corel-en/"
        # config["data_path"] = "/root/code/corel-en/"
        config["topK"] = -1
        config["n_class"] = 100

    if config["dataset"] == "nuswide":
        config["data_path"] = "/hdd/sxz/gan/pytorch-CycleGAN-and-pix2pix-master/results/nuswide-e40_pretrained/test_latest/"
        config["topK"] = -1
        config["n_class"] = 21
    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "/dataset/nus_wide_m/"
    if config["dataset"] == "coco":
        config["data_path"] = "/dataset/COCO_2014/"
    if config["dataset"] == "voc2012":
        config["data_path"] = "/dataset/"
    config["data"] = {
        "train_set": {"list_path": config["data_path"] + "train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": config["data_path"] + "database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": config["data_path"] + "test.txt", "batch_size": config["batch_size"]}}
    return config


draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]

def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
    #  https://blog.csdn.net/HackerTom/article/details/89425729
    n_query = qF.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    Rank = np.argsort(CalcHammingDist(qF, rF))
    P, R = [], []
    for k in tqdm(draw_range):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all
        P.append(np.mean(p))
        R.append(np.mean(r))
    return P, R


class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        if 'result' in data_path:
            self.imgs = [(data_path + val.split()[0].replace('.jpg','_fake.png'), np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        # path1 = path.split('/')[-1]
        # path2 = path1.split('_')[0]
        # label = (int(path2)-1)//100
        # return img, target, index,label
        return img, target, index,target

    def __len__(self):
        return len(self.imgs)


class ImagePic(object):

    def __init__(self, data_path, data_set, transform):
        # self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform
        if 'train' in data_set:
            # root = os.path.join(root,'test3_pretrained/test_latest/images')
            # root = os.path.join(root,'1test30401_pretrained/test_latest/images')
            # root = os.path.join(root,'test30351_pretrained/test_latest/images')
            # root = '/home/sxz/code/gan/cifar-10-batches-py/train'
            # root = os.path.join(root,'cifar-1-e30_pretrained/test_latest/images')
            # root = os.path.join(root,'cifar-3-e40train_pretrained/test_latest/images')
            # root = os.path.join(root,'corel-e50train_pretrained/test_latest/images')
            # root = '/home/sxz/code/data/Corel10k/train'
            # root = '/home/sxz/code/gan/pytorch-CycleGAN-and-pix2pix-master/datasets/cifar10/trainA'
            root = '/hdd/sxz/C/gan/pytorch-CycleGAN-and-pix2pix-master/results/cifar-3-e40train_pretrained/test_latest/images'
            # root = "/hdd/sxz/C/gan/pytorch-CycleGAN-and-pix2pix-master/results/corel-e40train_pretrained/test_latest/images"
            # root = '/hdd/sxz/0dataset/cifar-10-batches-py/train'
            root = '/hdd/sxz/data/Corel10k/train'

        else:
            # root = os.path.join(root,'test31_pretrained/test_latest/images')
            # root = os.path.join(root,'test3040_pretrained/test_latest/images')
            # root = os.path.join(root,'test3035_pretrained/test_latest/images')
            # root = '/home/sxz/code/gan/cifar-10-batches-py/testA'
            # root = os.path.join(root,'cifar-1-e30test_pretrained/test_latest/images')
            # root = os.path.join(root,'corel-e50_pretrained/test_latest/images')
            # root = '/home/sxz/code/data/Corel10k/test'
            # root = '/home/sxz/code/gan/pytorch-CycleGAN-and-pix2pix-master/datasets/cifar10/testA'
            
            root = '/hdd/sxz/C/gan/pytorch-CycleGAN-and-pix2pix-master/results/cifar-3-e40_pretrained/test_latest/images'
            # root = '/hdd/sxz/C/gan/pytorch-CycleGAN-and-pix2pix-master/results/corel-e40_pretrained/test_latest/images'
            # root = '/hdd/sxz/0dataset/cifar-10-batches-py/test'
            root = '/hdd/sxz/data/Corel10k/test'
            
        imgs = []
        for path in os.listdir(root):
            # path_prefix = path[:1]
            path_prefix = path.split('.')[0]
            # label = int(path_prefix)
            label = (int(path_prefix)-1)//100
            imgs.append((os.path.join(root, path), label))
            
            # # ##############################cifar
            # path_prefix = path[:1]
            # path_prefix = path.split('_')[0]
            # label = int(path_prefix)
            # # label = int(path_prefix)
            # imgs.append((os.path.join(root, path), label))
        
        # train_path_list, val_path_list = self._split_data_set(imgs)
        self.imgs = imgs

    def __getitem__(self, index):
        # path, target = self.imgs[index]
        path = self.imgs[index][0]
        label = self.imgs[index][1]

        target = np.eye(100, dtype=np.int8)[label]
        # target = np.eye(10, dtype=np.int8)[label]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index,label

    def __len__(self):
        return len(self.imgs)


def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


def get_data(config):
    
    #     return cifar_dataset(config)

    dsets = {}
    dset_loaders = {}
    
    if "nuswide" in config["dataset"] :#Sor 'corel' in config["dataset"]:
        data_config = config["data"]
        for data_set in ["train_set", "test", "database"]:
            dsets[data_set] = ImageList(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform(config["resize_size1"], config["crop_size"], data_set))
            print(data_set, len(dsets[data_set]))
            dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                        batch_size=data_config[data_set]["batch_size"],
                                                        shuffle=True, num_workers=4)

        return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
            len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])                                          
            
    else:
        data_path =''
        for data_set in ["train_set", "test"]:
            dsets[data_set] = ImagePic(data_path,
                                        data_set,
                                        transform=image_transform(config["resize_size1"], config["crop_size"], data_set))
            print(data_set, len(dsets[data_set]))
            dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                        batch_size=config["batch_size"],
                                                        shuffle=True, num_workers=4)
        return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["train_set"], \
            len(dsets["train_set"]), len(dsets["test"]), len(dsets["train_set"])



def compute_result1(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    # print('____________________________________________')
    # print(bs[0][0])
    # print(clses[0][0])
    return torch.cat(bs).sign(), torch.cat(clses)
# , _
def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    # print('____________________________________________')
    # print(bs[0][0])
    # print(clses[0][0])
    return torch.cat(bs).sign(), torch.cat(clses)
# 

def ccompute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls,_ , _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    print('____________________________________________')
    print(bs[0][0])
    print(clses[0][0])
    return torch.cat(bs).sign(), torch.cat(clses)
# 

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
    # for iter in tqdm(range(num_query)):
        # print (queryL[iter, :])
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

if __name__ == "__main__":
    # root = "/home/sxz/code/gan/pytorch-CycleGAN-and-pix2pix-master/results"
    root = '/home/sxz/code/gan/pytorch-CycleGAN-and-pix2pix-master/results/corel-e50_pretrained/test_latest/images'
    train_dataset = data_cifar(root, train=True)
    # test_dataset = data_cifar(root, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for data, label in train_dataloader:
        print(data.shape)
        print(label)
        break