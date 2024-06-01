import numpy as np
import copy

np.random.seed(42)

def build_dataset(dataset,num_meta,num_classes):


    img_num_list  = [num_meta] * num_classes

    data_list_val = {}
    for j in range(num_classes):
        data_list_val[j] = [i for i, label in enumerate(dataset.labels) if label == j]

    idx_to_meta = []
    idx_to_train = []

    for cls_idx, img_id_list in data_list_val.items():
        np.random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        idx_to_meta.extend(img_id_list[:img_num])
        idx_to_train.extend(img_id_list[img_num:])
    train_data = copy.deepcopy(dataset)
    train_data_meta = copy.deepcopy(dataset)

    train_data_meta.img_path = np.delete(dataset.img_path,idx_to_train,axis=0)
    train_data_meta.labels = np.delete(dataset.labels, idx_to_train, axis=0)
    train_data.img_path = np.delete(dataset.img_path, idx_to_meta, axis=0)
    train_data.labels = np.delete(dataset.labels, idx_to_meta, axis=0)

    return train_data_meta, train_data


def get_img_num_per_cls(dataset, imb_factor=None, num_meta=None):

    if dataset == 'cifar10':
        img_max = (50000-num_meta)/10
        cls_num = 10

    if dataset == 'cifar100':
        img_max = (50000-num_meta)/100
        cls_num = 100

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


