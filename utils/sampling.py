#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import collections

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def coco_iid(dataset, num_users):
    """
    Sample I.I.D. client data from COCO dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(dataset.__len__()/num_users)
    dict_users, all_idxs = {}, [i for i in range(dataset.__len__())]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def coco_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from COCO dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    pass


def voc_iid(dataset, num_users):
    """
    Sample I.I.D. client data from PASCAL VOC dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(dataset.__len__()/num_users)
    dict_users, all_idxs = {}, [i for i in range(dataset.__len__())]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def voc_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from PASCAL VOC dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = collections.defaultdict(set)
    allocated = set()
    i = 0
    for class_name in dataset.class_image:
        class_idxs = dataset.__getindex__(class_name)
        class_idxs = set(class_idxs) - allocated
        allocated = allocated.union(class_idxs)
        class_idxs = list(class_idxs)
        id_1 = i % num_users
        id_2 = (i+1) % num_users
        selection = set(np.random.choice(class_idxs, int(len(class_idxs)//2), replace=False))
        dict_users[id_1] = dict_users[id_1].union(selection)
        remainder = set(class_idxs) - selection
        dict_users[id_2] = dict_users[id_2].union(remainder)
        i += 1
    return dict_users


def voc_noniid_dist(dataset, num_users):

    class_dist = {}
    for class_name in dataset.class_image:
        class_dist[class_name] = [0 for i in range(num_users)]

    dict_users = collections.defaultdict(set)
    allocated = set()
    i = 0
    for class_name in dataset.class_image:
        class_idxs = dataset.__getindex__(class_name)
        class_idxs = set(class_idxs) - allocated
        allocated = allocated.union(class_idxs)
        class_idxs = list(class_idxs)
        id_1 = i % num_users
        id_2 = (i + 1) % num_users
        selection = set(np.random.choice(class_idxs, int(len(class_idxs) // 2), replace=False))
        dict_users[id_1] = dict_users[id_1].union(selection)
        remainder = set(class_idxs) - selection
        dict_users[id_2] = dict_users[id_2].union(remainder)
        i += 1

    for i in range(num_users):

        img_idxs = list(dict_users[i])

        for idx in img_idxs:
            class_names = dataset.__getclass__(idx)
            for class_name in class_names:
                class_dist[class_name][i] += 1

    return class_dist


def voc_iid_dist(dataset, num_users):

    class_dist = {}
    for class_name in dataset.class_image:
        class_dist[class_name] = [0 for i in range(num_users)]

    num_items = int(dataset.__len__() / num_users)
    dict_users, all_idxs = {}, [i for i in range(dataset.__len__())]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))

        img_idxs = list(dict_users[i])

        for idx in img_idxs:
            class_names = dataset.__getclass__(idx)
            for class_name in class_names:
                class_dist[class_name][i] += 1

        all_idxs = list(set(all_idxs) - dict_users[i])

    return class_dist


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
