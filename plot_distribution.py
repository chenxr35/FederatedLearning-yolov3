import pandas as pd
from utils.options import args_parser
from utils.sampling import voc_noniid_dist, voc_iid_dist
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.utils import load_classes
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
import matplotlib.pyplot as plt
import seaborn as sns
import os

NUM = 10
IID = False

if __name__ == '__main__':

    os.makedirs("figure", exist_ok=True)

    args = args_parser()

    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    class_names = load_classes(data_config["names"])  # List of class names
    dataset_train = ListDataset(
        train_path,
        img_size=416,
        multiscale=args.multiscale_training,
        transform=AUGMENTATION_TRANSFORMS,
        class_names=class_names)

    if not IID:
        class_dist = voc_noniid_dist(dataset_train, NUM)
    else:
        class_dist = voc_iid_dist(dataset_train, NUM)

    class_dist = pd.DataFrame(class_dist)

    class_dist = class_dist.transpose()

    plt.figure(figsize=(10, 6))
    sns.heatmap(data=class_dist, cmap=sns.cubehelix_palette(as_cmap=True), center=3000, annot=True, fmt ='g')

    plt.xlabel('Client ID')
    plt.ylabel('Class Label')

    plt.savefig(f'figure/dist_{NUM}_iid{IID}.png')