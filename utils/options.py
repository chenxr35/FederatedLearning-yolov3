#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=60, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=3, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=16, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='yolov3', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='voc', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=20, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    # yolo arguments
    parser.add_argument("-d", "--data", type=str, default="config/voc.data", help="Path to data config file (.data)")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--config", type=str, default="config/yolov3-voc.cfg",
                        help="Path to model definition file (.cfg)")
    parser.add_argument("--pretrained_weights", type=str, default="../PyTorch-YOLOv3/weights/darknet53.conv.74",
                        help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=2, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5,
                        help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--checkpoint_interval", type=int, default=2,
                        help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=2,
                        help="Interval of epochs between evaluations on validation set")
    parser.add_argument('--algo', type=str, default='fedavg', help="name of federated learning algorithm")
    parser.add_argument('--mu', type=float, default=0.01, help="mu parameter for FedProx algorithm")
    parser.add_argument('--avgm_beta', type=float, default=0.7, help="server momentum for FedAvgM algorithm")
    parser.add_argument("--logdir", type=str, default="log",
                        help="Directory for training log files (e.g. for TensorBoard)")

    args = parser.parse_args()
    return args
