#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import copy
import numpy as np
import torch
import os

from utils.sampling import coco_iid, voc_iid, voc_noniid
from utils.options import args_parser
from models.Update import LocalUpdateYOLO
from models.Fed import FedAvg, FedAvgM
from models.utils import flatten_weights
  
# change image_dir "images" to "JPEGImages", add parameter 'class_names' and method '__getindex__' in ListDataset
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.models import load_model  # add parameter device in load_model
from pytorchyolo.test import _evaluate  # add parameter device in _evaluate
from pytorchyolo.utils.utils import load_classes, worker_seed_set, provide_determinism
from torch.utils.data import DataLoader
from pytorchyolo.utils.logger import Logger

import datetime

if __name__ == '__main__':

    now = datetime.datetime.now()
    now = f'{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}'

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    if args.seed != -1:
        provide_determinism(args.seed)

    iid = 'iid' if args.iid or args.num_users == 1 else 'noniid'

    logdir = f"{args.logdir}/{args.model}_{args.dataset}_{args.algo}_{args.num_users}_{iid}"
    logger = Logger(logdir)  # Tensorboard logger

    # Create output directories if missing
    checkpoints_dir = f"checkpoint/{args.model}_{args.dataset}_{args.algo}_{args.num_users}_{iid}/{now}"
    os.makedirs(checkpoints_dir, exist_ok=True)


    # load dataset and split users
    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])  # List of class names
    dataset_train = ListDataset(
        train_path,
        img_size=416,
        multiscale=args.multiscale_training,
        transform=AUGMENTATION_TRANSFORMS,
        class_names=class_names)
    dataset_test = ListDataset(
        valid_path,
        img_size=416,
        multiscale=args.multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    if args.dataset == 'coco':
        if args.iid:
            dict_users = coco_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in COCO')
    elif args.dataset == 'voc':
        if args.iid or args.num_users == 1:
            dict_users = voc_iid(dataset_train, args.num_users)
        else:
            dict_users = voc_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')


    # build model
    if args.model == 'yolov3':
        net_glob = load_model(args.config, args.device, args.pretrained_weights)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    last_loss = 0.0  # store last loss for convergence
    convergence = False  # signal of convergence


    # initialize each client
    mini_batch_size = net_glob.hyperparams['batch'] // net_glob.hyperparams['subdivisions']
    clients = []
    for idx in range(args.num_users):
        client = LocalUpdateYOLO(args=args, dataset=dataset_train, idxs=dict_users[idx], batch_size=mini_batch_size, client_id=idx, logger=logger)
        clients.append(client)


    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = DataLoader(
        dataset_train,
        batch_size=mini_batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=dataset_train.collate_fn,
        worker_init_fn=worker_seed_set)

    args.dataloader_size = len(dataloader)

    # Load validation dataloader
    validation_dataloader = DataLoader(
        dataset_test,
        batch_size=mini_batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=dataset_test.collate_fn,
        worker_init_fn=worker_seed_set)

    if args.algo == 'scaffold':
        control_global = copy.deepcopy(net_glob)
        control_global.to(args.device)
        control_weights = control_global.state_dict()
        # model for local control variates
        local_controls = [copy.deepcopy(control_global) for i in range(args.num_users)]
        # initialize total delta to 0 (sum of all control_delta, triangle Ci)
        delta_c = copy.deepcopy(net_glob.state_dict())
        # sum of delta_y / sample size
        delta_x = copy.deepcopy(net_glob.state_dict())

    elif args.algo == 'fedavgm':
        m_weights = copy.deepcopy(w_glob)
        m_flat = flatten_weights(m_weights, from_dict=True)
        m_flat = np.zeros_like(m_flat)

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(1, args.epochs+1):

        if args.algo == 'scaffold':
            for ci in delta_c:
                delta_c[ci] = 0.0
            for ci in delta_x:
                delta_x[ci] = 0.0

        loss_locals = []
        num_examples = [] # data size of each user
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = clients[idx]

            if args.algo == 'scaffold':
                w, loss, local_delta_c, local_delta, control_local_w = local.scaffold_train(net=copy.deepcopy(net_glob).to(args.device), net_glob=net_glob, glob_ep=iter-1, control_local=local_controls[idx], control_global=control_global)

                if iter != 1:
                    local_controls[idx].load_state_dict(control_local_w)

                # line16
                for key in delta_c:
                    if iter == 1:
                        delta_x[key] = torch.add(delta_x[key], w[key], alpha=len(dict_users[idx]))
                    else:
                        delta_x[key] = torch.add(delta_x[key], local_delta[key], alpha=len(dict_users[idx]))
                        delta_c[key] = torch.add(delta_c[key], local_delta_c[key], alpha=len(dict_users[idx]))
            else:
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), net_glob=net_glob, glob_ep=iter-1)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            num_examples.append(len(dict_users[idx]))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        if args.algo == 'fedavg' or args.algo == 'fedprox':
            w_glob = FedAvg(w_locals, num_examples)
        elif args.algo == 'fedavgm':
            w_glob, m_flat = FedAvgM(w_locals, w_glob, num_examples, m_flat, args.avgm_beta)

        elif args.algo == 'scaffold':
            # update the delta C (line 16)
            for key in delta_c:
                delta_c[key] = torch.div(delta_c[key], sum(num_examples))
                delta_x[key] = torch.div(delta_x[key], sum(num_examples))

            # update global control variate (line17)
            control_global_W = control_global.state_dict()
            # equation taking Ng, global step size = 1
            for key in control_global_W:
                if iter == 1:
                    w_glob[key] = delta_x[key]
                else:
                    w_glob[key] = torch.add(w_glob[key], delta_x[key])
                    sample_ratio = m / args.num_users
                    control_global_W[key] = torch.add(control_global_W[key], delta_c[key], alpha=sample_ratio)

            # update global control
            control_global.load_state_dict(control_global_W)

        else:
            exit("Error: unrecognized algorithm")

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        w_glob = net_glob.state_dict()

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        tensorboard_log = [
            ("train/average_loss", float(loss_avg))]
        logger.list_of_scalars_summary(tensorboard_log, iter*args.local_ep)

        # break if we achieve convergence, i.e., loss between two consecutive rounds is <0.0001
        # if abs(loss_avg - last_loss) < 0.0001:
        #     convergence = True

        # update the last loss
        last_loss = loss_avg

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if iter % args.checkpoint_interval == 0 or convergence:
            checkpoint_path = f"{checkpoints_dir}/{args.model}_ckpt_epoch{iter*args.local_ep}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(net_glob.state_dict(), checkpoint_path)

        # ########
        # Evaluate
        # ########

        if iter % args.evaluation_interval == 0 or convergence:
            print("\n---- Evaluating Model ----")

            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                net_glob,
                args.device,
                validation_dataloader,
                class_names,
                img_size=net_glob.hyperparams['height'],
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, iter*args.local_ep)

        # stop training
        # if convergence:
        #     break
