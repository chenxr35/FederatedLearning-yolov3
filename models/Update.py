#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import tqdm
import copy
import torch.optim as optim
from pytorchyolo.utils.utils import to_cpu, worker_seed_set
from pytorchyolo.utils.loss import compute_loss


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class YOLODatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        img_path, img, bb_targets = self.dataset.__getitem__(self.idxs[item])
        return img_path, img, bb_targets

    def collate_fn(self, batch):
        return self.dataset.collate_fn(batch)


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateYOLO(object):

    def __init__(self, args, dataset, idxs, batch_size, client_id, logger):
        self.args = args
        self.dataset = YOLODatasetSplit(dataset, idxs)
        self.client_id = client_id+1
        self.logger = logger

        # #################
        # Create Dataloader
        # #################

        # Load training dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.args.n_cpu,
            pin_memory=True,
            collate_fn=self.dataset.collate_fn,
            worker_init_fn=worker_seed_set)


    def train(self, net, net_glob, glob_ep):

        # ################
        # Create optimizer
        # ################

        params = [p for p in net.parameters() if p.requires_grad]

        if (net.hyperparams['optimizer'] in [None, "adam"]):
            optimizer = optim.Adam(
                params,
                lr=net.hyperparams['learning_rate'],
                weight_decay=net.hyperparams['decay'],
            )
        elif (net.hyperparams['optimizer'] == "sgd"):
            optimizer = optim.SGD(
                params,
                lr=net.hyperparams['learning_rate'],
                weight_decay=net.hyperparams['decay'],
                momentum=net.hyperparams['momentum'])
        else:
            exit("Unknown optimizer. Please choose between (adam, sgd).")


        # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
        # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
        # instead of: 0, 10, 20
        epoch_loss = []
        for epoch in range(1, self.args.local_ep + 1):

            print(f"\n---- Training Model ---- Client{self.client_id} ----")

            net.train()  # Set model to training mode

            batch_loss = []
            for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(self.dataloader, desc=f"Training Epoch {epoch + glob_ep * self.args.local_ep}")):
                batches_done = len(self.dataloader) * (epoch + self.args.local_ep * glob_ep) + batch_i

                imgs = imgs.to(self.args.device, non_blocking=True)
                targets = targets.to(self.args.device)

                outputs = net(imgs)

                loss, loss_components = compute_loss(outputs, targets, net)

                if self.args.algo == 'fedprox':
                    proximal_term = 0.0
                    # iterate through the current and global model parameters
                    for w, w_t in zip(net.parameters(), net_glob.parameters()):
                    # update the proximal term 
                        proximal_term += (w-w_t).norm(2)

                    loss += (self.args.mu/2)*proximal_term

                batch_loss.append(to_cpu(loss).item())

                loss.backward()

                ###############
                # Run optimizer
                ###############

                if batches_done % net.hyperparams['subdivisions'] == 0:
                    # Adapt learning rate
                    # Get learning rate defined in cfg
                    lr = net.hyperparams['learning_rate']
                    if batches_done < net.hyperparams['burn_in']:
                        # Burn in
                        lr *= (batches_done / net.hyperparams['burn_in'])
                    else:
                        # Set and parse the learning rate to the steps defined in the cfg
                        for threshold, value in net.hyperparams['lr_steps']:
                            if batches_done > threshold:
                                lr *= value

                    # Log the learning rate
                    self.logger.scalar_summary(f"train/Client{self.client_id}/learning_rate", lr, batches_done)

                    # Set learning rate
                    for g in optimizer.param_groups:
                        g['lr'] = lr

                    # Run optimizer
                    optimizer.step()

                    # Reset gradients
                    optimizer.zero_grad()

                # ############
                # Log progress
                # ############

                # Tensorboard logging
                tensorboard_log = [
                    (f"train/Client{self.client_id}/iou_loss", float(loss_components[0])),
                    (f"train/Client{self.client_id}/obj_loss", float(loss_components[1])),
                    (f"train/Client{self.client_id}/class_loss", float(loss_components[2])),
                    (f"train/Client{self.client_id}/loss", to_cpu(loss).item())]
                self.logger.list_of_scalars_summary(tensorboard_log, batches_done)

                net.seen += imgs.size(0)


            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def scaffold_train(self, net, net_glob, glob_ep, control_local, control_global):

        control_local_w = control_local.state_dict()
        control_global_w = control_global.state_dict()
        step_count = 0

        # ################
        # Create optimizer
        # ################

        params = [p for p in net.parameters() if p.requires_grad]

        if (net.hyperparams['optimizer'] in [None, "adam"]):
            optimizer = optim.Adam(
                params,
                lr=net.hyperparams['learning_rate'],
                weight_decay=net.hyperparams['decay'],
            )
        elif (net.hyperparams['optimizer'] == "sgd"):
            optimizer = optim.SGD(
                params,
                lr=net.hyperparams['learning_rate'],
                weight_decay=net.hyperparams['decay'],
                momentum=net.hyperparams['momentum'])
        else:
            exit("Unknown optimizer. Please choose between (adam, sgd).")


        # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
        # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
        # instead of: 0, 10, 20
        epoch_loss = []
        for epoch in range(1, self.args.local_ep + 1):

            print(f"\n---- Training Model ---- Client{self.client_id} ----")

            net.train()  # Set model to training mode

            batch_loss = []
            for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(self.dataloader, desc=f"Training Epoch {epoch + glob_ep * self.args.local_ep}")):
                batches_done = len(self.dataloader) * (epoch + self.args.local_ep * glob_ep) + batch_i

                imgs = imgs.to(self.args.device, non_blocking=True)
                targets = targets.to(self.args.device)

                outputs = net(imgs)

                loss, loss_components = compute_loss(outputs, targets, net)

                batch_loss.append(to_cpu(loss).item())

                loss.backward()

                ###############
                # Run optimizer
                ###############

                if batches_done % net.hyperparams['subdivisions'] == 0:
                    # Adapt learning rate
                    # Get learning rate defined in cfg
                    lr = net.hyperparams['learning_rate']
                    if batches_done < net.hyperparams['burn_in']:
                        # Burn in
                        lr *= (batches_done / net.hyperparams['burn_in'])
                    else:
                        # Set and parse the learning rate to the steps defined in the cfg
                        for threshold, value in net.hyperparams['lr_steps']:
                            if batches_done > threshold:
                                lr *= value

                    # Log the learning rate
                    self.logger.scalar_summary(f"train/Client{self.client_id}/learning_rate", lr, batches_done)

                    # Set learning rate
                    for g in optimizer.param_groups:
                        g['lr'] = lr

                    # Run optimizer
                    optimizer.step()

                    # Reset gradients
                    optimizer.zero_grad()

                    # ----------------------#
                    #   SCAFFOLD算法
                    # ----------------------#
                    local_weights = net.state_dict()
                    for w in local_weights:
                        # line 10 in algo
                        local_weights[w] = local_weights[w] - lr * (
                                    control_global_w[w] - control_local_w[w])

                    # update local model params
                    net.load_state_dict(local_weights)

                    step_count += 1

                # ############
                # Log progress
                # ############

                # Tensorboard logging
                tensorboard_log = [
                    (f"train/Client{self.client_id}/iou_loss", float(loss_components[0])),
                    (f"train/Client{self.client_id}/obj_loss", float(loss_components[1])),
                    (f"train/Client{self.client_id}/class_loss", float(loss_components[2])),
                    (f"train/Client{self.client_id}/loss", to_cpu(loss).item())]
                self.logger.list_of_scalars_summary(tensorboard_log, batches_done)

                net.seen += imgs.size(0)


            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # ----------------------#
        #   SCAFFOLD算法
        # ----------------------#
        for g in optimizer.param_groups:
            lr = g['lr']
        new_control_local_w = control_local.state_dict()
        control_delta = copy.deepcopy(control_local_w)
        # model_weights -> y_(i)
        model_weights = net.state_dict()
        global_weights = net_glob.state_dict()
        local_delta = copy.deepcopy(model_weights)
        K = step_count
        for w in model_weights:
            # line 12 in algo
            new_control_local_w[w] = new_control_local_w[w] - control_global_w[w] + (
                    global_weights[w] - model_weights[w]) / (K * lr)
            # line 13
            control_delta[w] = new_control_local_w[w] - control_local_w[w]
            local_delta[w] -= global_weights[w]
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, new_control_local_w


