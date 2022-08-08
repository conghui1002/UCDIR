#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

import loader
import builder
from sklearn.metrics.pairwise import cosine_similarity

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data-A', metavar='DIR Domain A', help='path to domain A dataset')
parser.add_argument('--data-B', metavar='DIR Domain B', help='path to domain B dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 2x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--clean-model', default='', type=str, metavar='PATH',
                    help='path to clean model (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--low-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')
parser.add_argument('--warmup-epoch', default=20, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco-v2/SimCLR data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--exp-dir', default='experiment_pcl', type=str,
                    help='the directory of the experiment')
parser.add_argument('--ckpt-save', default=20, type=int,
                    help='the frequency of saving ckpt')
parser.add_argument('--mocok-A', default=120906, type=int,
                    help='queue size; number of negative keys for domain A(default: 65536)')
parser.add_argument('--mocok-B', default=120906, type=int,
                    help='queue size; number of negative keys for domain B(default: 65536)')
parser.add_argument('--num-cluster', default='250,500,1000', type=str,
                    help='number of clusters for self entropy loss')

parser.add_argument('--instcon-weight', default=1.0, type=float,
                    help='the weight for instance contrastive loss after warm up')
parser.add_argument('--cwcon-weightstart', default=0.0, type=float,
                    help='the starting weight for cluster-wise contrastive loss')
parser.add_argument('--cwcon-weightsature', default=1.0, type=float,
                    help='the satuate weight for cluster-wise contrastive loss')
parser.add_argument('--cwcon-startepoch', default=20, type=int,
                    help='the start epoch for scluster-wise contrastive loss')
parser.add_argument('--cwcon-satureepoch', default=100, type=int,
                    help='the saturated epoch for cluster-wise contrastive loss')
parser.add_argument('--cwcon-filterthresh', default=0.2, type=float,
                    help='the threshold of filter for cluster-wise contrastive loss')
parser.add_argument('--selfentro-temp', default=0.2, type=float,
                    help='the temperature for self-entropy loss')
parser.add_argument('--selfentro-startepoch', default=20, type=int,
                    help='the start epoch for self entropy loss')
parser.add_argument('--selfentro-weight', default=20, type=float,
                    help='the start weight for self entropy loss')
parser.add_argument('--distofdist-startepoch', default=20, type=int,
                    help='the start epoch for dist of dist loss')
parser.add_argument('--distofdist-weight', default=20, type=float,
                    help='the start weight for dist of dist loss')
parser.add_argument('--prec-nums', default='1,5,15', type=str,
                    help='the evaluation metric')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.num_cluster = args.num_cluster.split(',')

    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))
    model = builder.UCDIR(
        models.__dict__[args.arch],
        dim=args.low_dim, K_A=args.mocok_A, K_B=args.mocok_B, m=args.moco_m,
        T=args.temperature, mlp=args.mlp, selfentro_temp=args.selfentro_temp,
        num_cluster=args.num_cluster,  cwcon_filterthresh=args.cwcon_filterthresh)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.clean_model:
        if os.path.isfile(args.clean_model):
            print("=> loading pretrained clean model '{}'".format(args.clean_model))

            loc = 'cuda:{}'.format(args.gpu)
            clean_checkpoint = torch.load(args.clean_model, map_location=loc)

            current_state = model.state_dict()
            used_pretrained_state = {}

            for k in current_state:
                if 'encoder' in k:
                    k_parts = '.'.join(k.split('.')[1:])
                    used_pretrained_state[k] = clean_checkpoint['state_dict']['module.encoder_q.'+k_parts]
            current_state.update(used_pretrained_state)
            model.load_state_dict(current_state)
        else:
            print("=> no clean model found at '{}'".format(args.clean_model))

    cudnn.benchmark = True

    traindirA = os.path.join(args.data_A, 'train')
    traindirB = os.path.join(args.data_B, 'train')

    train_dataset = loader.TrainDataset(traindirA, traindirB, args.aug_plus)
    eval_dataset = loader.EvalDataset(traindirA, traindirB)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)
    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)

    info_save = open(os.path.join(args.exp_dir, 'info.txt'), 'w')
    best_res_A = [0., 0., 0.]
    best_res_B = [0., 0., 0.]
    for epoch in range(args.epochs):

        features_A, features_B, _, _ = compute_features(eval_loader, model, args)

        features_A = features_A.numpy()
        features_B = features_B.numpy()

        if epoch == 0:
            model.queue_A.data = torch.tensor(features_A).T.cuda()
            model.queue_B.data = torch.tensor(features_B).T.cuda()

        cluster_result = None
        if epoch >= args.warmup_epoch:
            cluster_result = run_kmeans(features_A, features_B, args)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, info_save, cluster_result)

        # test for one epoch
        features_A, features_B, targets_A, targets_B = compute_features(eval_loader, model, args)
        features_A = features_A.numpy()
        targets_A = targets_A.numpy()

        features_B = features_B.numpy()
        targets_B = targets_B.numpy()

        prec_nums = args.prec_nums.split(',')
        res_A, res_B = retrieval_precision_cal(features_A, targets_A, features_B, targets_B,
                                               preck=(int(prec_nums[0]), int(prec_nums[1]), int(prec_nums[2])))

        if (best_res_A[0] + best_res_B[0]) / 2 < (res_A[0] + res_B[0]) / 2:
            best_res_A = res_A
            best_res_B = res_B

    info_save.write("Best result:\n")
    info_save.write("Domain A->B: P@1: {}; P@5: {}; P@15: {} \n".format(best_res_A[0], best_res_A[1], best_res_A[2]))
    info_save.write("Domain B->A: P@1: {}; P@5: {}; P@15: {} \n".format(best_res_B[0], best_res_B[1], best_res_B[2]))


def train(train_loader, model, criterion, optimizer, epoch, args, info_save, cluster_result):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    losses = {'Inst_A': AverageMeter('Inst_Loss_A', ':.4e'),
              'Inst_B': AverageMeter('Inst_Loss_B', ':.4e'),
              'Cwcon_A': AverageMeter('Cwcon_Loss_A', ':.4e'),
              'Cwcon_B': AverageMeter('Cwcon_Loss_B', ':.4e'),
              'SelfEntropy': AverageMeter('Loss_SelfEntropy', ':.4e'),
              'DistLogits': AverageMeter('Loss_DistLogits', ':.4e'),
              'Total_loss': AverageMeter('Loss_Total', ':.4e')}

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time,
         losses['SelfEntropy'],
         losses['DistLogits'],
         losses['Total_loss'],
         losses['Inst_A'], losses['Inst_B'],
         losses['Cwcon_A'], losses['Cwcon_B']],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images_A, image_ids_A, images_B, image_ids_B, cates_A, cates_B) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.gpu is not None:
            images_A[0] = images_A[0].cuda(args.gpu, non_blocking=True)
            images_A[1] = images_A[1].cuda(args.gpu, non_blocking=True)
            image_ids_A = image_ids_A.cuda(args.gpu, non_blocking=True)

            images_B[0] = images_B[0].cuda(args.gpu, non_blocking=True)
            images_B[1] = images_B[1].cuda(args.gpu, non_blocking=True)
            image_ids_B = image_ids_B.cuda(args.gpu, non_blocking=True)

        losses_instcon, \
        q_A,  q_B, \
        losses_selfentro, \
        losses_distlogits, \
        losses_cwcon = model(im_q_A=images_A[0], im_k_A=images_A[1],
                             im_id_A=image_ids_A, im_q_B=images_B[0],
                             im_k_B=images_B[1], im_id_B=image_ids_B,
                             cluster_result=cluster_result,
                             criterion=criterion)

        inst_loss_A = losses_instcon['domain_A']
        inst_loss_B = losses_instcon['domain_B']

        losses['Inst_A'].update(inst_loss_A.item(), images_A[0].size(0))
        losses['Inst_B'].update(inst_loss_B.item(), images_B[0].size(0))

        loss_A = inst_loss_A * args.instcon_weight
        loss_B = inst_loss_B * args.instcon_weight

        if epoch >= args.warmup_epoch:

            cwcon_loss_A = losses_cwcon['domain_A']
            cwcon_loss_B = losses_cwcon['domain_B']

            losses['Cwcon_A'].update(cwcon_loss_A.item(), images_A[0].size(0))
            losses['Cwcon_B'].update(cwcon_loss_B.item(), images_B[0].size(0))

            if epoch <= args.cwcon_startepoch:
                cur_cwcon_weight = args.cwcon_weightstart
            elif epoch < args.cwcon_satureepoch:
                cur_cwcon_weight = args.cwcon_weightstart + (args.cwcon_weightsature - args.cwcon_weightstart) * \
                                   ((epoch - args.cwcon_startepoch) / (args.cwcon_satureepoch - args.cwcon_startepoch))
            else:
                cur_cwcon_weight = args.cwcon_weightsature

            loss_A += cwcon_loss_A * cur_cwcon_weight
            loss_B += cwcon_loss_B * cur_cwcon_weight

        all_loss = (loss_A + loss_B) / 2

        if epoch >= args.selfentro_startepoch:

            losses_selfentro_list = []
            for key in losses_selfentro.keys():
                losses_selfentro_list.extend(losses_selfentro[key])

            losses_selfentro_mean = torch.mean(torch.stack(losses_selfentro_list))
            losses['SelfEntropy'].update(losses_selfentro_mean.item(), images_A[0].size(0))

            all_loss += losses_selfentro_mean * args.selfentro_weight

        if epoch >= args.distofdist_startepoch:

            losses_distlogits_list = []
            for key in losses_distlogits.keys():
                losses_distlogits_list.extend(losses_distlogits[key])

            losses_distlogits_mean = torch.mean(torch.stack(losses_distlogits_list))
            losses['DistLogits'].update(losses_distlogits_mean.item(), images_A[0].size(0))

            all_loss += losses_distlogits_mean * args.distofdist_weight

        losses['Total_loss'].update(all_loss.item(), images_A[0].size(0))

        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            info = progress.display(i)
            info_save.write(info + '\n')


def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()

    features_A = torch.zeros(eval_loader.dataset.domainA_size, args.low_dim).cuda()
    features_B = torch.zeros(eval_loader.dataset.domainB_size, args.low_dim).cuda()

    targets_all_A = torch.zeros(eval_loader.dataset.domainA_size, dtype=torch.int64).cuda()
    targets_all_B = torch.zeros(eval_loader.dataset.domainB_size, dtype=torch.int64).cuda()

    for i, (images_A, indices_A, targets_A, images_B, indices_B, targets_B) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images_A = images_A.cuda(non_blocking=True)
            images_B = images_B.cuda(non_blocking=True)

            targets_A = targets_A.cuda(non_blocking=True)
            targets_B = targets_B.cuda(non_blocking=True)

            feats_A, feats_B = model(im_q_A=images_A, im_q_B=images_B, is_eval=True)

            features_A[indices_A] = feats_A
            features_B[indices_B] = feats_B

            targets_all_A[indices_A] = targets_A
            targets_all_B[indices_B] = targets_B

    return features_A.cpu(), features_B.cpu(), targets_all_A.cpu(), targets_all_B.cpu()


def run_kmeans(x_A, x_B, args):

    print('performing kmeans clustering')
    results = {'im2cluster_A': [], 'centroids_A': [],
               'im2cluster_B': [], 'centroids_B': []}
    for domain_id in ['A', 'B']:
        if domain_id == 'A':
            x = x_A
        elif domain_id == 'B':
            x = x_B
        else:
            x = np.concatenate([x_A, x_B], axis=0)

        for seed, num_cluster in enumerate(args.num_cluster):
            # intialize faiss clustering parameters
            d = x.shape[1]
            k = int(num_cluster)
            clus = faiss.Clustering(d, k)
            clus.verbose = True
            clus.niter = 20
            clus.nredo = 5
            clus.seed = seed
            clus.max_points_per_centroid = 2000
            clus.min_points_per_centroid = 2
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = args.gpu
            index = faiss.IndexFlatL2(d)

            clus.train(x, index)
            D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
            im2cluster = [int(n[0]) for n in I]

            # get cluster centroids
            centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

            # convert to cuda Tensors for broadcast
            centroids = torch.Tensor(centroids).cuda()
            centroids_normed = nn.functional.normalize(centroids, p=2, dim=1)
            im2cluster = torch.LongTensor(im2cluster).cuda()

            results['centroids_'+domain_id].append(centroids_normed)
            results['im2cluster_'+domain_id].append(im2cluster)

    return results


def retrieval_precision_cal(features_A, targets_A, features_B, targets_B, preck=(1, 5, 15)):

    dists = cosine_similarity(features_A, features_B)

    res_A = []
    res_B = []
    for domain_id in ['A', 'B']:
        if domain_id == 'A':
            query_targets = targets_A
            gallery_targets = targets_B

            all_dists = dists

            res = res_A
        else:
            query_targets = targets_B
            gallery_targets = targets_A

            all_dists = dists.transpose()
            res = res_B

        sorted_indices = np.argsort(-all_dists, axis=1)
        sorted_cates = gallery_targets[sorted_indices.flatten()].reshape(sorted_indices.shape)
        correct = (sorted_cates == np.tile(query_targets[:, np.newaxis], sorted_cates.shape[1]))

        for k in preck:
            total_num = 0
            positive_num = 0
            for index in range(all_dists.shape[0]):

                temp_total = min(k, (gallery_targets == query_targets[index]).sum())
                pred = correct[index, :temp_total]

                total_num += temp_total
                positive_num += pred.sum()
            res.append(positive_num / total_num * 100.0)

    return res_A, res_B


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        return ' '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.5 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
