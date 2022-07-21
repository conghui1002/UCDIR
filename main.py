#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from tqdm import tqdm
import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def train(train_loader, model, criterion, optimizer, epoch, args, cluster_result):
    # switch to train mode
    model.train()

    crosscon_feat_domains = ['A', 'B']

    for i, (images_A, image_ids_A, images_B, image_ids_B, cates_A, cates_B) in enumerate(train_loader):

        rest_image_ids_A = torch.tensor([id for id in torch.tensor(range(args.mocok_A)) if id not in image_ids_A])
        rest_image_ids_B = torch.tensor([id for id in torch.tensor(range(args.mocok_B)) if id not in image_ids_B])

        if args.gpu is not None:
            images_A[0] = images_A[0].cuda(args.gpu, non_blocking=True)
            images_A[1] = images_A[1].cuda(args.gpu, non_blocking=True)
            image_ids_A = image_ids_A.cuda(args.gpu, non_blocking=True)
            rest_image_ids_A = rest_image_ids_A.cuda(args.gpu, non_blocking=True)

            images_B[0] = images_B[0].cuda(args.gpu, non_blocking=True)
            images_B[1] = images_B[1].cuda(args.gpu, non_blocking=True)
            image_ids_B = image_ids_B.cuda(args.gpu, non_blocking=True)
            rest_image_ids_B = rest_image_ids_B.cuda(args.gpu, non_blocking=True)

        ####
        # compute output
        output_A, target_A, output_B, target_B, losses_instcon, \
        output_proto_dict, target_proto_dict, losses_protocon, q_A,  q_B, \
        losses_crosscon, losses_selfentro, losses_distlogits, losses_cwcon = model(im_q_A=images_A[0], im_k_A=images_A[1],
                                                                                    im_id_A=image_ids_A, rest_im_id_A=rest_image_ids_A,
                                                                                    im_q_B=images_B[0], im_k_B=images_B[1],
                                                                                    im_id_B=image_ids_B, rest_im_id_B=rest_image_ids_B,
                                                                                    cluster_result=cluster_result,
                                                                                    num_clusterproto=args.num_clusterproto,
                                                                                    num_clustercross=args.num_clustercross,
                                                                                    if_add_crosscon=args.if_crosscon,
                                                                                    crosscon_feat_domains=crosscon_feat_domains,
                                                                                    crosscon_stability=not args.nocrosscon_stability,
                                                                                    if_add_selfentro=args.if_selfentro,
                                                                                    num_clusterselfentro=args.num_clusterselfentro,
                                                                                    criterion=criterion)
        # InfoNCE loss
        inst_loss_A = losses_instcon['domain_A']
        inst_loss_B = losses_instcon['domain_B']

        if epoch < args.warmup_epoch:
            cur_instcon_weight = args.instcon_weightwarm
        else:
            cur_instcon_weight = args.instcon_weight

        loss_A = inst_loss_A * cur_instcon_weight
        loss_B = inst_loss_B * cur_instcon_weight

        if epoch >= args.warmup_epoch:
            # ProtoNCE loss
            for feat_domain_id in ['A', 'B']:
                if feat_domain_id == 'A':
                    images = images_A
                else:
                    images = images_B

                if args.if_protonceshared:
                    add_domain_id = ['All']
                else:
                    add_domain_id = []

                for proto_domain_id in [feat_domain_id]+add_domain_id:

                    loss_proto = torch.mean(torch.stack(losses_protocon['feat_' + feat_domain_id + '_proto_' + proto_domain_id]))

                    if feat_domain_id == 'A':
                        loss_A += (loss_proto * args.proto_weight / (len(add_domain_id) + 1))
                    else:
                        loss_B += (loss_proto * args.proto_weight / (len(add_domain_id) + 1))

            # CW loss
            cwcon_loss_A = losses_cwcon['domain_A']
            cwcon_loss_B = losses_cwcon['domain_B']

            if epoch <= args.cwcon_startepoch:
                cur_cwcon_weight = args.cwcon_weightstart
            elif epoch < args.cwcon_satureepoch:
                cur_cwcon_weight = args.cwcon_weightstart + (args.cwcon_weightsature - args.cwcon_weightstart) \
                                    * ((epoch - args.cwcon_startepoch) / (args.cwcon_satureepoch - args.cwcon_startepoch))
            else:
                cur_cwcon_weight = args.cwcon_weightsature

            loss_A += cwcon_loss_A * cur_cwcon_weight
            loss_B += cwcon_loss_B * cur_cwcon_weight

        # compute gradient and do SGD step
        all_loss = (loss_A + loss_B) / 2

        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

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
    """
    Args:
        x: data to be clustered
    """

    print('performing kmeans clustering')
    results = {'im2cluster_A': [], 'centroids_A': [], 'density_A': [], 'Dcluster_A': [], 'centroids_A_wonorm': [],
               'im2cluster_B': [], 'centroids_B': [], 'density_B': [], 'Dcluster_B': [], 'centroids_B_wonorm': [],
               'im2cluster_All': [], 'centroids_All': [], 'density_All': [], 'Dcluster_All': [], 'centroids_All_wonorm': []}
    for domain_id in ['A', 'B', 'All']:
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
            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = args.gpu
            index = faiss.IndexFlatL2(d)

            clus.train(x, index)
            D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
            im2cluster = [int(n[0]) for n in I]

            # get cluster centroids
            centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

            # sample-to-centroid distances for each cluster
            Dcluster = [[] for c in range(k)]
            for im, i in enumerate(im2cluster):
                Dcluster[i].append(D[im][0])

            # concentration estimation (phi)
            density = np.zeros(k)
            for i, dist in enumerate(Dcluster):
                if len(dist) > 1:
                    d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                    density[i] = d

                    # if cluster only has one point, use the max to estimate its concentration
            dmax = density.max()
            for i, dist in enumerate(Dcluster):
                if len(dist) <= 1:
                    density[i] = dmax
            density = density.clip(np.percentile(density, 10),
                                   np.percentile(density, 90))  # clamp extreme values for stability
            density = args.temperature * density / density.mean()  # scale the mean to temperature

            # convert to cuda Tensors for broadcast
            centroids = torch.Tensor(centroids).cuda()
            centroids_normed = nn.functional.normalize(centroids, p=2, dim=1)
            im2cluster = torch.LongTensor(im2cluster).cuda()
            density = torch.Tensor(density).cuda()

            results['centroids_'+domain_id].append(centroids_normed)
            results['centroids_' + domain_id+'_wonorm'].append(centroids)
            results['density_'+domain_id].append(density)
            results['im2cluster_'+domain_id].append(im2cluster)
            results['Dcluster_'+domain_id].append(Dcluster)

    return results

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

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        print(lr)
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
