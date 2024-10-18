import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import CLIPVAD
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option
from scipy import ndimage

def select_topk_embeddings(scores, features_, k):
    _, idx_DESC = scores.sort(descending=True, dim=1)
    idx_topk = idx_DESC[:, :k]
    idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, features_.shape[2]])
    selected_embeddings = torch.gather(features_, 1, idx_topk)
    return selected_embeddings

def easy_snippets_mining(actionness, fertures_, args):
    actionness = actionness.squeeze()
    select_idx = torch.ones_like(actionness).cuda()
    # select_idx = self.dropout(select_idx)

    actionness_drop = actionness * select_idx

    actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
    actionness_rev_drop = actionness_rev * select_idx

    easy_act_mu = select_topk_embeddings(actionness_drop, fertures_, args.k_easy)

    easy_bkg_mu = select_topk_embeddings(actionness_rev_drop, fertures_, args.k_easy)

    k=max(1, int(fertures_.shape[-2] // args.k_easy))
    return easy_act_mu, easy_bkg_mu

def hard_snippets_mining(actionness, fertures_, args):
    actionness = actionness.squeeze()
    aness_np = actionness.cpu().detach().numpy()
    aness_median = np.median(aness_np, 1, keepdims=True)
    aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

    erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1,args.M))).astype(aness_np.dtype)
    erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1,args.m))).astype(aness_np.dtype)
    idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
    aness_region_inner = actionness * idx_region_inner
    hard_act_mu = select_topk_embeddings(aness_region_inner, fertures_, k=args.k_hard)

    dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,args.m))).astype(aness_np.dtype)
    dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,args.M))).astype(aness_np.dtype)
    idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
    aness_region_outer = actionness * idx_region_outer
    hard_bkg_mu = select_topk_embeddings(aness_region_outer, fertures_, k=args.k_hard)

    return hard_act_mu, hard_bkg_mu


# 欧氏距离
def Euclidean_distance_single(x, y):
    distance = []
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            dist = torch.cdist(x[:, i, :], y[:, j, :])
            distance.append(dist)

    distance = torch.stack(distance, dim=-1)
    return distance.mean(-1)


# KL 散度
def KL_divergence_single(x, y):
    distance = []
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            x_ij = x[:, i, :] + 1e-5
            y_ij = y[:, j, :] + 1e-5

            term1 = 0.5 * torch.einsum('bd,bd,bd->b', [(y_ij - x_ij), 1 / y_ij, (y_ij - x_ij)])
            term2 = 0.5 * (torch.log(y_ij).sum(-1) - torch.log(x_ij).sum(-1))
            term3 = 0.5 * ((x_ij / y_ij).sum(-1))

            dist = term1 + term2 + term3 - 0.5 * x_ij.shape[1]
            distance.append(1 / (dist + 1))

    distance = torch.stack(distance, dim=-1)
    return distance.mean(-1)


# Bhattacharyya 距离
def Bhattacharyya_distance_single(x, y):
    distance = []
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            x_ij = x[:, i, :] + 1e-5
            y_ij = y[:, j, :] + 1e-5

            term1 = 0.125 * torch.einsum('bd,bd,bd->b', [(x_ij - y_ij), 2 / (x_ij + y_ij), (x_ij - y_ij)])
            term2 = 0.5 * (torch.log((x_ij + y_ij) / 2).sum(-1) - (torch.log(x_ij).sum(-1) + torch.log(y_ij).sum(-1)))

            dist = term1 + term2
            distance.append(1 / (dist + 1))

    distance = torch.stack(distance, dim=-1)
    return distance.mean(-1)


# 马氏距离
def Mahalanobis_distance_single(x, y):
    distance = []
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            cov_inv = 2 / (x[:, i, :] + y[:, j, :] + 1e-5)
            dist = torch.einsum('bd,bd,bd->b', [(x[:, i, :] - y[:, j, :]), cov_inv, (x[:, i, :] - y[:, j, :])])
            distance.append(1 / (dist + 1))

    distance = torch.stack(distance, dim=-1)
    return distance.mean(-1)


def Intra_ProbabilsticContrastive(hard_query, easy_pos, easy_neg, args):


    if args.metric == 'Mahala':
        pos_distance = Mahalanobis_distance_single(hard_query, easy_pos)
        neg_distance = Mahalanobis_distance_single(hard_query, easy_neg)

    elif args.metric == 'KL_div':
        pos_distance = 0.5 * (KL_divergence_single(hard_query, easy_pos) + KL_divergence_single(easy_pos, hard_query))
        neg_distance = 0.5 * (KL_divergence_single(hard_query, easy_neg) + KL_divergence_single(easy_neg, hard_query))

    elif args.metric == 'Bhatta':
        pos_distance = Bhattacharyya_distance_single(hard_query, easy_pos)
        neg_distance = Bhattacharyya_distance_single(hard_query, easy_neg)

    elif args.metric == 'Euclidean':
        pos_distance = Euclidean_distance_single(hard_query, easy_pos)
        neg_distance = Euclidean_distance_single(hard_query, easy_neg)

    if args.loss_type == 'frobenius':
        loss = torch.norm(1 - pos_distance) + torch.norm(neg_distance)
        return loss

    elif args.loss_type == 'neg_log':
        loss = -1 * (torch.log(pos_distance) + torch.log(1 - neg_distance))
        return loss.mean()


def CLASM3(logits, feature_, device, lengths,  args):
    # 超参

    easy_act, easy_bkg = easy_snippets_mining(logits, feature_, args)
    hard_act, hard_bkg = hard_snippets_mining(logits, feature_, args)

    action_prob_contra_loss = args.alpha5 * Intra_ProbabilsticContrastive(hard_act, easy_act, easy_bkg, args)
    background_prob_contra_loss = args.alpha6 * Intra_ProbabilsticContrastive(hard_bkg, easy_bkg, easy_act, args)


    return action_prob_contra_loss + background_prob_contra_loss

def CLAS4(logits, labels, lengths, device, args):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])  # 将标签转换为目标形式
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    # 前一半为正常视频，后一半为异常视频
    batch_size = logits.shape[0]
    half_batch_size = batch_size // 2

    normal_scores = torch.zeros(half_batch_size).to(device)  # 存储正常视频的均分
    abnormal_scores = torch.zeros(half_batch_size).to(device)  # 存储异常视频的均分
    topk_abnormal_scores = torch.zeros(half_batch_size).to(device)  # 存储异常视频 top-k 片段的均分

    for i in range(batch_size):
        valid_length = lengths[i]
        k = int(valid_length / 16 + 1)

        # 取出有效的 logits 并计算均值
        if i < half_batch_size:
            # 处理正常视频
            normal_scores[i] = torch.mean(logits[i, 0:valid_length])
        else:
            # 处理异常视频
            abnormal_scores[i - half_batch_size] = torch.mean(logits[i, 0:valid_length])
            tmp, _ = torch.topk(logits[i, 0:valid_length], k=k, largest=True)
            topk_abnormal_scores[i - half_batch_size] = torch.mean(tmp)

    # 构建三元对比损失
    # 1. 正常视频的均分 < 异常视频的均分
    normal_vs_abnormal_loss = F.relu(normal_scores - abnormal_scores + args.alpha1).mean()

    # 2. 异常视频的均分 < top-k 异常片段的均分
    abnormal_vs_topk_loss = F.relu(abnormal_scores - topk_abnormal_scores + args.alpha2).mean()

    # 总损失 = 两个对比损失
    total_loss = normal_vs_abnormal_loss + abnormal_vs_topk_loss

    return total_loss




def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2, features_ = model(visual_features, None, prompt_text, feat_lengths)
            #loss1
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()
            #loss2
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()
            #loss3
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1

            # loss4 = CLASM3(logits1[32:], features_[32:], device, feat_lengths, args)

            loss5 = CLAS4(logits1, text_labels, feat_lengths, device, args)

            # loss = loss1 + loss2 + loss3 + loss4 + loss5

            loss = loss1 + loss2 + loss3 + loss5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                AP = AUC

                if AP > ap_best:
                    ap_best = AP 
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                
        scheduler.step()
        
        torch.save(model.state_dict(), '../model/model_cur.pth')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)