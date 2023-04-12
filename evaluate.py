import torch


def evaluate(pred, ground_truth, top_num, rr_num):
    performance = {}
    sharp = torch.zeros(len(rr_num))
    roi = torch.zeros(len(rr_num))
    rank_score = torch.zeros(len(rr_num))
    for i in range(len(rr_num)):
        rank_top = 0
        pred_sort = torch.argsort(pred[:, i], dim=0, descending=True)
        gt_sort = torch.argsort(ground_truth[:, i], dim=0, descending=True)
        pred_topN = pred_sort[:top_num]
        for j in range(top_num):
            gt_top = torch.where(gt_sort == pred_topN[j])[0]
            residual = gt_top - i if gt_top - i > 0 else 0
            rank_top += 100 / (100 + residual)
        rank_top /= top_num
        topN = ground_truth[pred_topN, i]
        sharp[i] = (torch.mean(topN) - 0.05) / torch.std(topN) if torch.abs(torch.std(topN)) > 1e-1 else torch.mean(
            topN) / 1e-1
        roi[i] = torch.mean(topN)
        rank_score[i] = rank_top
    performance['sharp'] = sharp
    performance['roi'] = roi
    performance['rank_score'] = rank_score
    return performance['sharp'], performance['roi'], performance['rank_score']


def get_correlation(x, y, loss_weight, device):
    return - torch.corrcoef(torch.cat([x.unsqueeze(0), y.unsqueeze(0)]))[0][1]


def get_loss(x, y, loss_weight, device):
    mse = torch.mean((x - y) ** 2)
    sort_x = x.argsort(0, True).argsort(0, False)
    sort_y = y.argsort(0, True).argsort(0, False)
    loss_res = abs(sort_x - sort_y) / (sort_y + 1)
    # loss_res = loss_res * ((x - y) ** 2)
    rank_mean = torch.mean(loss_res)
    # rank_mean = (loss_res - torch.min(loss_res)) / (torch.max(loss_res) - torch.min(loss_res))
    loss = loss_weight[0] * mse + loss_weight[1] * rank_mean
    # return rank_mean
    return loss


def get_loss1(pred, ground_truth, ped_weight, loss_weight, rr_num, device, compare=0):
    loss_list = []
    for i in rr_num:
        if compare:
            loss_list.append(get_loss2(pred[:, i], ground_truth[:, i]))
        else:
            loss_rr = get_loss(pred[:, i], ground_truth[:, i], loss_weight, device)
            loss_cor = get_correlation(pred[:, i], ground_truth[:, i], loss_weight, device)
            # loss_sum = 0.6 * loss_rr + 0.4 * loss_cor
            loss_sum = loss_rr
            loss_list.append(loss_sum)
    loss = torch.mean(ped_weight * torch.stack(loss_list))
    # loss = torch.mean(torch.stack(loss_list))
    return loss


def get_loss2(x, y):
    mse = torch.mean((x - y) ** 2)
    return mse
