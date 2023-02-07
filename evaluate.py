import torch


# def evaluate(pred, ground_truth, top_num, rr_num):
#     performance = {}
#     sharp = []
#     for j in range(len(rr_num)):
#         pred_top5 = torch.zeros((pred.shape[0], 5))
#         pred_one = torch.argsort(pred[:, :, j], dim=1, descending=True)
#         for i in range(top_num):
#             topN = torch.where(pred_one == i)
#             pred_top5[:, i] = ground_truth[:, :, 0][topN[0], topN[1]]
#         sharp.append(torch.mean(torch.mean(pred_top5, dim=1)) / torch.mean(torch.std(pred_top5, dim=1)))
#     performance['sharp'] = sharp
#     return performance


def evaluate(pred, ground_truth, top_num, rr_num):
    performance = {}
    sharp = torch.zeros(len(rr_num))
    irr = torch.zeros(len(rr_num))
    for j in range(len(rr_num)):
        pred_top5 = torch.zeros(top_num)
        pred_one = torch.argsort(pred[:, j], dim=0, descending=True)
        for i in range(top_num):
            topN = torch.where(pred_one == i)
            pred_top5[i] = ground_truth[:, j][topN[0][0]]
        sharp[j] = torch.mean(pred_top5) / torch.std(pred_top5) if torch.abs(torch.std(pred_top5)) > 1e-2 else torch.mean(pred_top5) / 1e-2
        irr[j] = torch.mean(pred_top5)
    performance['sharp'] = sharp
    performance['irr'] = irr
    return performance


def get_correlation(x, y):
    return - torch.corrcoef(torch.cat([x.unsqueeze(0), y.unsqueeze(0)]))[0][1]


def get_loss1(pred, ground_truth, loss_weight, rr_num):
    loss_list = []
    for i in rr_num:
        loss_rr = get_correlation(pred[:, i], ground_truth[:, i])
        loss_list.append(loss_rr)
    loss = torch.mean(loss_weight * torch.stack(loss_list))
    # loss = torch.mean(torch.stack(loss_list))
    return loss


def get_loss2(pred, ground_truth, loss_weight, rr_num, top_num):
    loss_list = []
    for i in rr_num:
        pred_sort = torch.sort(pred[:, i], dim=0, descending=True)[1]
        gt_sort = torch.sort(ground_truth[:, i], dim=0, descending=True)[1]
        pred_top = pred_sort[:top_num]
        gt_top = gt_sort[:top_num]
        loss_rr = get_correlation(pred_top, gt_top)
        loss_list.append(loss_rr)
    loss = torch.mean(loss_weight * torch.stack(loss_list))
    return loss