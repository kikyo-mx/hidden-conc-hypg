import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn
import numpy as np
from evaluate import evaluate
from evaluate import get_loss
from dataloader import EOD_data
from models.HGAT import HGAT
import os
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

seed = 123456789
np.random.seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# pre_config
device = 'cuda'
log_path = 'C:\\Users\\kikyo\\code\\qt\\hidden-conc-hypg\\log\\'
writer = SummaryWriter(log_path)
os.remove(log_path + os.listdir(log_path)[0])
market_name = 'NASDAQ'
data_path = 'D:/data/qt/data/rr8_volumn'
col_train = ['rr', 'rr5', 'rr10', 'rr20', 'rr30', 'volumn']
col_label = ['rr-1', 'rr-5', 'rr-30']
ecod_in = len(col_train)
inci_mat = np.load('hg_test.npy')
tickers = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
tickers = np.genfromtxt(os.path.join(data_path, '..', tickers),
                        dtype=str, delimiter='\t', skip_header=False)
PreConcept = torch.from_numpy(np.load(r'D:\data\qt\data\PreConceptWeight.npy')).to(device)
hyp_input = torch.from_numpy(inci_mat).to(device)

# hyper parameters
seq_len = 8
steps = 1
lr = 0.01
top_num = 5
epochs = 50
d_model = 128
rr_num = [0, 1, 2]
ues_attn = False
loss_weight = torch.tensor([1, 1, 1]).to(device)
flag = ['train', 'test', 'val']

# load data
data_set = EOD_data(root_path=data_path, PreConcept=PreConcept, flag=flag[1], tickers=tickers, col_train=col_train,
                    col_label=col_label, seq_len=seq_len)
print(len(data_set), 'data_set')
train_size = int(len(data_set) * 0.7)
validate_size = int(len(data_set) * 0.1)
test_size = len(data_set) - train_size - validate_size
train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(data_set,
                                                                              [train_size, validate_size, test_size])
train_loader = DataLoader(train_dataset, shuffle=False, drop_last=True)
validate_loader = DataLoader(validate_dataset, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True)

# load model
model = HGAT(d_model, ecod_in, seq_len, rr_num, ues_attn).to(device)
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
    else:
        torch.nn.init.uniform_(p)

optimizer_hgat = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_hgat, gamma=0.3, milestones=[15, 40])
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_hgat, mode='min', factor=0.1, patience=5, threshold=1e-4)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer_hgat, step_size=20, gamma=0.1, last_epoch=-1)


# train
loop1 = tqdm(range(epochs), total=epochs)
for epoch in loop1:
    start_time = time.time()
    train_loss = []
    model.train()
    for i, (batch_x, batch_mask, batch_y, batch_Pre) in enumerate(train_loader):
        optimizer_hgat.zero_grad()
        output = model(batch_x.squeeze(0).to(device), hyp_input, batch_Pre[0])
        mask = batch_mask.to(device).bool().squeeze(0)
        pred = output * mask.unsqueeze(1)
        ground_truth = batch_y.to(device).squeeze(0) * mask.unsqueeze(1)
        loss_train = get_loss(pred, ground_truth, loss_weight, rr_num)
        train_loss.append(loss_train.cpu().detach().numpy())
        loss_train.backward()
        optimizer_hgat.step()
    train_loss = np.average(train_loss)
    # scheduler1.step()
    scheduler2.step(train_loss)
    # scheduler3.step()
    writer.add_scalar('tra_loss', train_loss, global_step=epoch)

    # validate
    # val_pre = torch.zeros(len(validate_loader), len(tickers), 3)
    # val_gt = torch.zeros(len(validate_loader), len(tickers), 3)
    with torch.no_grad():
        val_loss = []
        # val_pre = torch.zeros((len(validate_loader), len(tickers), 3))
        # val_gt = torch.zeros((len(validate_loader), len(tickers), 3))
        val_performance = torch.zeros(len(validate_loader), len(rr_num))
        for i, (batch_x, batch_mask, batch_y, batch_Pre) in enumerate(validate_loader):
            output = model(batch_x.squeeze(0).to(device), hyp_input, batch_Pre[0])
            mask = batch_mask.to(device).bool().squeeze(0)
            pred = output * mask.unsqueeze(1)
            ground_truth = batch_y.to(device).squeeze(0) * mask.unsqueeze(1)
            loss_val = get_loss(pred, ground_truth, loss_weight, rr_num)
            val_loss.append(loss_val.cpu().detach().numpy())
            # val_pre[i] = pred.cpu()
            # val_gt[i] = ground_truth.cpu()
            val_sharp = evaluate(pred, ground_truth, top_num, rr_num)['sharp']
            if not torch.isinf(val_sharp).any():
                val_performance[i] = val_sharp
        val_performance = torch.mean(val_performance, dim=0)
        val_loss = np.average(val_loss)
    end_time = time.time()
    writer.add_scalar('val_loss', val_loss, global_step=epoch)
    writer.add_scalar('val_sharp1', val_performance[0], global_step=epoch)
    writer.add_scalar('val_sharp5', val_performance[1], global_step=epoch)
    writer.add_scalar('val_sharp30', val_performance[2], global_step=epoch)
    loop1.set_description(f'Epoch [{epoch + 1}/{epochs}] Train Performance')
    loop1.set_postfix(tra_loss=format(train_loss, '.5f'), val_loss=format(val_loss, '.5f'),
                      val_sharp=val_performance, time=format(end_time - start_time, '.2f'))

# test
with torch.no_grad():
    test_loss = []
    test_performance = torch.zeros(len(test_loader), len(rr_num))
    for i, (batch_x, batch_mask, batch_y, batch_Pre) in enumerate(test_loader):
        output = model(batch_x.squeeze(0).to(device), hyp_input, batch_Pre[0])
        mask = batch_mask.to(device).bool().squeeze(0)
        pred = output * mask.unsqueeze(1)
        ground_truth = batch_y.to(device).squeeze(0) * mask.unsqueeze(1)
        loss_test = get_loss(pred, ground_truth, loss_weight, rr_num)
        test_loss.append(loss_test.cpu().detach().numpy())
        test_sharp =evaluate(pred, ground_truth, top_num, rr_num)['sharp']
        if not torch.isinf(val_sharp).any():
            test_performance[i] = test_sharp
    test_performance = torch.mean(test_performance, dim=0)
    test_loss = np.average(test_loss)
print('[test_loss:{:.6} test_sharp:{}]'.format(test_loss, test_performance))
