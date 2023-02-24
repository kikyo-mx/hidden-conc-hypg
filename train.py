import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn
import numpy as np
from evaluate import evaluate
from evaluate import get_loss1, get_loss2
from dataloader import EOD_data
from models.HGAT import HGAT
import os
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from datetime import datetime

seed = 123456789
np.random.seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# pre_config
device = 'cuda:0'
log_path = '/home/kikyo/code/hidden-conc-hypg/log/'
writer = SummaryWriter(log_path)
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_path])
url = tb.launch()

# os.remove(log_path + os.listdir(log_path)[0])
market_name = 'NASDAQ'
data_path = '/home/kikyo/data/qt/rr8_volumn'
col_train = ['rr', 'rr5', 'rr10', 'rr20', 'rr30', 'volumn']
col_label = ['rr-1', 'rr-5', 'rr-30']
ecod_in = len(col_train)
inci_mat = np.load('hg_none1.npy')
tickers = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
tickers = np.genfromtxt(os.path.join(data_path, '..', tickers),
                        dtype=str, delimiter='\t', skip_header=False)
PreConcept = torch.from_numpy(np.load('/home/kikyo/data/qt/PreConceptWeight_none1.npy')).to(device)
hyp_input = torch.from_numpy(inci_mat).to(device)

# hyper parameters
seq_len = 8
steps = 1
lr = 0.01
top_num = 5
epochs = 70
d_model = 128
rr_num = [0, 1, 2]
use_PreAttn = True
use_HidAttn = True
loss_weight = torch.ones(len(rr_num)).to(device)
flag = ['train', 'test', 'val']

# load data
data_set = EOD_data(root_path=data_path, PreConcept=PreConcept, flag=flag[1], tickers=tickers, col_train=col_train,
                    col_label=col_label, seq_len=seq_len)
train_size = int(len(data_set) * 0.5)
validate_size = int(len(data_set) * 0.3)
test_size = len(data_set) - train_size - validate_size
train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(data_set,
                                                                              [train_size, validate_size, test_size])
train_loader = DataLoader(train_dataset, shuffle=False, drop_last=True)
validate_loader = DataLoader(validate_dataset, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True)

print(len(data_set), 'data_set')
print(len(train_loader), 'train_loader')

# load model
model = HGAT(d_model=d_model, ecod_in=ecod_in, seq_len=seq_len, rr_num=rr_num, use_PreAttn=use_PreAttn, use_HidAttn=use_HidAttn, Hedge_index=inci_mat[1], device=device).to(device)
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
    else:
        torch.nn.init.uniform_(p)

optimizer_hgat = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_hgat, gamma=0.3, milestones=[15, 40])
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_hgat, mode='min', factor=0.2, patience=5, threshold=1e-4)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer_hgat, step_size=10, gamma=0.1, last_epoch=-1)


# train
loop1 = tqdm(range(epochs), total=epochs)
for epoch in loop1:
    start_time = time.time()
    train_loss = []
    # rank_loss_train = []
    model.train()
    for i, (batch_x, batch_mask, batch_y, batch_Pre) in enumerate(train_loader):
        # print(i, len(train_loader), datetime.now().strftime("%H:%M:%S"))
        optimizer_hgat.zero_grad()
        output = model(batch_x.squeeze(0).to(device), hyp_input, batch_Pre[0])
        mask = batch_mask.to(device).bool().squeeze(0)
        pred = output * mask.unsqueeze(1)
        ground_truth = batch_y.to(device).squeeze(0) * mask.unsqueeze(1)
        # loss_rank_train = get_loss2(pred, ground_truth, loss_weight, rr_num, top_num)
        loss_train = get_loss1(pred, ground_truth, loss_weight, rr_num)
        train_loss.append(loss_train.cpu().detach().numpy())
        # rank_loss_train.append(loss_rank_train.cpu().detach().numpy())
        loss_train.backward()
        optimizer_hgat.step()
    train_loss = np.average(train_loss)
    # scheduler1.step()
    scheduler2.step(train_loss)
    # scheduler3.step()
    writer.add_scalar('tra_loss', train_loss, global_step=epoch)
    # writer.add_scalar('tra_rank_loss', rank_loss_train, global_step=epoch)

    # validate
    # print('start_val', datetime.now().strftime("%H:%M:%S"))
    with torch.no_grad():
        val_loss = []
        # rank_loss_val = []
        val_per = {'sharp': torch.zeros(len(validate_loader), len(rr_num)),
                           'irr': torch.zeros(len(validate_loader), len(rr_num)),
                           'rank_score': torch.zeros(len(validate_loader), len(rr_num))}
        for i, (batch_x, batch_mask, batch_y, batch_Pre) in enumerate(validate_loader):
            # print(i, len(validate_loader), time.time())
            output = model(batch_x.squeeze(0).to(device), hyp_input, batch_Pre[0])
            mask = batch_mask.to(device).bool().squeeze(0)
            pred = output * mask.unsqueeze(1)
            ground_truth = batch_y.to(device).squeeze(0) * mask.unsqueeze(1)
            loss_val = get_loss1(pred, ground_truth, loss_weight, rr_num)
            # loss_rank_val = get_loss2(pred, ground_truth, loss_weight, rr_num, top_num)
            val_loss.append(loss_val.cpu().detach().numpy())
            # rank_loss_val.append(loss_rank_val.cpu().detach().numpy())
            val_per['sharp'][i], val_per['irr'][i], val_per['rank_score'][i] = evaluate(pred, ground_truth, top_num, rr_num)
        # val_per['sharp'] = torch.mean(val_per['sharp'][~torch.all(val_per['sharp'] == 0, dim=1)], dim=0)
        val_per['sharp'] = torch.mean(val_per['sharp'], dim=0)
        val_per['irr'] = torch.mean(val_per['irr'], dim=0)
        val_per['rank_score'] = torch.mean(val_per['rank_score'], dim=0)
        val_loss = np.average(val_loss)
        # rank_loss_val = np.average(rank_loss_val)
    end_time = time.time()
    # print('end', datetime.now().strftime("%H:%M:%S"))
    writer.add_scalar('val_loss', val_loss, global_step=epoch)
    # writer.add_scalar('val_rank_loss', rank_loss_val, global_step=epoch)
    for j in rr_num:
        writer.add_scalar('val_sharp' + str(j), val_per['sharp'][j], global_step=epoch)
        writer.add_scalar('val_irr' + str(j), val_per['irr'][j], global_step=epoch)
        writer.add_scalar('rank_score' + str(j), val_per['rank_score'][j], global_step=epoch)
    loop1.set_description(f'Epoch [{epoch + 1}/{epochs}] Train Performance')
    loop1.set_postfix(tra_loss=format(train_loss, '.5f'), val_loss=format(val_loss, '.5f'),
                      val_sharp=val_per['sharp'], val_irr=val_per['irr'],
                      val_rank=val_per['rank_score'], time=format(end_time - start_time, '.2f'))

# test
with torch.no_grad():
    test_loss = []
    test_per = {'sharp': torch.zeros(len(test_loader), len(rr_num)),
                       'irr': torch.zeros(len(test_loader), len(rr_num)),
                       'rank_score': torch.zeros(len(test_loader), len(rr_num))}
    for i, (batch_x, batch_mask, batch_y, batch_Pre) in enumerate(test_loader):
        output = model(batch_x.squeeze(0).to(device), hyp_input, batch_Pre[0])
        mask = batch_mask.to(device).bool().squeeze(0)
        pred = output * mask.unsqueeze(1)
        ground_truth = batch_y.to(device).squeeze(0) * mask.unsqueeze(1)
        # loss_test = get_loss1(pred, ground_truth, loss_weight, rr_num)
        loss_test = get_loss2(pred, ground_truth, loss_weight, rr_num, top_num)
        test_loss.append(loss_test.cpu().detach().numpy())
        test_per['sharp'][i], test_per['irr'][i], test_per['rank_score'][i] = evaluate(pred, ground_truth, top_num, rr_num)
    test_per['sharp'] = torch.mean(test_per['sharp'], dim=0)
    test_per['irr'] = torch.mean(test_per['irr'], dim=0)
    test_per['rank_score'] = torch.mean(test_per['rank_score'], dim=0)
    test_loss = np.average(test_loss)
print('[test_loss:{:.6} test_irr:{} test_sharp:{} test_rank:{}]'.format(test_loss, test_per['irr'], test_per['sharp'], test_per['rank_score']))
