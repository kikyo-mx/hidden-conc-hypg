from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class EOD_data(Dataset):
    def __init__(self, root_path, flag='train', market='NASDAQ', PreConcept=None, tickers=None, seq_len=4,
                 col_train=None, col_label=None, steps=1, rr_num=[0, 1, 2]):
        # init
        assert flag in ['train', 'test', 'val']
        self.col_train = col_train
        self.col_label = col_label
        self.steps = steps
        self.rr_num = rr_num
        self.seq_len = seq_len
        self.root_path = root_path
        self.PreConcept = PreConcept
        self.market = market
        self.tickers = tickers
        self.__read_data__()

    def __read_data__(self):
        for index, ticker in enumerate(self.tickers):
            single_EOD = np.genfromtxt(
                os.path.join(self.root_path, self.market + '_' + ticker + '.csv'),
                dtype=np.float32, delimiter=',', skip_header=False)
            if self.market == 'NASDAQ':
                single_EOD = single_EOD[1:, [0, 1, 2, 3, 4, 5, 9, 6, 7, 8]]
            if index == 0:
                print('single EOD data shape:', single_EOD.shape)  # days*6
                self.eod_data = np.zeros([len(self.tickers), single_EOD.shape[0], len(self.col_train)],
                                         dtype=np.float32)
                self.masks = np.ones([len(self.tickers), single_EOD.shape[0]], dtype=np.int8)
                self.ground_truth = np.zeros([len(self.tickers), single_EOD.shape[0], len(self.col_label)],
                                             dtype=np.float32)
            for row in range(single_EOD.shape[0]):
                if abs(single_EOD[row][-1] + 1234) < 1e-8:
                    self.masks[index][row] = 0.0
                for col in range(single_EOD.shape[1]):
                    if abs(single_EOD[row][col] + 1234) < 1e-8:
                        single_EOD[row][col] = 1
            self.eod_data[index, :, :] = single_EOD[:, 1:7]
            self.ground_truth[index, :, :] = single_EOD[:, 7:][:, self.rr_num]

    def __getitem__(self, index):
        mask_batch = self.masks[:, index: index + self.seq_len + 1]
        mask_batch = np.min(mask_batch, axis=1)
        seq_x = self.eod_data[:, index:index + self.seq_len, :]
        seq_y = self.ground_truth[:, index + self.seq_len]
        seq_PreC = self.PreConcept[index + self.seq_len]
        return seq_x, mask_batch, seq_y, seq_PreC

    def __len__(self):
        return self.eod_data.shape[1] - self.seq_len
