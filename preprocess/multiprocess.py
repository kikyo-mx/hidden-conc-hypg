from multiprocessing import Pool
import argparse
from datetime import datetime
import json
import numpy as np
import operator
import os
import pandas as pd

market = 'NYSE'
data_path_eod = '/home/kikyo/data/qt/google_finance/'
data_path_attn = '/home/kikyo/data/qt/attention_data_' + market + '/'
hpg = np.load('/home/kikyo/code/hidden-conc-hypg/hg_' + market + '.npy')
data_path_rr = '/home/kikyo/data/qt/rr8_volumn_' + market + '/'
date_format = '%Y-%m-%d %H:%M:%S'
tickers = '/home/kikyo/data/qt/' + market + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
tickers = np.genfromtxt(tickers, dtype=str, delimiter='\t', skip_header=False)
begin_time = datetime.strptime('2012-11-19 00:00:00', date_format)
col = ['rr', 'rr5', 'rr10', 'rr20', 'rr30', 'volumn', 'rr-1', 'rr-5', 'rr-30']


def attn_data(ticker):
    single_EOD0 = np.genfromtxt(os.path.join(data_path_eod, market + '_' + ticker + '_30Y.csv'), dtype=str,
                                delimiter=',', skip_header=True)
    single_data = []
    for day in range(len(single_EOD0)):
        day_time = datetime.strptime(single_EOD0[day][0].split(' ')[0] + ' 00:00:00', date_format)
        if day_time >= begin_time and day + 30 < len(single_EOD0):
            data_dict = {}
            single_EOD = single_EOD0[:, 1:].astype(np.float64)
            days_weight = single_EOD[day][0] * single_EOD[day][4]
            data_dict['hyper_weight'] = days_weight if days_weight != 0 else -1234
            single_data.append(data_dict)
    single_data = pd.DataFrame(single_data)
    if len(single_data) < 1244:
        row = np.full((1244 - single_data.shape[0], 1), -1234)
        index = range(single_data.shape[0], 1244)
        pd_1234 = pd.DataFrame(row, index=index, columns=['hyper_weight'])
        single_data = pd.concat([single_data, pd_1234])
    elif len(single_data) > 1244:
        single_data = single_data.drop(index=range(1244, single_data.shape[0]))
    single_data.to_csv(data_path_attn + market + '/' + ticker + '.csv')


def attn_weight(row):
    attention_weight = np.zeros((1, hpg.shape[1]))
    for i in range(np.max(hpg[1])):
        hyper_index = np.where(hpg[1] == i)
        tickers_rel = tickers[hpg[0][hyper_index]]
        weight_list = []
        for ticker in tickers_rel:
            data_weight_path = pd.read_csv(data_path_attn + ticker + '.csv', index_col=0)
            amount = data_weight_path['hyper_weight'][row]
            weight_list.append(amount) if amount > -1000 else weight_list.append(0)
        hyper_attn = np.array(weight_list) / sum(weight_list) if sum(weight_list) > 1 else np.array(weight_list)
        attention_weight[0, hyper_index[0][0]:hyper_index[0][0] + len(hyper_index[0])] = hyper_attn
    np.save('/home/kikyo/data/qt/ConceptWeightList/' + str(row) + '.npy', attention_weight)


row_list = range(1244)
pool = Pool(processes=64)
print('start')
pool.map(attn_weight, row_list)
print('finish')