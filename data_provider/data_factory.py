import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from utils.timefeatures import time_features

class Forecasting_Dataset(Dataset):
    def __init__(self, datatype, mode="train", history_length=72, label_length=48):
            
        self.val_sample = 5
        if datatype=='electricity':
            dataset = get_dataset("electricity_nips", regenerate=False)
            # 168
            self.history_length = history_length
            self.pred_length = 24
            self.test_sample = 7
            freq = 'h'
        elif datatype=='exchange':
            # 요거 체크
            dataset = get_dataset("exchange_rate_nips", regenerate=False)
            self.history_length = history_length
            self.pred_length = 30
            self.test_sample = 7
            freq = 'h'
        elif datatype=='traffic':
            dataset = get_dataset("traffic_nips", regenerate=False)
            self.history_length = history_length
            self.pred_length = 24
            self.test_sample = 7
            freq = 'h'
        elif datatype=='solar':
            dataset = get_dataset("solar_nips", regenerate=False)
            self.history_length = history_length
            self.pred_length = 24
            self.test_sample = 7
            freq = 'h'
        elif datatype=='wiki':    
            dataset = get_dataset("wiki2000_nips", regenerate=False)
            self.history_length = history_length
            self.pred_length = 30
            self.test_sample = 5
            freq = 'd'
        elif datatype=='taxi':    
            dataset = get_dataset("taxi_30min", regenerate=False)
            self.history_length = history_length
            self.pred_length = 24
            self.test_sample = 56
            freq = '30T'
        
        self.test_length= self.pred_length*self.test_sample
        self.valid_length = self.pred_length*self.val_sample
        
        self.seq_length = self.history_length + self.pred_length
        self.label_length = label_length
            
        train_grouper = MultivariateGrouper(max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))
        dataset_train = train_grouper(dataset.train)

        self.main_data = np.transpose(dataset_train[0]['target'])
        self.date_data = pd.DataFrame([dataset_train[0]['start']+i for i in range(len(self.main_data))]).values
        test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                        max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))


        record_hist = self.seq_length
        ###########
        # record_hist = 192
        ###########
        dataset_test = test_grouper(dataset.test)
        result_test = []
        result_test_date = []
        for test in dataset_test:
            result_test.append(np.expand_dims(test['target'],0))
            result_test_date.append(pd.DataFrame([test['start']+i for i in range(test['target'].shape[-1])]).values)

        dataset_test = np.concatenate(result_test,axis=0).transpose(0,2,1)[:,-record_hist:]
        dataset_test_date = np.stack(result_test_date)[:,-record_hist:]

        concat_result = []
        concat_result_date = []
        concat_result.append(dataset_test[0])
        concat_result_date.append(dataset_test_date[0])
        for cand, cand_date in zip(dataset_test[1:],dataset_test_date[1:]):
            concat_result.append(cand[-self.pred_length:])
            concat_result_date.append(cand_date[-self.pred_length:])
        concat_result = np.concatenate(concat_result, axis=0)
        concat_result_date = np.concatenate(concat_result_date, axis=0)
        self.main_data = np.concatenate([self.main_data[:-int(record_hist-self.pred_length)],concat_result],axis=0)
        self.date_data = np.concatenate([self.date_data[:-int(record_hist-self.pred_length)],concat_result_date],axis=0).squeeze()

        self.date_data = pd.PeriodIndex(self.date_data, freq=freq).to_timestamp().values
        self.date_data = time_features(pd.to_datetime(self.date_data), freq=freq).transpose(1, 0)
        self.mean_data = np.mean(self.main_data, axis=0)
        self.std_data = np.std(self.main_data, axis=0)
        self.std_data = np.clip(self.std_data,a_min=0.001,a_max=1e+7)
        
        if datatype == 'electricity':
            datafolder = './data/electricity_nips'
            self.test_length= 24*7
            self.valid_length = 24*5
                        
            paths=datafolder+'/data.pkl' 
            #shape: (T x N)
            #mask_data is usually filled by 1
            with open(paths, 'rb') as f:
                self.main_data, self.mask_data = pickle.load(f)
            paths=datafolder+'/meanstd.pkl'
            with open(paths, 'rb') as f:
                self.mean_data, self.std_data = pickle.load(f)
        
        self.main_data = (self.main_data - self.mean_data) / self.std_data
        total_length = len(self.main_data)
        
        if mode == 'train': 
            start = 0
            end = total_length - self.seq_length - self.valid_length - self.test_length + 1
            self.use_index = np.arange(start,end,1)
        if mode == 'val': #valid
            start = total_length - self.seq_length - self.valid_length - self.test_length + self.pred_length
            end = total_length - self.seq_length - self.test_length + self.pred_length
            self.use_index = np.arange(start,end,self.pred_length)
        if mode == 'test': #test
            start = total_length - self.seq_length - self.test_length + self.pred_length
            end = total_length - self.seq_length + self.pred_length
            self.use_index = np.arange(start,end,self.pred_length)
        
    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]
        s = [self.main_data[index:index+self.history_length],
             self.main_data[index+self.history_length-self.label_length:index+self.seq_length],
             self.date_data[index:index+self.history_length],
             self.date_data[index+self.history_length-self.label_length:index+self.seq_length]]

        return s
    def __len__(self):
        return len(self.use_index)
    
    def inverse_transform(self, data):
        denormed = data * self.std_data
        denormed += self.mean_data
        return denormed
        

def data_provider(args, flag):
    datatype = args.data
    history_length = args.seq_len
    batch_size = args.batch_size
    if flag == 'test':
        batch_size = 1
        shuffle = 0
    else:
        shuffle = 1
        
    data_set = Forecasting_Dataset(datatype,mode=flag,history_length=history_length,label_length=args.label_len)

    data_loader = DataLoader(
        data_set, batch_size=batch_size, shuffle=shuffle)

    return data_set, data_loader