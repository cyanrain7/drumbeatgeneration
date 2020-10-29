import os
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt


class Dataset(object):
    def __init__(self,batch_size,path):
        self.dataname = "Groove"
        # self.x_dims = 32*27
        self.c_dims = 7
        self.shape = [32 ,27]
        self.data_x,self.data_y,self.data_label = self.load_dataset(path)
        self.train_x,self.train_y,self.train_label,self.test_x,self.test_y,self.test_label=self.load_pretrain_dataset(8192)
        self.length= self.data_x.shape[0]
        self.batch_size=batch_size
    
    def load_dataset(self,path):
        data=np.load(path)
        
        X=data[:,:-1].reshape(data.shape[0],self.shape[0],self.shape[1])
        Label=data[:,-1].astype('int8')

        seed = 1025
        np.random.seed(seed)
        index = np.arange(data.shape[0])
        np.random.shuffle(index)
        X=X[index]
        x=X[:,:,:9]
        y=X
        Label=Label[index]

        return x,y,Label

    def next_batch(self, iter_num=0):
        rota_num = self.length/self.batch_size - 1
        
        if iter_num % rota_num == 0:
            index = np.arange(self.length)
            np.random.shuffle(index)
            self.data_x = self.data_x[index]
            self.data_y = self.data_y[index]
            self.data_label = self.data_label[index]

        start=int(iter_num % rota_num) * self.batch_size
        end=int(iter_num % rota_num+1) * self.batch_size
        return self.data_x[start:end],self.data_y[start:end], self.data_label[start:end]

    def load_pretrain_dataset(self,seg_num):
        return self.data_x[:seg_num],self.data_y[:seg_num],self.data_label[:seg_num],self.data_x[seg_num:],self.data_y[seg_num:],self.data_label[seg_num:]

    def next_pretrain_batch(self, iter_num=0):
        length=len(self.train_x)
        rota_num = length/self.batch_size - 1
        
        if iter_num % rota_num == 0:
            index = np.arange(length)
            np.random.shuffle(index)
            self.train_x = self.train_x[index]
            self.train_y = self.train_y[index]
            self.train_label = self.train_label[index]
        start=int(iter_num % rota_num) * self.batch_size
        end=int(iter_num % rota_num+1) * self.batch_size
        return self.train_x[start:end], self.train_y[start:end], self.train_label[start:end]

    def next_pretest_batch(self, iter_num=0):
        length=len(self.test_x)
        rota_num = length/self.batch_size - 1
        
        if iter_num % rota_num == 0:
            index = np.arange(length)
            np.random.shuffle(index)
            self.test_x = self.test_x[index]
            self.test_y = self.test_y[index]
            self.test_label = self. test_label[index]
        start=int(iter_num % rota_num) * self.batch_size
        end=int(iter_num % rota_num+1) * self.batch_size
        return self.test_x[start:end], self.test_y[start:end],self.test_label[start:end]
