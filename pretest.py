# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 13:56:30 2023

@author: chens
"""
import pandas as pd
import numpy as np


class Ts2Reg():
    """
    def transform(df,window=30,pred_steps=1):
        df = pd.Series(df)
        assert len(df) > (window+pred_steps), "the length of timeseries must be greater than the length of the time (window+pred_stpes)"
        
        n = len(df)-window
        x_c = []
        y_c = []
        x_lst = df.iloc[n:(n+window)].tolist()
        for i in range(n-pred_steps+1):
            x_tmp = df.iloc[i:(i+window)].tolist()
            y_tmp = df.iloc[(i+window):(i+window+pred_steps)]
            x_c.append(x_tmp)
            y_c.append(y_tmp)
        
        #依次返回: x,y,最后一个时间周期的取值(window,用于预测下一期)
        return np.array(x_c),np.array(y_c),np.array(x_lst).reshape(1,-1)
    
    """
    def transform(df,window=31):
        df = pd.Series(df)
        assert len(df) > window, "the length of timeseries must be greater than the length of the time window"
        
        n = len(df)-window
        x_c = []
        y_c = []
        x_lst = df.iloc[n:(n+window)].tolist()
        for i in range(n):
            x_tmp = df.iloc[i:(i+window)].tolist()
            y_tmp = df.iloc[(i+window)]
            x_c.append(x_tmp)
            y_c.append(y_tmp)
        
        #依次返回: x,y,最后一个时间周期的取值(window,用于预测下一期)
        return np.array(x_c),np.array(y_c),np.array(x_lst).reshape(1,-1)
    

class threesigma():
    def three_sigma(s):
        mean, std = np.mean(s), np.std(s)
        lower, upper = mean-3*std, mean+3*std
        return lower, upper
        
    def ts_3sigma(true,pred):
        sigma = np.std(true[1:]-true[:-1])/true.mean()
        threesigma = 3*sigma
        upper = pred*(1+threesigma)
        lower = pred*(1-threesigma)
        return upper,lower
    
        
    
    
    
