# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 14:28:51 2023

@author: chens
"""

from alarm.init_config import initialize   
from alarm.load_data import gen_data   
from alarm.detection import fit
from alarm.detection import predict   
from alarm.detection import alarm
from alarm.detection import plot

import pandas as pd
import numpy as np


#初始化配置参数
config=initialize('./config')
#生成数据
data_set=gen_data(data_path=config.data_path,
                  target_name=config.target_name,
                  date_name=config.date_name,
                  model_type=config.model_type,
                  pred_steps=config.pred_steps)
#模型训练
model=fit(data_set.train_dataset,config.model_type,config.params_config_path)
"""
#模型评价指标
model['rf_model']._eval(model['rf_model'],
                             data_set.train_dataset['train_rf'].iloc[:,:31],
                             data_set.train_dataset['train_rf']['y'])
"""
#预测
pred_res=predict(data_set.infer_dataset,data_set.three_sigma,model)
#异常值检测
test=pd.read_csv('./data/aum_daifa_test.txt')  

upper=pred_res['yhat_upper'].iloc[0]
lower=pred_res['yhat_lower'].iloc[0]
true_v=test['y'].iloc[0] #实际值

flag=alarm(upper,lower,true_v)
ds=str(pred_res['ds'].iloc[0]).split(' ')[0]
pred_v=pred_res['yhat'].iloc[0]

print("日期:{}\n预测结果:{}\n上界:{}\n下界:{}\n真值为:{}\n预警结果为{}".format(ds,pred_v,upper,lower,true_v,flag))


