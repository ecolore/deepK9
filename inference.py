# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:03:06 2023

@author: joseph@艾鍗學院

I try to make it to more organized and clear.


https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/standardDeepLabCut_UserGuide.md


pip install deeplabcut[gui]


"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications.resnet import decode_predictions

from normalized import csv_std

Model=r"tok.h5"
#The project_path need to modify manually to fit your situation
# project_path=r'E:\DeepLabCut\topic-joseph-2023-10-01'  
# config_path=os.path.join(project_path,'config.yaml')
# video_path= os.path.join(project_path,'videos')

# print('project_path:',project_path)
# print('config_path:',config_path)
# print('video_path:',video_path)

def predict(t_df):
    global Model
    
    t_df = t_df.iloc[:,:40].values.astype(float)
    t_df=t_df[np.newaxis,:,:]

    model = tf.keras.models.load_model(Model)
    m_ans=model.predict(t_df)
    idx=np.argmax(m_ans)
    

    return idx,m_ans[0][idx]



# s_name=f'mydog123DLC_resnet50_topicOct1shuffle1_75000.csv'
# unfilter_csv = os.path.join(video_path,s_name)
# std_csv = csv_std(unfilter_csv)
# std_data=std_csv.auto_md()
# print(std_data.head())
# cls,conf=predict(model,std_data)
# print('Predict {} (conf:{:.2f})'.format(cls,conf))



