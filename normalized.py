
import os
import numpy as np
import pandas as pd

def std_fun(data):

    to_drop = [(col[0], col[1]) for col in data.columns if col[1] == "likelihood"]
    data_new = data.drop(columns=to_drop)

    xmax=data_new.iloc[:,0::2].max(axis=1)
    ymax=data_new.iloc[:,1::2].max(axis=1)
    xmin=data_new.iloc[:,0::2].min(axis=1)
    ymin=data_new.iloc[:,1::2].min(axis=1)
    
    l_std=[]
    for i in range(len(xmax)):  # get r for each frame
        a=((xmax[i]-xmin[i])**2+(ymax[i]-ymin[i])**2)**0.5
        l_std.append(a)

    xmean=data_new.iloc[:,0::2].mean(axis=1)  #mean(x) of a frame
    ymean=data_new.iloc[:,1::2].mean(axis=1)  #mean(y) of a frame
    #xy_mean=pd.Series(list(zip(xmean, ymean))) 
    data_new.iloc[:, ::2] = data_new.iloc[:, ::2].sub(xmean, axis=0) #x-mean(x) of a frame
    data_new.iloc[:,1::2] = data_new.iloc[:,1::2].sub(ymean, axis=0) #y-mean(y) of a frame
   
    for i,j in enumerate(data_new.columns):
        i=i//2
        data_new[j] = data_new.apply(lambda row: row[j]/l_std[i], axis=1)

    return data_new

class csv_std:
    def __init__(self, file):
        self.file = file
        self.df = pd.read_csv(self.file, header=[1, 2], index_col=0)
        self.z_score=self.df
        self.max_frame=30

    def to_std(self):
        self.z_score = std_fun(self.df)

    def re_rows(self):
        interval = len(self.z_score) // self.max_frame
        self.z_score = self.z_score.iloc[::interval].head(self.max_frame)

    def save(self, filename):
        #output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.file))),filename)
        self.z_score.to_csv(filename, index=False) #without saving index column
       

    def auto_run(self, filename):
        self.auto_md()
        self.save(filename)

    def auto_md(self):
        self.to_std()
        self.re_rows()
       
        return self.z_score
