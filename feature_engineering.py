import pandas as pd
import numpy as np
import asyncio
import os,sys,datetime

import scipy as sp
import statsmodels.tsa.stattools as sts
import statsmodels.api as sm
import matplotlib.pyplot as plt
import talib
import pickle

import xgboost as xgb

from sklearn import metrics
from sklearn.impute import KNNImputer
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

#用于生成特征的函数
class my_features_functions:

    def __init__(self):
        #print(self.__str__())
        return

    #计算价量相关性
    def p2vol(self,open_price,high_price,low_price,close_price,vol):
        price=np.array(close_price)
        vol=np.array(vol)
        return np.corrcoef(price,vol)[0][1]

    #计算高价对低价的回归系数
    def low2high(self,open_price,high_price,low_price,close_price,vol):
        low=np.array(low_price)
        high=np.array(high_price)

        X=sm.add_constant(low)
        model = sm.OLS(high,X)
        results = model.fit()
        return results.params[1]

    #低成交量动量-高成交动量
    def lowmom_highrev(self,open_price,high_price,low_price,close_price,vol):
        df=pd.DataFrame([open_price,close_price,vol],index=["open","close","vol"]).T
        df.sort_values(by="vol",inplace=True)
        df["ret"]=df["close"]/df["open"]-1
        count_num=int(len(df)/4)
        count_num=max(2,count_num)
        return (df.head(count_num)["ret"].mean()-df.tail(count_num)["ret"].mean())

    #eff
    def trend_eff(self,open_price,high_price,low_price,close_price,vol):
        df=pd.DataFrame([open_price,high_price,low_price,close_price],index=["open","high","low","close"]).T
        total_chg=df["close"].iloc[-1]/df["open"].iloc[0]-1
        df["abs_chg"]=np.abs(df["close"]/df["open"]-1)
        return total_chg/df["abs_chg"].sum()

    #最高值距离开盘价
    def max_high_open(self,open_price,high_price,low_price,close_price,vol):
        df=pd.DataFrame([open_price,high_price,low_price,close_price],index=["open","high","low","close"]).T
        return df["high"].max()/df["open"].iloc[0]-1

    #最低值距离开盘价
    def min_low_open(self,open_price,high_price,low_price,close_price,vol):
        df=pd.DataFrame([open_price,high_price,low_price,close_price],index=["open","high","low","close"]).T
        return df["low"].min()/df["open"].iloc[0]-1

    #最大波动比开收盘价波动
    def maxvola(self,open_price,high_price,low_price,close_price,vol):
        df=pd.DataFrame([open_price,high_price,low_price,close_price],index=["open","high","low","close"]).T
        maxvolatility=df["high"].max()/df["low"].min()-1
        close2open=df["close"].iloc[-1]/df["open"].iloc[0]-1
        if close2open!=0:
            return maxvolatility/close2open-1
        else:
            return maxvolatility

    #上涨k线占比
    def up_percent(self,open_price,high_price,low_price,close_price,vol):
        df=pd.DataFrame([open_price,high_price,low_price,close_price],index=["open","high","low","close"]).T
        up_count=len(df[df["close"]>df["open"]])
        down_count=len(df[df["close"]<df["open"]])
        if (up_count+down_count)>0:
            return (up_count-down_count)/(up_count+down_count)
        else:
            return 0

    #交易量倒数加权均值
    def vol_weighted_ma(self,open_price,high_price,low_price,close_price,vol):
        df=pd.DataFrame([open_price,high_price,low_price,close_price,vol],index=["open","high","low","close","vol"]).T
        df["vol_reverse"]=1/np.log(df["vol"]+1)
        df["weight"]=df["vol_reverse"]/(df["vol_reverse"].sum())
        weight_ma=(df["close"]*df["weight"]).sum()/(df["weight"].sum())
        return df["close"].iloc[-1]/weight_ma-1

    #普通均值
    def normal_ma(self,open_price,high_price,low_price,close_price,vol):
        return np.mean(np.array(close_price))

    #收盘价距离最低价
    def close2_minlow(self,open_price,high_price,low_price,close_price,vol):
        df=pd.DataFrame([open_price,high_price,low_price,close_price],index=["open","high","low","close"]).T
        return df["close"].iloc[-1]/np.min(low_price)-1

    #收盘价距离最高价
    def close2_maxhigh(self,open_price,high_price,low_price,close_price,vol):
        df=pd.DataFrame([open_price,high_price,low_price,close_price],index=["open","high","low","close"]).T
        return df["close"].iloc[-1]/np.max(high_price)-1
    
    #提供所有函数列表
    def get_all_methold():
        method_list=[]
        for func in my_features_functions.__dict__:
            method_list.append(func)
        #print(method_list)
        my_func_list=filter(lambda m: not m.startswith("__") and not m.endswith("__"),method_list)
        my_func_list=list(my_func_list)
        my_func_list.remove("get_all_methold")
        return my_func_list

#利用时间序列数据(K线)生成滚动特征
class feature_generator(BaseEstimator, TransformerMixin):
    """
    sklearn库主要使用numpy数组, 所以将dataframe全部转化为numpy数组.
    初始化时添加所需要的函数
    """
    def __init__(self, func_obj,rolling_window=20):
        self.func_obj=func_obj
        self.rolling_window=rolling_window

    def fit(self, open_price,high_price,low_price,close_price,vol, y=None):
        return self

    def transform(self,df):
        """返回numpy数组"""
        result=df["close"].rolling(window=self.rolling_window). \
        apply(lambda x:self.func_obj(df.loc[x.index,"open"], \
                                     df.loc[x.index,"high"],df.loc[x.index,"low"], \
                                     df.loc[x.index,"close"],df.loc[x.index,"volume"]))
        return np.array(result)

class feature_engineering():
    
    def __init__(self,func_list,rolling_window=10):
        self.func_list=func_list
        self.rolling_window=10
    
    def generate_transform_list(self,func_list,feature_generator):
        my_transform_list=[]
        name_list=[]
        for func in func_list:
            my_transform_list.append([func.__name__,feature_generator(func,self.rolling_window)])
            name_list.append(func.__name__)
        return my_transform_list,name_list
    
    def output_feature(self,data):
        #获取所有特征工程函数
        my_transform_list,name_list=self.generate_transform_list(self.func_list,feature_generator)
        features_pipline = FeatureUnion(transformer_list=my_transform_list,n_jobs=-1)
        pip_result=features_pipline.transform(data)
        pip_df_result=pd.DataFrame(pip_result.reshape(len(data),len(self.func_list)),columns=name_list)
        return pip_df_result


