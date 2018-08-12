#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
bootstrap filter to "BenchmarkNonLinearModel.RData"
===========================================================
Created on Sun Aug 12 23:57:03 2018

@author: narrowly
"""

import numpy as np
#import pandas as pd
import scipy.stats as sct
#import random
#import pyper
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# 前処理
np.random.seed(4521)
N = 10000 # N_particle

#r = pyper.R(use_numpy='True', use_pandas='True')
#r("load('data/BenchmarkNonLinearModel.RData')")
#y = r.get('y').astype(np.float64)
#t_max = r.get('t_max')
#m0 = r.get('m0'); C0 = r.get('C0'); W = r.get('W'); V = r.get('V')
#x_true = r.get('x_true')

# %% コード11-5
##【logsumexp】
# 線形領域での規格化(入力：対数値ベクトル, 出力：規格化後の対数値ベクトル)
def normalize(l):
    max_ind = np.argmax(l)
    ind_rm = np.repeat(True, l.shape[0])
    ind_rm[max_ind] = False
    return l - l[max_ind] - np.log1p(np.sum(np.exp(l[ind_rm] - l[max_ind])))

# %% コード11-6
## 【系統リサンプリング】
# <https://qiita.com/kenmatsu4/items/c5232b1499dfd00e877d>を参考に実装
def F_inv(w_cumsum, ind, u):
    if np.any(w_cumsum < u) == False:
        k = 0
    else:
        k = np.max(ind[w_cumsum < u]) + 1
    return k
    
def sys_resampling(N, w):
    w = np.exp(w) # wを線形領域に戻す
    w_cumsum = np.cumsum(w)
    ind = np.arange(N)
    u = np.linspace(0, 1 - (np.random.uniform(size=1)/N)[0], num = N) # 系統リサンプリング
    #u = np.random.uniform(0, 1 - (np.random.uniform(size=1)/N), size=N) # 層化リサンプリング
    k = np.array([F_inv(w_cumsum, ind, val) for val in u])
    return k

# %% コード11-8
W = 1; V = 2; m0 = 10; C0 = 9

# 状態方程式における非線形関数
def f(x, t):
    return (x / 2) + (25 * x / (1 + x**2)) + (8 * np.cos(1.2 * t))

# 観測方程式における非線形関数
def h(x):
    return (x**2) / 20

t_max = 100
# データの初期化
x_true = np.repeat(np.nan, repeats=t_max+1)
y = np.repeat(np.nan, repeats=t_max+1)

x_true[0] = m0
for it in np.arange(t_max)+1:
    x_true[it] = f(x_true[it - 1], it) + np.random.normal(scale=np.sqrt(W), size=1)
    y[it] = h(x_true[it]) + np.random.normal(scale=np.sqrt(V), size=1)

x_true = x_true[1:]
y = y[1:]

# %% コード11-9 (コード11-7がベース)
# データの整形
y = np.r_[np.nan, y]

# リサンプリング用のインデックス列
k = np.repeat(np.arange(N)[:, np.newaxis], t_max+1, axis=1)


# 事前分布の設定
# 粒子 (実現値)
x = np.zeros(shape=(t_max+1, N))
x[0,:] = np.random.normal(loc=m0 , scale=np.sqrt(C0) , size=N)


# 粒子(重み)
w = np.zeros(shape=(t_max+1, N))
w[0,:] = np.log(1/N)

# 時間順方向の処理
for tt in np.arange(t_max)+1 :
    # 提案分布：状態方程式を使用。粒子を生成する。
    x[tt,:] = np.random.normal(loc=f(x=x[tt-1,:], t=tt), scale=np.sqrt(W), size=N)
    # 重みを更新。 bootstrap filterなので、観測方程式をそのまま使えば良い。
    w[tt,:] = w[tt-1,:] + sct.norm.logpdf(y[tt], loc=h(x=x[tt,:]), scale=np.sqrt(V))
    # 重みの規格化
    w[tt,:] = normalize(w[tt,:])
    
    # リサンプリング
    # リサンプリング用のインデックス列
    k[:,tt] = sys_resampling(N = N, w = w[tt,:])
    # 粒子のリサンプリング：リサンプリング後のインデックス列を新たな通番とする。
    x[tt,:] = x[tt, k[:, tt]]
    # 重みのリセット
    w[tt,:] = np.log(1/N)

w = w[1:]; x = x[1:]; k = k[1:]; y = y[1:]

# 各時刻の粒子平均値をプロット
plt.figure(1)
plt.plot(x_true, marker=".", linestyle="--")
plt.plot(np.mean(x, axis=1), marker=".", linestyle="--")
plt.plot(y, marker=".", color="grey", alpha=0.3)
plt.title('Benchmark NonLinear Model : filtering')
plt.legend(["x_true", "filtered", "observed"], frameon=True, edgecolor="b")
plt.tight_layout(); plt.show()
plt.savefig("result/mean_11-9.png")

# 50%区間のプロット
plt.figure(2)
plt.plot(np.arange(t_max), x_true, marker=".", linestyle="--")
plt.fill_between(np.arange(t_max), y1=np.percentile(x, 75, axis=1), 
                 y2=np.percentile(x, 25, axis=1), facecolor="b", alpha=0.3)
plt.plot(y, marker=".", color="grey", alpha=0.3)
plt.title('Benchmark NonLinear Model : filtering')
plt.legend(["x_true", "observed", "filtered"], frameon=True, edgecolor="b")
plt.tight_layout(); plt.show()
plt.savefig('result/percentile_11-9.png')