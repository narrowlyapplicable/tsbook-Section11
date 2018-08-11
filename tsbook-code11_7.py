#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
bootstrap filter to "ArtifitialLocalLevelModel.RData"
===========================================================
# 注意：元スクリプトに倣い、事前分布を時点0とし、本来の時点を＋１シフトして扱う
Created on Sun Jul 29 16:04:29 2018

@author: narrowly
"""

import numpy as np
import pandas as pd
import scipy.stats as sct
import random
import pyper
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# 前処理
np.random.seed(21)
N = 10000 # N_particle

r = pyper.R(use_numpy='True', use_pandas='True')
r("load('data/ArtifitialLocalLevelModel.RData')")
y = r.get('y').astype(np.float64)
t_max = r.get('t_max')
mod = r.get('mod')

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

    
# %% コード11-7 (コード11-1がベース)
# データの整形
y = np.r_[np.nan, y]

# リサンプリング用のインデックス列
k = np.repeat(np.arange(N)[:, np.newaxis], t_max+1, axis=1)


# 事前分布の設定
# 粒子 (実現値)
x = np.zeros(shape=(t_max+1, N))
x[0,:] = np.random.normal(loc=mod["m0"] , scale=np.sqrt(mod["C0"][0][0]) , size=N)


# 粒子(重み)
w = np.zeros(shape=(t_max+1, N))
w[0,:] = np.log(1/N)

# 時間順方向の処理
for tt in np.arange(t_max)+1 :
    # 提案分布：状態方程式を使用。粒子を生成する。
    x[tt,:] = np.random.normal(loc=x[tt-1,:], scale=np.sqrt(mod["V"][0][0]), size=N)
    # 重みを更新。 bootstrap filterなので、観測方程式をそのまま使えば良い。
    w[tt,:] = w[tt-1,:] + sct.norm.logpdf(y[tt], loc=x[tt,:], scale=np.sqrt(mod["V"]))
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
plt.plot(y, marker=".", linestyle="--")
plt.plot(np.mean(x, axis=1), marker=".", linestyle="--")
plt.title('Bootstrap Filter : filtering')
plt.legend(["data", "filtered"], frameon=True, edgecolor="b")
plt.tight_layout(); plt.show()
plt.savefig("result/mean_11-7.png")

# 50%区間のプロット
plt.figure(2)
plt.plot(np.arange(t_max), y, marker=".", linestyle="--")
plt.fill_between(np.arange(t_max), y1=np.percentile(x, 75, axis=1), 
                 y2=np.percentile(x, 25, axis=1), facecolor="b", alpha=0.3)
plt.title('Bootstrap Filter : filtering')
plt.legend(["data", "filtered"], frameon=True, edgecolor="b")
plt.tight_layout(); plt.show()
plt.savefig('result/percentile_11-7.png')