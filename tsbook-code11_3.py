#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
bootstrap filter to "ArtifitialLocalLevelModel.RData"
===========================================================
Created on Wed Aug  1 22:04:25 2018

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
np.random.seed(4521)
N = 10000 # N_particle

r = pyper.R(use_numpy='True', use_pandas='True')
r("load('data/ArtifitialLocalLevelModel.RData')")
y = r.get('y').astype(np.float64)
t_max = r.get('t_max')
mod = r.get('mod')

# %% コード11-1
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
w[0,:] = 1/N

# 時間順方向の処理
for tt in np.arange(t_max)+1 :
    # 提案分布：状態方程式を使用。粒子を生成する。
    x[tt,:] = np.random.normal(loc=x[tt-1,:], scale=np.sqrt(mod["V"][0][0]), size=N)
    # 重みを更新。 bootstrap filterなので、観測方程式をそのまま使えば良い。
    w[tt,:] = w[tt-1,:] * sct.norm.pdf(y[tt], loc=x[tt,:], scale=np.sqrt(mod["V"]))
    # 重みの規格化
    w[tt,:] = w[tt,:] / np.sum(w[tt,:])
    
    # リサンプリング
    # リサンプリング用のインデックス列 (python3.6以降)
    k[:,tt] = random.choices(np.arange(N), k=N, weights=w[tt,:])
    # 粒子のリサンプリング：リサンプリング後のインデックス列を新たな通番とする。
    x[tt,:] = x[tt, k[:, tt]]
    # 重みのリセット
    w[tt,:] = 1/N

w = w[1:]; x = x[1:]; k = k[:, 1:]; y = y[1:]

# =============================================================================
#  ここまでコード11ー1
# =============================================================================
# %% ここからコード11-2
# 未来時刻のデータ領域確保
x = np.r_[x, np.zeros(shape=(10, N))]
w = np.r_[w, np.zeros(shape=(10, N))]

# 時間方向の処理
for tt in np.arange(10)+t_max:
    # 状態方程式から1step先の状態を生成
    x[tt,:] = np.random.normal(loc=x[tt-1,:], scale=np.sqrt(mod["W"][0]), size=N)
    # 粒子の重みはそのまま
    w[tt,:] = w[tt-1,:]

# =============================================================================
# ここまでコード11-2
# =============================================================================
# %% ここからコード11-3
# 未来時刻の情報を考慮し、粒子を再選択する関数
def smoothing_index(t_current):
    index = np.arange(N) # 現時刻t_currentにおけるインデックス列
    for tt in np.arange(t_current, t_max):
        index = index[k[:,tt]]
    return index
# 粒子を再選択
ki = np.array([x[tt, smoothing_index(tt)] for tt in range(0,t_max-1)], dtype=w.dtype)
ki = np.c_[ki.T, x[t_max,:]].T

# 各時刻の平滑化後平均値をプロット
plt.figure(1)
plt.plot(y, marker=".", linestyle="--")
plt.plot(np.mean(ki, axis=1), marker=".", linestyle="--")
plt.title('Bootstrap Filter : smoothing')
plt.legend(["data", "filtered"], frameon=True, edgecolor="b")
plt.tight_layout(); plt.show()

# 50%区間のプロット
plt.figure(2)
plt.plot(np.arange(t_max), y, marker=".", linestyle="--")
plt.fill_between(np.arange(t_max), y1=np.percentile(ki, 75, axis=1), 
                 y2=np.percentile(ki, 25, axis=1), facecolor="b", alpha=0.3)
plt.title('Bootstrap Filter : smoothing')
plt.legend(["data", "filtered"], frameon=True, edgecolor="b")
plt.tight_layout(); plt.show()
