#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
Liu and West Filter to "ArtifitialLocalLevelModel.RData"
===========================================================
# 注意：元スクリプトに倣い、事前分布を時点0とし、本来の時点を＋１シフトして扱う
Created on Wed Aug 15 12:59:50 2018

@author: narrowly
"""

import numpy as np
import pandas as pd
import scipy.stats as sct
import random
import pyper
import matplotlib.pyplot as plt
plt.style.use('ggplot')
np.random.seed(4521)

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

# %% コード11-11
## 【カーネル平滑化】
# パラメータに対する人為的な移動平均を実行する
def kernel_smoothing(realization, w, a):
    # wを線形領域に戻す
    w = np.exp(w)
    # 重みづけ平均と分散
    mean_realization = np.average(realization, weights = w)
    var_realization = np.average((realization - mean_realization)**2, weights = w)
    
    # 人為的な移動平均による、平均と分散減少分
    mu = a * realization + (1-a) * mean_realization
    sigma2 = (1 - a**2) * var_realization
    
    return {"mu":mu, "sigma":np.sqrt(sigma2)}
    
# %% コード11-12 (コード11-7がベース)
# 粒子フィルタの事前設定
N = 10000
a = 0.975
W_max = 10 * np.var(np.diff(y))
V_max = 10 * np.var(y)

# データの整形
y = np.r_[np.nan, y]

# 事前分布の設定
# 粒子(実現値) : パラメータW
W = np.zeros(shape = (t_max+1, N))
W[0,:] = np.log(np.random.uniform(low=0, high=W_max, size=N))

# 粒子(実現値) : パラメータV
V = np.zeros(shape=(t_max+1, N))
V[0,:] = np.log(np.random.uniform(low=0, high=V_max, size=N))

# 粒子(実現値) : 状態
x = np.zeros(shape=(t_max+1, N))
x[0,:] = np.random.normal(loc=0, scale=np.sqrt(1e+7) , size=N)

# 粒子(重み)
w = np.zeros(shape=(t_max+1, N))
w[0,:] = np.log(1/N)

# 時間順方向の処理(カーネル平滑化+補助粒子フィルタ)
for tt in np.arange(t_max)+1 :
    # パラメータの人為的な移動平均
    W_ks = kernel_smoothing(realization= W[tt-1,:], w= w[tt-1,:], a=a)
    V_ks = kernel_smoothing(realization= V[tt-1,:], w= w[tt-1,:], a=a)
    
    # リサンプリング(相当)
    # 補助変数列
    probs = w[tt-1,:] + sct.norm.logpdf(y[tt], loc=x[tt-1,:], scale=np.sqrt(np.exp(V_ks["mu"])))
    k = sys_resampling(N=N, w=normalize(probs.reshape(-1)))
    
    # 連続値の提案分布からパラメータの実現値を抽出（リフレッシュ） 
    W[tt,:] = np.random.normal(loc = W_ks["mu"][k], scale=W_ks["sigma"], size=N)
    V[tt,:] = np.random.normal(loc = V_ks["mu"][k], scale=V_ks["sigma"], size=N)
    
    # 状態方程式から粒子(実現値)を生成
    x[tt,:] = np.random.normal(loc=x[tt-1, k], scale=np.sqrt(np.exp(W[tt,:])), size=N)
    
    # 観測方程式：粒子(重み)を更新
    w[tt,:] = sct.norm.logpdf(y[tt], loc=x[tt,:], scale=np.sqrt(np.exp(V[tt,:]))) - sct.norm.logpdf(y[tt], loc = x[tt-1, k], scale = np.sqrt(np.exp(V_ks["mu"][k])))
    
    # 重みの規格化
    w[tt,:] = normalize(w[tt,:])

w = w[1:]; x = x[1:]; y = y[1:]

# 各時刻の粒子平均値をプロット
plt.figure(1)
plt.plot(y, marker=".", linestyle="--")
plt.plot(np.mean(x, axis=1), marker=".", linestyle="--")
plt.title('Liu and West Filter : filtering')
plt.legend(["data", "filtered"], frameon=True, edgecolor="b")
plt.tight_layout(); plt.show()
plt.savefig("result/mean_11-12.png")

# 50%区間のプロット
plt.figure(2)
plt.plot(np.arange(t_max), y, marker=".", linestyle="--")
plt.fill_between(np.arange(t_max), y1=np.percentile(x, 75, axis=1), 
                 y2=np.percentile(x, 25, axis=1), facecolor="b", alpha=0.3)
plt.title('Liu and West Filter : filtering')
plt.legend(["data", "filtered"], frameon=True, edgecolor="b")
plt.tight_layout(); plt.show()
plt.savefig('result/percentile_11-12.png')