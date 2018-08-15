#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================
auxiliary particle filter to "Nile"
=========================================
Created on Tue Aug 14 18:21:00 2018

@author: narrowly
"""

import numpy as np
import pandas as pd
import pyper
import scipy.stats as sct
import matplotlib.pyplot as plt
#import seaborn as sns
plt.style.use('ggplot')
np.random.seed(4521)

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

# %% コード11-10(pypeR)
r = pyper.R(use_numpy='True', use_pandas='True')
r('set.seed(4521)')
# ナイル川の流量データ
r('y <- Nile')
r('t_max <- length(y)')

# ローカルレベルを構築する
r('library(dlm)')
r('build_dlm <- function(par) {dlmModPoly(order=1, dV=exp(par[1]), dW=exp(par[2]))}')

# パラメータの最尤推定
r('fit_dlm <- dlmMLE(y=y, parm=rep(0,2), build=build_dlm)')
r('mod <- build_dlm(fit_dlm$par)')

t_max = r.get('t_max')
y = r.get('y').astype(np.float64)
mod = r.get('mod')

# %% コード11-10(python)
# 粒子フィルタの事前設定
N = 10000

# データの整形
y = np.r_[np.nan, y]

# 実効サンプルサイズを格納するための配列
ESS = np.repeat(N, repeats=t_max+1)

# 事前分布の設定
# 粒子（実現値）
# 粒子 (実現値)
x = np.zeros(shape=(t_max+1, N))
x[0,:] = np.random.normal(loc=mod["m0"] , scale=np.sqrt(mod["C0"][0][0]) , size=N)


# 粒子(重み)
w = np.zeros(shape=(t_max+1, N))
w[0,:] = np.log(1/N)

# 時間順方向の処理
for tt in np.arange(t_max)+1 :
    # リサンプリング(相当)
    # 補助変数列
    probs = w[tt-1,:] + sct.norm.logpdf(y[tt], loc=x[tt-1,:], scale=np.sqrt(mod["V"]))
    k = sys_resampling(N=N, w=normalize(probs.reshape(-1)))
    
    # 状態方程式から粒子(実現値)を生成
    x[tt,:] = np.random.normal(loc=x[tt-1, k], scale=np.sqrt(mod["W"][0][0]), size=N)
    
    # 観測方程式：粒子(重み)を更新
    w[tt,:] = sct.norm.logpdf(y[tt], loc=x[tt,:], scale=np.sqrt(mod["V"])) -  sct.norm.logpdf(y[tt], loc = x[tt, k], scale = np.sqrt(mod["V"]))
    
    # 重みの規格化
    w[tt,:] = normalize(w[tt,:])
    
    # 実効サンプルサイズ
    ESS[tt] = 1 / np.dot(np.exp(w[tt,:]), np.exp(w[tt,:]))

w = w[1:]; x = x[1:]; ESS = ESS[1:]; y = y[1:]

# 実効サンプルサイズを保存し、平均も求める
APF_ESS = ESS
APF_m = np.average(x, axis=1, weights = np.exp(w))

# 各時刻の粒子平均値をプロット
plt.figure(1)
plt.plot(y, marker=".", linestyle="--")
plt.plot(np.mean(x, axis=1), marker=".", linestyle="--")
plt.title('Auxiliary Particle Filter : filtering')
plt.legend(["data", "filtered"], frameon=True, edgecolor="b")
plt.tight_layout(); plt.show()
plt.savefig("result/mean_11-10.png")

# 50%区間のプロット
plt.figure(2)
plt.plot(np.arange(t_max), y, marker=".", linestyle="--")
plt.fill_between(np.arange(t_max), y1=np.percentile(x, 75, axis=1), 
                 y2=np.percentile(x, 25, axis=1), facecolor="b", alpha=0.3)
plt.title('Auxiliary Particle Filter : filtering')
plt.legend(["data", "filtered"], frameon=True, edgecolor="b")
plt.tight_layout(); plt.show()
plt.savefig('result/percentile_11-10.png')

plt.figure(3)
plt.plot(APF_ESS)
plt.title('Effective Sample Size')
plt.tight_layout()
plt.savefig('result/samplesize_11-10.png')