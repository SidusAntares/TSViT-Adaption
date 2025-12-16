# 说明

本文件夹存储着模型不同设置后实验跑出的结果

## 1. mmd
标准mmd实现

## 2.add alpha + mk-mmd
计算每一层mk-mmd，乘以系数alpha再相加，即$loss=loss_{cls}+\lambda \Sigma \alpha_i loss_{mk-mmd_i}$

## 3.mk-mmd
无alpha仅mmd，$loss=loss_{cls}+\lambda \Sigma loss_{mk-mmd}$，$\lambda=0.5、1.5$




