# 说明

本文件夹存储着模型不同设置后实验跑出的结果

## 1. mmd
标准mmd实现

## 2.add alpha + mk-mmd
计算每一层mk-mmd，乘以系数alpha再相加，即$loss=loss_{cls}+\lambda \Sigma \alpha_i loss_{mk-mmd_i}$

简单实现的mk-mmd
## 3.mk-mmd
无alpha仅mmd，$loss=loss_{cls}+\lambda \Sigma loss_{mk-mmd}$，$\lambda=0.5、1.5$

简单实现的mk-mmd
## 4.add alpha_sum + mk-mmd
$loss=loss_{cls}+\lambda \Sigma \alpha_i loss_{mk-mmd_i}$
且$\Sigma \alpha_i = \alpha _{sum}$

简单实现的mk-mmd
## 5.stage + mk-mmd
损失设置如4所示 ，加入分阶段

第一阶段只更新分类损失 ，第二阶段更新分类损失+域对齐损失

前数十个epoch只更新分类损失

简单实现的mk-mmd
### 1)
前二十个epoch更新第一阶段，之后交替更新

$\alpha_{sum}$=1.0

$\lambda=0.5$

简单实现的mk-mmd
### 2)
前四十个epoch更新第一阶段，之后一直第二阶段

$\alpha_{sum}$=5.0

$\lambda=1.0$

简单实现的mk-mmd

### 3)

前四十个epoch更新第一阶段，之后一直第二阶段

$\alpha_{sum}$=5.0

$\lambda=1.0$

2015DAN的mk-mmd

### 4)

在3)的基础上，添加ndvi，ndwi通道

### 5)

4)同样设置，无域适应

### 6)
同4)，统计了全局数据，重写标准化