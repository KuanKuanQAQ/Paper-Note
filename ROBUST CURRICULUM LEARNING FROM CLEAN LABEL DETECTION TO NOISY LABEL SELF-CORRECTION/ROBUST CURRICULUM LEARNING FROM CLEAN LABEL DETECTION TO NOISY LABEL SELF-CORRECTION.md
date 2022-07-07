# ROBUST CURRICULUM LEARNING: FROM CLEAN LABEL DETECTION TO NOISY LABEL SELF-CORRECTION

两段式耐噪方法，在每个 batch 中：

1. 过滤噪声数据，使用干净数据训练；
2. 对噪声数据重新标注。

**RoCL**，一个非标准的两段式耐噪学习方法：

整个训练过程分为多个 episode，每个 episode 又分为两个 phase：

* phase 1 使用小 origin-label loss 筛选样本，更新模型，
* phase 2 使用小 pseudo-label loss 筛选样本，更新模型。

在 phase 2 结束后进入下一个 episode 的 phase 1。在原标签和伪标签之间交替训练，形成了本文的 curriculum。课程学习中的 curriculum 不是只有从简单样本到困难样本这一种定式，任何训练过程中合理的样本选择方法都可以称为 curriculum。

**Origin-label loss** 用于评估一个样本原标签的可靠程度，在普通 loss 的基础上加入了 EMA。

<img src="asset/eq1.png" alt="eq1" style="zoom:50%;" />

**Pseudo-label loss** 用于评估一个样本由模型给出的伪标签的可靠程度，在 consistency loss 的基础上加入了 EMA。

<img src="asset/eq3.png" alt="eq3" style="zoom:50%;" />

其中 $\zeta_t(i)=\frac{1}{m}\sum^m_{j=1}l(f(x_i;\theta_t),f_t(x_i^{(j)};\bar \theta_t))$，为 m 个 augmentations 的伪标签的平均损失值。值得注意的是，伪标签的生成使用的是 mean teacher 方法：$\bar \theta_t=\gamma\theta_{t-1}+(1-\gamma)\bar \theta_{t-1}$。另外，使用 MixMatch 生成 augmentations。

```
Mean teacher: Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consis- tency targets improve semi-supervised deep learning results. In Advances in Neural Information Processing Systems 30 (NeurIPS), pp. 1195–1204. 2017.

Mixmatch: David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A Raffel. Mixmatch: A holistic approach to semi-supervised learning. In Advances in Neural Information Processing Systems 32 (NeurIPS), pp. 5050–5060. 2019.
```

**样本选择标准**

<img src="asset/eq5.png" alt="eq5" style="zoom:50%;" />

分别使用带 temperature parameter 的 softmax 计算每个样本 i 的原标签可靠程度和伪标签可靠程度。

对于每个样本而言，它是否在一个 phase 被选取加入到训练中，由 $p_t(i)$ 和 $q_t(i)$ 加权结果 $P_t(i)$ 决定：

<img src="asset/eq4.png" alt="eq4" style="zoom:50%;" />

其中，$P_t(i)$ 为样本 i 加入训练的概率，所有加入训练的样本组成了 $S_t$，也就是 Eq.1 和 Eq.3 中的 $S_t$。于是，RoCL 中的样本选择标准由参数组 $(\tau_1, \tau_2, \lambda)$ 决定，也就是参数组 $(\tau_1, \tau_2, \lambda)$ 控制课程。

$\tau_1$ 为负（正）时，$l_t(i)$ 越小（大）则 $p_t(i)$ 越大（小）；$\tau_2$ 为负（正）时，$c_t(i)$ 越小（大）则 $q_t(i)$ 越大（小）。虽然 $p_t(i)$ 和 $q_t(i)$ 都可以独立用于选择样本，但是它们的单独使用会过度关注小损失样本或模型对各个 augmentations 的预测已经一致正确的样本，最终导致 little progress can be made。

```
温度参数不仅可以控制增减性，还可以控制“尖锐程度”。温度参数越大，softmax 输出的概率就越集中于某几项。
```

但是当 $p_t(i)$ 和 $q_t(i)$ 联合使用，且 $\tau_1$ 和 $\tau_2$ 一正一负时可以改善这一状况。当 $\tau_1$ 为负，$\tau_2$ 为正时，$l_t(i)$ 较小 $c_t(i)$ 较大的样本会获得高概率，也就是选择模型输出不一致的干净样本（即尚未被完全学习的样本）；当 $\tau_1$ 为正，$\tau_2$ 为负时，$l_t(i)$ 较大 $c_t(i)$ 较小的样本会获得高概率，也就是选择伪标签正确的噪声样本。

所以课程 $(\tau_1,\tau_2,\lambda)$ 可以这样设计：$\tau_1$ 由负到正，$\tau_2$ 正好相反，$\lambda$ 随 $\tau_1$ 单调递增。这种课程下，模型先学习选择模型输出不一致的干净样本，再学习有着正确伪标签的噪声样本，对应了 RoCL 方法中的 phase 1 和 phase 2。


**样本梯度计算**
$$
G_t(i)=\frac{\lambda p_t(i)}{P_t(i)}\nabla_\theta l(f(x_i;\theta_t),y_i)+\frac{(1-\lambda)q_t(i)}{P_t(i)}\nabla_\theta\zeta_t(i) \tag 8
$$

梯度来自于两个梯度的加权，权重为一个样本 $p_t(i)$ 和 $q_t(i)$ 的归一化结果。也就是说，如果一个样本是由一个大 $p_t(i)$ 选出的，那么其梯度也就会更与 origin-label loss 相关；反之则是与 pseudo-label loss 相关。

<img src="asset/Algorithm 1.png" alt="Algorithm 1" style="zoom: 50%;" />