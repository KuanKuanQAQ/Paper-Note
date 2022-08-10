关于 SimSiam 的博客挺多，我觉得这个不错。

https://sh-tsang.medium.com/review-simsiam-exploring-simple-siamese-representation-learning-3c84ceb61702

## 5 Hypothesis

#### 5.1 Formulation

SimSiam 本质是在以 EM 算法的模式工作，隐式地包含了两组变量，形成了两个潜在的子问题。而 stop-gradient 就是引入额外一组变量的结果。

考虑 eq5 形式的损失函数：

eq5

其中 $\mathcal F$ 是参数为 $\theta$ 的网络，$\mathcal T$ 是数据增强，$x$ 是样本。我们在整个样本和数据增强的分布上计算期望 $E[\cdot]$。SimSiam 隐式地包含了两组变量，其中一组是模型参数 $\theta$，另一组则是 eq5 中的 $\eta_x$。这个变量表示的是样本 $x$ 的“完美” embedding，也就是最终我们希望得到的样本表示。

然而这个表示我们是不知道的，所以就成为了一个待求解的变量。这就是 EM 算法中的隐变量。举例来说，k-means 算法中的隐变量是样本究竟属于哪个类别，而另一个变量则是各个类别的中心；GMM 的隐变量是样本属于各个高斯分布的概率向量，而另一个变量则是各个高斯分布的均值和方差。SimSiam 的隐变量是每个样本的完美 embedding $\eta_x$，而另一个变量则是模型参数。

它们的共同的特点是，两组变量不能（不易）同时优化，找不到目标函数的极小值点。所以不得不把转化为固定一组变量，调整另一组，交替进行的子问题进行优化。还有很多机器学习算法可以被总结成类 EM 算法。在 SimSiam 中，优化步骤就是，先固定 $\eta_x$ 优化 $\theta$，再固定 $\theta$ 优化 $\eta_x$，也就是“两个潜在的子问题”。具体如下：

**Solving for $\theta$**

在这个子问题中，$\eta$ 是一个常数。使用 SGD 调整 $\theta$。所以 stop-gradient 操作也就非常自然了。

**Solving for $\eta$**

我们可以独立求解每一个 $\eta_x$，只要分别最小化这个期望 $E_{\mathcal{T}}[\|\mathcal F_{\theta^t}(\mathcal T(x))-\eta_x\|^2_2]$，注意，这是在数据增强 $\mathcal T$ 的分布上求期望。显然，这个期望的最小值点如下 eq9：

eq9

**One-step alternation**

在每一步迭代中，首先从 $\mathcal T$ 的分布中随机取出一种增强 $\mathcal T'$，根据这一种增强近似 eq9 中的 $E_\mathcal T$：

eq10

这么做显然是有损失的，所以作者使用了 predictor h 去根据一种增强预测整个增强分布上的期望。在实际实现中，把期望 $E_\mathcal T$ 完全计算出来是不现实的，但是用一个神经网络（也就是 predictor h）去学习预测这个期望却是有可能的。

在 eq10 中，通过上一个时刻 t 的模型参数 $\theta_t$ 计算出 $\eta^t$ 之后，就可以使用 SGD 计算出本时刻 t+1 的模型参数 $\theta^{t+1}$ 了：

eq11

这里对样本 $x$ 和 $\mathcal T$ 求期望。按照 mini-batch SGD 的方法，计算 eq11 在 $x$ 上的期望是从 mini batch 近似。而 $\mathcal T$ 则是利用对称损失函数，具体可见原文 4.6 节。

**Multi-step alternation**

在每一步迭代中，SGD 可以多更新几步，理论上 SimSiam 也能发挥作用。但是如果我们要 SGD 多步，那就需要预先计算出更多的 $\eta_x$，留给 SGD 计算 eq 11。

根据作者的实验，multi-step alternation 可以让 SimSiam 的结果更好，但是提升很小（68.1% 到 68.9%）

