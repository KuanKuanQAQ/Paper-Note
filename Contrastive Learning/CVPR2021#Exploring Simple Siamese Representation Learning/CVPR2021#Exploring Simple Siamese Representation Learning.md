关于 SimSiam 的博客挺多，我觉得这个不错。

https://sh-tsang.medium.com/review-simsiam-exploring-simple-siamese-representation-learning-3c84ceb61702

## 5 Hypothesis

#### 5.1 Formulation

SimSiam 本质是在以 EM 算法的模式工作，隐式地包含了两组变量，形成了两个潜在的子问题。而 stop-gradient 就是引入额外一组变量的结果。

考虑 eq5 形式的损失函数：



其中 $\mathcal F$ 是参数为 $\theta$ 的网络，$\mathcal T$ 是数据增强，$x$ 是样本。我们在整个样本和数据增强的分布上计算期望 $E[\cdot]$。上文提到，SimSiam 隐式地包含了两组变量，其中一组是模型参数 $\theta$，另一组则是 eq5 中的 $\eta_x$。这个变量表示的是样本 $x$ 的“完美” embedding，也就是最终我们希望得到的样本表示。

然而这个表示我们是不知道的，所以就成为了一个待求解的变量。这就是 EM 算法中的隐变量。举例来说，k-means 算法中的隐变量是样本究竟属于哪个类别，而另一个变量则是各个类别的中心；GMM 的隐变量是样本属于各个分布的概率向量，而另一个变量则是各个分布的均值和方差。SimSiam 的隐变量是每个样本的完美 embedding $\eta_x$，而另一个变量则是模型参数。

它们的共同的特点是，两组变量不能（不易）同时优化，找不到目标函数的极小值点。所以不得不把优化策略转化为固定一组变量，调整另一组，交替进行。其实还有很多机器学习算法可以被总结成类 EM 算法。在 SimSiam 中，优化步骤就是，先固定 $\eta_x$ 优化 $\theta$，再固定 $\theta$ 优化 $\eta_x$，也就是上文提到的“两个潜在的子问题”。具体如下：

**Solving for $\theta$**



**Solving for $\eta$**
