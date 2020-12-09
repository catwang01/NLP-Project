[toc]

# Transformer 面试题

## 1. 为什么 Transformer 里使用 layer norm 而不使用 batch norm？/ BatchNorm 和 LayerNorm 的区别

1. 如果对文本数据用 bn，相当于对相同位置的词进行标准化。但是在 nlp 问题里面，即使是相同位置，不同句式的的重点也有所不同。按照位置去进行标准化不符合 nlp 规律

2. 对于样本是向量的普通问题来说，ln 很奇怪，因为 ln 会讲某个样本所有变量向量化。假设我们的样本 [age, height, salary] 三个维度分别表示年龄，身高和工资，ln 相当于会讲 `(age + height + salary) / 3`  去计算均值，这个均值的含义很不明确。

但是对于文本问题，ln 却不会有这种问题。对于文本问题，ln 相当于是对单词对 embedding 去做标准化，由于我们认为 embedding 的数据范围是相同的，因此不会有刚才的问题。 

3. 文本数据的长度一般不固定，而图像一般会 clip 成相同大小，对 batch norm 的计算不友好。如果通过 padding 的方式搞成相同长度，计算是会带来有的地方是 padding 的单词，不是真正的单词的情况。

##  2. 为什么 softmax 之前要除以 sqrt(dk)？

Attention 公式

$$
attention = softmax(\frac{QK^T}{\sqrt{d_k}})V^T
$$


### 除以 $d_k$ 的目的

这个主要是为了缓解 softmax  的导数为 0 的问题。

$$
y = softmax(x) = \frac{exp(x)}{1^T exp(x)}
$$

$$
\begin{aligned}
d y &= \frac{exp(x)}{1^T exp(x)} \odot dx - \frac{exp(x)}{(1^T exp(x))^2} d (1^T exp(x)) \\
&= y \odot dx - \frac{exp(x)}{(1^T exp(x))^2} 1^T (exp(x) \odot dx) \\
&= y \odot dx - \frac{exp(x)}{(1^T exp(x))^2} (exp(x)^T dx) \\
&= y \odot dx - y y^T dx) \\
&= diag(y) dx - y y^T dx) \\
&= (diag(y)  - y y^ T)dx
\end{aligned}
$$

因此 

$$
\frac{d y}{dx} = diag(y) - y y^T
$$

**当 $d_k$ 较大时，$y$ 更有可能接近于 one-hot 向量**，此时 $\frac{d y}{dx} \approx 0$，因此无法更新参数。

### 为什么 $d_k$ 较大时， $y$ 更有可能接近于 one-hot 向量？

我们知道 $softmax$ 可以让大的更大，小的更小。也就是说，如果 y的各分量中有一个数比其他数大的多的时候，$softmax(y)$ 的结果接近于 one-hot。

而 $d_k$ 实际上是点乘之后的随机变量的方差。为此，我们需要添加一些假定：

假设有两个随机向量 $q,k \in R^{d_k}$, 它们均为标准多元正态分布，并且相互独立

他们的点积的均值和方差分别为

$$
\begin{aligned}
E(q \cdot k) &= Eq \cdot Ek \\ 
&= 0
\end{aligned}
$$

$$
\begin{aligned}
var(q \cdot k) &= E(q^2 \cdot k^2) - (E(q \cdot k) ) ^2 \\
&= E(q^2) \cdot E(k^2)  \\
&= (var(q) + E(q) ^ 2) \cdot (var(k) + E(k) ^ 2)  \\
&= 1 \cdot 1 = d_k
\end{aligned}
$$

因此 $d_k$ 越大，说明 $q \cdot k$ 的方差越大，方差越大，说明出现某个值显著大于其他值的可能性越大，因此 softmax 的结果约有可能接近于 one-hot 向量。

### 为什么除以 $d_k$ 不除以别的？

根据上面的解释，我们知道了 $d_k$ 是 $q \cdot k$ 的方差。我们希望这个方差稳定于 1， 那么只需要在计算 softmax 之前除以 $\sqrt{d_k}$ 即可

$$
var(\frac{q \cdot k }{\sqrt{d_k}}) = 1
$$


## 3. 为什么要多个头

原 paper 中的解释：多头可以允许模型从不同的子空间进行 attention。

有人研究过这个问题，不是所有的头都完全不同，也不是所有的头都完全相同。在多个头中，有许多头学到的 pattern 是相似的，但是也有一些头学到的 pattern 和其他头不同。

一个猜想：有可能是初始化不同的原因导致的。

首先，我们知道多个头计算的过程完全相同，不同的就只是初始化的过程不同而已。因此，我们可以猜测不同的头学到的结果不同是由于初始化所带来的。

有一种假设，不同的头表示的是不同的空间，假设有 a，b，c，d 四个空间，那么每个空间都对应一组头的参数。初始化会影响到是否能够收敛到某个空间中的最优值。

假设有的初始化可以收敛到 a 空间的最优值，有的初始化可以收敛到 b 空间的最优值。这样就可以解释有的头的 pattern 相似，而有的头 的 pattern 不相似的现象。

如果这假定成立，那么多个头的好处就是可以尝试不同的初始化，从而收敛到不同空间的概率就大，但是相对的，计算量就大。

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20201207172500.png)

## 4. transformer 是如何并行的？

transformer 是由 6 个 encoder + 6 个 decoder 组成的。

1. 6 个 decoder 之间不能并行，必须每个词每个词输出
2. 6 个 encoder 之间不能并行，必须先计算前一个 encoder 之后在计算后一个 encoder
3. 每个 encoder 由 fnn + self.attention 组成。
    1. fnn 本身可以并行，因为就只是不同的矩阵乘法，一下可以同时计算所有单词的乘法。
    2. self.attention 本身也可以并行。这里的并行是说 self.attention 在计算矩阵乘法时一下子考虑了所有的单词，而不是顺次考虑单词
    3. fnn 和 self.attention 之间不能并行，因为 fnn 是以 self.attention 的输出作为输入的，当然不能并行。


# References

1. [(7 封私信 / 80 条消息) transformer中的attention为什么scaled? - 知乎](https://www.zhihu.com/question/339723385)