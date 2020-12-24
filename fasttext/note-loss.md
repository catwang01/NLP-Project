[toc]

# Fasttext loss.h 和 loss.cc 文件 源码解析

## 头文件

```cpp
#pragma once

#include <memory>
#include <random>
#include <vector>

#include "matrix.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

class Loss {
 private:
  void findKBest(
      int32_t k,
      real threshold,
      Predictions& heap,
      const Vector& output) const;

 protected:
  std::vector<real> t_sigmoid_; // sigmoid 函数的查找表
  std::vector<real> t_log_; // log 函数的查找表
  std::shared_ptr<Matrix>& wo_; // 输出层的矩阵，对于不同的算法，这个矩阵的含义不同

  real log(real x) const;
  real sigmoid(real x) const;

 public:
  explicit Loss(std::shared_ptr<Matrix>& wo);
  virtual ~Loss() = default;

  virtual real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) = 0;
  virtual void computeOutput(Model::State& state) const = 0; // 计算 output, 保存在 state.output 中

  virtual void predict(
      int32_t /*k*/,
      real /*threshold*/,
      Predictions& /*heap*/,
      Model::State& /*state*/) const;
};

class BinaryLogisticLoss : public Loss {
 protected:
  real binaryLogistic(
      int32_t target,
      Model::State& state,
      bool labelIsPositive, // 样本是否为正样本
      real lr,
      bool backprop) const; // 根据 target，计算这个样本对应的损失。如果该样本是正样本，返回 sigmoid(x)，否则，返回 1 - sigmoid(x)

 public:
  explicit BinaryLogisticLoss(std::shared_ptr<Matrix>& wo);
  virtual ~BinaryLogisticLoss() noexcept override = default;
  void computeOutput(Model::State& state) const override;
};

class OneVsAllLoss : public BinaryLogisticLoss {
 public:
  explicit OneVsAllLoss(std::shared_ptr<Matrix>& wo);
  ~OneVsAllLoss() noexcept override = default;
  real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) override;
};

class NegativeSamplingLoss : public BinaryLogisticLoss {
 protected:
  static const int32_t NEGATIVE_TABLE_SIZE = 10000000;

  int neg_; // 每个正样本对应 neg_ 个负样本
  std::vector<int32_t> negatives_; // 这个是负采样的预取表
  std::uniform_int_distribution<size_t> uniform_; 
  int32_t getNegative(int32_t target, std::minstd_rand& rng);

 public:
  explicit NegativeSamplingLoss(
      std::shared_ptr<Matrix>& wo,
      int neg,
      const std::vector<int64_t>& targetCounts);
  ~NegativeSamplingLoss() noexcept override = default;

  real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) override;
};

class HierarchicalSoftmaxLoss : public BinaryLogisticLoss {
 protected:
  struct Node {
    int32_t parent;
    int32_t left;
    int32_t right;
    int64_t count;
    bool binary; // 是其父节点的左节点还是右节点
  };

  std::vector<std::vector<int32_t>> paths_;
  std::vector<std::vector<bool>> codes_; // 编码
  std::vector<Node> tree_;
  int32_t osz_;
  void buildTree(const std::vector<int64_t>& counts);
  void dfs(
      int32_t k,
      real threshold,
      int32_t node,
      real score,
      Predictions& heap,
      const Vector& hidden) const;

 public:
  explicit HierarchicalSoftmaxLoss(
      std::shared_ptr<Matrix>& wo,
      const std::vector<int64_t>& counts);
  ~HierarchicalSoftmaxLoss() noexcept override = default;
  real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) override;
  void predict(
      int32_t k,
      real threshold,
      Predictions& heap,
      Model::State& state) const override;
};

class SoftmaxLoss : public Loss {
 public:
  explicit SoftmaxLoss(std::shared_ptr<Matrix>& wo);
  ~SoftmaxLoss() noexcept override = default;
  real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) override;
  void computeOutput(Model::State& state) const override;
};

} // namespace fasttext
```

## 各类详解

### Loss 类


Loss 类是一个纯虚类，其暴露了下面两个接口   

```cpp
virtual real forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) = 0;
```

`forward` 接口用来计算前向传播。其中有一个 backprop 参数可以控制是否计算梯度。

```cpp
virtual void computeOutput(Model::State& state) const = 0;
```

`computeOutput` 函数描述了如何利用中间层 `state.hidden` 来计算 output 的值，并写入 `state.output` 中。主要是在 `SoftmaxLoss` 中使用到

`BinaryLogisticLoss` 继承自 Loss，实现了其  computeOutput 方法。 `OneVsAllLoss` 继承自 `BinaryLogisticLoss`，实现了其 forward 方法。

#### 列表近似

Loss 函数的构造函数中有一段代码，是用来初始化列表的。

使用查表的方式来减少计算。假设，我们想计算 $x \in [0, 1]$ 上的函数 `f(x)` 的值。

如果 `f(x)` 使用非常频繁，我们可以采取计算一个列表，然后用列表的内容近似。 我们可以使用表刻度的方式来。假设我们需要将区间 [0, 1] 分成 n 份，那么我们需要标 n + 1 个刻度。每个刻度对应一个整数 $i \in [0,1,...,n]$ 我们需要通过某个映射将 $x \in [0,1]$ 映射到 $i \in [0, 1, ..., n]$

$$
x = \frac{i}{n}, i \in [0, 1, ..., n]
$$

也就是说，我们只需要 计算 $x =  \frac{i}{n} \in [0, 1]$ 对应的 $f(x)$ 的值即可。


下面是填表的代码

```
初始化表 T，大小为 n + 1
for i in [0, n]:
    T[i] = f(i / n)
```

我们填表得到表 T 后改如何使用呢？

假设我们要计算 $f(a)$，我们希望在表 T 中找到 i，满足

$$
a = \frac{i}{n}
$$

得到

$$
i = a * n
$$

由于 a 不太可能刚好满足 $a * n$ 是整数，因此我们取

$$
i = \lfloor a * n  \rfloor
$$


刚才，我们假设我们要计算 $x \in [0, 1]$ 的 $f(x)$ 函数的值。现在，我们拓广一下， 我们要计算 $x \in [-max, max]$ 上 $f(x)$ 的值，注意到 $\frac{x + max}{2 * max} \in [0, 1]$，因此我们可以取映射

$$
\frac{x + max}{2 * max} = \frac{i}{n} \in [0, max]
$$

即 

$$
x = \frac{2 * i * max}{n} - max
$$

这样，假设我们想计算 $f(a)$ 的值。我们可以找到对应的 i，即我们希望找到

$$
a = \frac{2 * i * max}{n} - max
$$

得

$$
i = \left\lfloor \frac{(a + max) * n}{max * 2} \right\rfloor
$$

这里的 i 计算出来可能会是小数，我们将其强制转化为整数来近似

因此 $f(a) \approx T[(a + max) * n / max / 2]$

#### Loss::sigmoid 和 Loss::log

令上述 `n = SIGMOID_TABLE_SIZE`, `max = MAX_SIGMOID`，就可以得到下面的代码:

```cpp
Loss::Loss(std::shared_ptr<Matrix>& wo) : wo_(wo) {
  t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
  }

  t_log_.reserve(LOG_TABLE_SIZE + 1);
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log_.push_back(std::log(x));
  }
}

real Loss::log(real x) const {
  if (x > 1.0) {
    return 0.0;
  }
  int64_t i = int64_t(x * LOG_TABLE_SIZE);
  return t_log_[i];
}

real Loss::sigmoid(real x) const {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int64_t i =
        int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid_[i];
  }
}
```

### BinaryLogisticLoss

`BinarayLogisticLoss` 实现了 `Loss` 类的 `computeOutput` 方法。

```cpp
BinaryLogisticLoss::BinaryLogisticLoss(std::shared_ptr<Matrix>& wo)
    : Loss(wo) {}

real BinaryLogisticLoss::binaryLogistic(
    int32_t target,
    Model::State& state,
    bool labelIsPositive,
    real lr,
    bool backprop) const {
  real score = sigmoid(wo_->dotRow(state.hidden, target));
  if (backprop) {
    real alpha = lr * (real(labelIsPositive) - score);
    state.grad.addRow(*wo_, target, alpha);
    wo_->addVectorToRow(state.hidden, target, alpha);
  }
  if (labelIsPositive) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

void BinaryLogisticLoss::computeOutput(Model::State& state) const {
  Vector& output = state.output;
  output.mul(*wo_, state.hidden);
  int32_t osz = output.size();
  for (int32_t i = 0; i < osz; i++) {
    output[i] = sigmoid(output[i]);
  }
}
```

### OneVsAllLoss

`OneVsAllLoss` 继承自 `BinaryLogisticLoss`，实现了其 forward 方法。

```cpp
OneVsAllLoss::OneVsAllLoss(std::shared_ptr<Matrix>& wo)
    : BinaryLogisticLoss(wo) {}

real OneVsAllLoss::forward(
    const std::vector<int32_t>& targets,  // targets 中的元素是正例
    int32_t /* we take all targets here */,
    Model::State& state,
    real lr,
    bool backprop) {
  real loss = 0.0;
  int32_t osz = state.output.size();
  for (int32_t i = 0; i < osz; i++) { // 对于所有的 output, 计算 loss
    bool isMatch = utils::contains(targets, i); // isMatch == true 表示是正例，isMatch == false 表示是负例
    loss += binaryLogistic(i, state, isMatch, lr, backprop);
  }

  return loss;
}
```

### NegativeSamplingLoss

#### 初始化 negatives_

我们要进行负采样。负采样应该和频率有关系，这里取

$$
p[i] = \frac{count(i)}{\sum_{j=1}^n count(j)}
$$

作为采样频率。假设我们的预采样表大小为 NEGATIVE_TABLE_SIZE, 那么这个表中就应该添加 `p[i] * NEGATIVE_TABLE_SIZE` 个单词 i。

预采样的时候直接随机就可以。不需要其他操作。

```cpp
NegativeSamplingLoss::NegativeSamplingLoss(
    std::shared_ptr<Matrix>& wo,
    int neg,
    const std::vector<int64_t>& targetCounts)
    : BinaryLogisticLoss(wo), neg_(neg), negatives_(), uniform_() {
  real z = 0.0;
  for (size_t i = 0; i < targetCounts.size(); i++) {
    z += pow(targetCounts[i], 0.5);
  }
  for (size_t i = 0; i < targetCounts.size(); i++) {
    real c = pow(targetCounts[i], 0.5);
    // 向 negatives_ 中添加 c / z * NEGATIVE_TABLE_SIZE 个样本
    for (size_t j = 0; j < c * NegativeSamplingLoss::NEGATIVE_TABLE_SIZE / z;
         j++) {
      negatives_.push_back(i);
    }
  }
  uniform_ = std::uniform_int_distribution<size_t>(0, negatives_.size() - 1);
}

int32_t NegativeSamplingLoss::getNegative(
    int32_t target,
    std::minstd_rand& rng) {
  int32_t negative;
  do {
    negative = negatives_[uniform_(rng)];
  } while (target == negative);
  return negative;
}
```

```cpp
real NegativeSamplingLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  assert(targetIndex >= 0);
  assert(targetIndex < targets.size());
  int32_t target = targets[targetIndex]; // 一个正样本
  real loss = binaryLogistic(target, state, true, lr, backprop);

  for (int32_t n = 0; n < neg_; n++) { // 随机抽取 neg_ 个负样本，并添加到 loss 上
    auto negativeTarget = getNegative(target, state.rng);
    loss += binaryLogistic(negativeTarget, state, false, lr, backprop); 
  }
  return loss;
}

```

### HierarchicalSoftmaxLoss

#### wo_ 的含义

在 `HierarchicalSoftmaxLoss` 中， `wo_` 的含义是每个内节点所对应的参数。

#### huffmann tree 的构建

```cpp
HierarchicalSoftmaxLoss::HierarchicalSoftmaxLoss(
    std::shared_ptr<Matrix>& wo,
    const std::vector<int64_t>& targetCounts)
    : BinaryLogisticLoss(wo),
      paths_(),
      codes_(),
      tree_(),
      osz_(targetCounts.size()) {
  buildTree(targetCounts);
}

void HierarchicalSoftmaxLoss::buildTree(const std::vector<int64_t>& counts) {
  tree_.resize(2 * osz_ - 1); // 树中的节点树为 2 * osz_ - 1
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) { // 初始化节点
    tree_[i].parent = -1;
    tree_[i].left = -1;
    tree_[i].right = -1;
    tree_[i].count = 1e15;
    tree_[i].binary = false;
  }
  for (int32_t i = 0; i < osz_; i++) {
    tree_[i].count = counts[i];
  }
  int32_t leaf = osz_ - 1; // [0, osz_ - 1] 这范围保存的是叶节点
  int32_t node = osz_; // [osz_, 2 * osz_-2] 保存的是内节点
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
    int32_t mini[2] = {0};
    // 在 leaf 和 node 两个位置找最小值，找两次
    for (int32_t j = 0; j < 2; j++) {
        // 如果 node 处的值比较大
      if (leaf >= 0 && tree_[leaf].count < tree_[node].count) {
        mini[j] = leaf--;
      } else {
        // 如果 node 处的值比较小
        mini[j] = node++;
      }
    }
    tree_[i].left = mini[0];
    tree_[i].right = mini[1];
    tree_[i].count = tree_[mini[0]].count + tree_[mini[1]].count;
    tree_[mini[0]].parent = i;
    tree_[mini[1]].parent = i;
    tree_[mini[1]].binary = true;  // 右节点看作正样本
  } // 树构建完毕
  // 计算编码
  for (int32_t i = 0; i < osz_; i++) { // 从叶节点开始向上
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree_[j].parent != -1) { // 根节点不计算入编码中
      path.push_back(tree_[j].parent - osz_); // tree_[j].parent - osz_ 是内节点的编号
                                            // 在 trees_ 数组中，内节点位于 j in [osz_, 2 * osz_ -2] 的位置上
                                            // 我们给每个内节点一个编号 i，i in [0, osz_ - 2]，以对应 wo_ 的每一行
                                            // 我们选择 i = trees[j].parent - osz_ 作为映射，将 j in [osz_, 2 * osz_-2] 映射到 [0, osz_ - 2]
      code.push_back(tree_[j].binary);
      j = tree_[j].parent;
    }
    paths_.push_back(path);
    codes_.push_back(code);
  }
}
```

```cpp
real HierarchicalSoftmaxLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  real loss = 0.0;
  int32_t target = targets[targetIndex];
  const std::vector<bool>& binaryCode = codes_[target];
  const std::vector<int32_t>& pathToRoot = paths_[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) { // binaryCode[i] 表示节点 i 是正样本还是负样本
    loss += binaryLogistic(pathToRoot[i], state, binaryCode[i], lr, backprop);
  }
  return loss;
}
```


#### forward

同时， `HierarchicalSoftmaxLoss` 重写了父类的 `predict` 函数。

在预测时，我们需要从根节点开始，遍历整个 huffmann tree，计算每个叶节点上的分数。在 `predict` 中，实现了剪枝：

1. 利用 threshold 剪枝，如果 score 小于 threshold，那么就不计算这个概率
2. 利用 k 来剪枝，我们只计算 topK 的概率。

```cpp
void HierarchicalSoftmaxLoss::predict( 
    int32_t k, // 我们值求 topK 的那些概率
    real threshold, // 小于 threshold 的值丢掉
    Predictions& heap, // 保存 topK 的概率结果
    Model::State& state) const {
        // 2 * osz_2 - 2 是 huffmann tree 的根节点在 trees_ 中的坐标
  dfs(k, threshold, 2 * osz_ - 2, 0.0, heap, state.hidden);
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void HierarchicalSoftmaxLoss::dfs(
    int32_t k,
    real threshold,
    int32_t node,
    real score,
    Predictions& heap,
    const Vector& hidden) const {
  if (score < std_log(threshold)) { // 当前节点的值已经小于阈值，由于 score 是越往下走越小，因此可以在这里剪枝
    return;
  }
  if (heap.size() == k && score < heap.front().first) { // 当前节点不可能是 topK 的结果了，剪枝
    return;
  }

  if (tree_[node].left == -1 && tree_[node].right == -1) { // 如果到叶节点，将结果添加到 heap 中
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }

  real f = wo_->dotRow(hidden, node - osz_);
  f = 1. / (1 + std::exp(-f));

  // 不是叶节点，继续 dfs 
  // 注意 score 每次都会加上一个小于 0 的数，因此 score 是越来越小的
  dfs(k, threshold, tree_[node].left, score + std_log(1.0 - f), heap, hidden);
  dfs(k, threshold, tree_[node].right, score + std_log(f), heap, hidden);
}
```

### SoftmaxLoss

```cpp
SoftmaxLoss::SoftmaxLoss(std::shared_ptr<Matrix>& wo) : Loss(wo) {}

void SoftmaxLoss::computeOutput(Model::State& state) const {
  Vector& output = state.output;
  output.mul(*wo_, state.hidden); // state.hidden 是个 Vector
  // 下面在计算 softmax，稳定算法
  real max = output[0], z = 0.0;
  int32_t osz = output.size();
  for (int32_t i = 0; i < osz; i++) {
    max = std::max(output[i], max);
  }
  for (int32_t i = 0; i < osz; i++) {
    output[i] = exp(output[i] - max);
    z += output[i];
  }
  for (int32_t i = 0; i < osz; i++) {
    output[i] /= z;
  }
}

real SoftmaxLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  computeOutput(state);

  assert(targetIndex >= 0);
  assert(targetIndex < targets.size());
  int32_t target = targets[targetIndex];

  if (backprop) {
    int32_t osz = wo_->size(0);
    for (int32_t i = 0; i < osz; i++) {
      real label = (i == target) ? 1.0 : 0.0;
      real alpha = lr * (label - state.output[i]);
      state.grad.addRow(*wo_, i, alpha);
      wo_->addVectorToRow(state.hidden, i, alpha);
    }
  }
  return -log(state.output[target]);
};
```

# References
1. [fastText 源码分析 - Helei's Tech Notes](https://heleifz.github.io/14732610572844.html)
2. [【算法】赫夫曼树（Huffman）的构建和应用（编码、译码） - 外婆的 - 博客园](https://www.cnblogs.com/penghuwan/p/8308324.html)
3. [fasttext源码解析（2） - 知乎](https://zhuanlan.zhihu.com/p/65687490)