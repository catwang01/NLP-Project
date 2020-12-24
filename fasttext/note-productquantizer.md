[toc]

#  Fasttext productquantizer 源码解析

## 头文件

```cpp
#pragma once

#include <cstring>
#include <istream>
#include <ostream>
#include <random>
#include <vector>

#include "real.h"
#include "vector.h"

namespace fasttext {

class ProductQuantizer {
 protected:
  const int32_t nbits_ = 8;
  const int32_t ksub_ = 1 << nbits_; // 聚成多少类
  const int32_t max_points_per_cluster_ = 256; 
  const int32_t max_points_ = max_points_per_cluster_ * ksub_; // 只对这么多的
  const int32_t seed_ = 1234;
  const int32_t niter_ = 25;
  const real eps_ = 1e-7;

  int32_t dim_; // 原始向量维数
  int32_t nsubq_; // 划分为多少个子空间
  int32_t dsub_; // 每个空间的的维数
  int32_t lastdsub_; // 最后一个子空间维数，因为可能无法整除，因此最后一个维度的大小可能和之前的大小不同

  std::vector<real> centroids_; // 存放所有的质心
                                // 大小为 dim_ * ksub_

  std::minstd_rand rng;

 public:
  ProductQuantizer() {}
  ProductQuantizer(int32_t, int32_t);

  real* get_centroids(int32_t m, uint8_t i);   // 返回下标为 m 的子空间中下标为 i 的聚类中心的位置
  const real* get_centroids(int32_t m, uint8_t i) const; // 返回下标为 m 的子空间中下标为 i 的聚类中心的位置

  real assign_centroid(const real*, const real*, uint8_t*, int32_t) const; // 将某个向量分到离他最近的类中
  void Estep(const real*, const real*, uint8_t*, int32_t, int32_t) const; // 将 n 个向量分类
  void MStep(const real*, real*, const uint8_t*, int32_t, int32_t);
  void kmeans(const real*, real*, int32_t, int32_t);
  void train(int n, const real* x); // x 是 n 个向量拼接起来的
                                    // 在 x 上做训练

  real mulcode(const Vector&, const uint8_t*, int32_t, real) const;
  void addcode(Vector&, const uint8_t*, int32_t, real) const;
  void compute_code(const real*, uint8_t*) const;
  void compute_codes(const real*, uint8_t*, int32_t) const;

  void save(std::ostream&) const;
  void load(std::istream&);
};

} // namespace fasttext
```

## 代码解析

### 压缩的核心思想

实际上就是所谓的 Product Quantization。这部分可以上网查找。

### 构造函数初始化

```cpp
ProductQuantizer::ProductQuantizer(int32_t dim, int32_t dsub)
    : dim_(dim),
      nsubq_(dim / dsub),
      dsub_(dsub),
      centroids_(dim * ksub_),
      rng(seed_) {
  lastdsub_ = dim_ % dsub;
  if (lastdsub_ == 0) { // 恰好可以整除
    lastdsub_ = dsub_;
  } else {
    nsubq_++; // 不能整除
  }
  // 有 (nsubq_ - 1) * dsub_ + lastdsub_ = dim_
}
```


### assign_centroid

计算离 x 最近的质心的距离，并修改 code 为最近质心的idx

```cpp
real ProductQuantizer::assign_centroid(
    const real* x,
    const real* c0,
    uint8_t* code,
    int32_t d) const {
  const real* c = c0;
  // 用第一个子空间上的初始化
  real dis = distL2(x, c, d);
  code[0] = 0;
  // 迭代其他 ksub_ - 1 个子空间
  for (auto j = 1; j < ksub_; j++) {
    c += d; // c 是指针，这个相当于指针向前移动 d 个单位，移到下个向量的开头
    real disij = distL2(x, c, d);
    if (disij < dis) {
      code[0] = (uint8_t)j;
      dis = disij;
    }
  }
  return dis;
}
```


### train 

```cpp
void ProductQuantizer::train(int32_t n, const real* x) {
  if (n < ksub_) {
    throw std::invalid_argument(
        "Matrix too small for quantization, must have at least " +
        std::to_string(ksub_) + " rows");
  }
  // 从 x 中选择至多 max_points_ 个点，放到 xslice 中，之后就用 xslice 中的数据做 kmeans
  std::vector<int32_t> perm(n, 0);
  std::iota(perm.begin(), perm.end(), 0);
  auto d = dsub_;
  auto np = std::min(n, max_points_);
  auto xslice = std::vector<real>(np * dsub_);
  for (auto m = 0; m < nsubq_; m++) { // 遍历 nsubq_ 个子空间
    if (m == nsubq_ - 1) {
      d = lastdsub_; // 如果是最后一个空间，d = lastsub_
    }
    if (np != n) { // 说明待压缩的向量数大于 max_points_，因此需要随机抽样
      std::shuffle(perm.begin(), perm.end(), rng);
    }
    for (auto j = 0; j < np; j++) {
      memcpy(
          xslice.data() + j * d,
          x + perm[j] * dim_ + m * dsub_,
          d * sizeof(real));
    }
    // 在 xslice 上做 kmeans
    kmeans(xslice.data(), get_centroids(m, 0), np, d); // 在下标为 m 的子空间中做 kmeans
  }
}
```

```cpp
void ProductQuantizer::kmeans(const real* x, real* c, int32_t n, int32_t d) {
  std::vector<int32_t> perm(n, 0);
  std::iota(perm.begin(), perm.end(), 0); // 获取 [0, 1, .., n-1]
  std::shuffle(perm.begin(), perm.end(), rng);
  for (auto i = 0; i < ksub_; i++) { // 随机选择 n 个向量中的 ksub_ 个用来初始化类中心
                                     // 这里利用 shuffle [0, 1, ..., n-1] 获取前 ksub_ 个来实现随机抽取 ksub_个
    memcpy(&c[i * d], x + perm[i] * d, d * sizeof(real));
  }
  auto codes = std::vector<uint8_t>(n);
  for (auto i = 0; i < niter_; i++) {
    Estep(x, c, codes.data(), d, n);
    MStep(x, c, codes.data(), d, n);
  }
}
```

```cpp
void ProductQuantizer::Estep( // 将向量分到 ksub_ 个簇中
    const real* x,
    const real* centroids,
    uint8_t* codes,
    int32_t d, // 
    int32_t n // 是待压缩的样本个数
    ) const {
  for (auto i = 0; i < n; i++) {
    assign_centroid(x + i * d, centroids, codes + i, d);
  }
}

void ProductQuantizer::MStep(
    const real* x0,
    real* centroids,
    const uint8_t* codes,
    int32_t d,
    int32_t n) {
  std::vector<int32_t> nelts(ksub_, 0); // 每一簇的样本数
  memset(centroids, 0, sizeof(real) * d * ksub_);
  const real* x = x0;
  for (auto i = 0; i < n; i++) {
    auto k = codes[i]; // 第 i 个样本属于第 k 簇
    real* c = centroids + k * d;
    for (auto j = 0; j < d; j++) { // 样本中心累加
      c[j] += x[j];
    }
    nelts[k]++; // 该簇样本数加 1
    x += d;
  }

  real* c = centroids;
  for (auto k = 0; k < ksub_; k++) {
    real z = (real)nelts[k];
    if (z != 0) {
      for (auto j = 0; j < d; j++) {
        c[j] /= z; // 样本中心求平均
      }
    }
    c += d; 
  }

  // 这里看不懂
  std::uniform_real_distribution<> runiform(0, 1);
  for (auto k = 0; k < ksub_; k++) {
    if (nelts[k] == 0) { // 如果某一簇没有样本
      int32_t m = 0;
      while (runiform(rng) * (n - ksub_) >= nelts[m] - 1) {
        m = (m + 1) % ksub_;
      }
      memcpy(centroids + k * d, centroids + m * d, sizeof(real) * d);
      for (auto j = 0; j < d; j++) {
        int32_t sign = (j % 2) * 2 - 1; // 偶数维度 sign=-1, 奇数维度山 sign=1
        centroids[k * d + j] += sign * eps_;
        centroids[m * d + j] -= sign * eps_;
      }
      // m 的数据分一半给 k
      nelts[k] = nelts[m] / 2; 
      nelts[m] -= nelts[k];
    }
  }
}
```

### compute_code && compute_codes

```cpp
void ProductQuantizer::compute_code(const real* x, uint8_t* code) const {
  auto d = dsub_;
  for (auto m = 0; m < nsubq_; m++) { // 对 nsubq_ 个子空间分别分类
    if (m == nsubq_ - 1) {
      d = lastdsub_;
    }
    assign_centroid(x + m * dsub_, get_centroids(m, 0), code + m, d);
  }
}

void ProductQuantizer::compute_codes(const real* x, uint8_t* codes, int32_t n)
    const {
  for (auto i = 0; i < n; i++) { // 对 n 个向量都计算 nsubq_
    compute_code(x + i * dim_, codes + i * nsubq_);
  }
}
```

### save && load

`save` 和 `load` 的部分就是使用 `write` 和 `read` 方法用来保存二进制文件。

```cpp
void ProductQuantizer::save(std::ostream& out) const {
  out.write((char*)&dim_, sizeof(dim_));
  out.write((char*)&nsubq_, sizeof(nsubq_));
  out.write((char*)&dsub_, sizeof(dsub_));
  out.write((char*)&lastdsub_, sizeof(lastdsub_));
  out.write((char*)centroids_.data(), centroids_.size() * sizeof(real));
}

void ProductQuantizer::load(std::istream& in) {
  in.read((char*)&dim_, sizeof(dim_));
  in.read((char*)&nsubq_, sizeof(nsubq_));
  in.read((char*)&dsub_, sizeof(dsub_));
  in.read((char*)&lastdsub_, sizeof(lastdsub_));
  centroids_.resize(dim_ * ksub_);
  for (auto i = 0; i < centroids_.size(); i++) { // 为什么要 for 循环？
                                                 // 而不是 in.read((char*)&centroids_, sizeof(real) * centroids_.size())
    in.read((char*)&centroids_[i], sizeof(real));
  }
}
```

# References
1. [fasttext源码解析（1） - 知乎](https://zhuanlan.zhihu.com/p/64960839)
