[toc]

# Fasttext 源码解析 vector.cc 和 matrix.cc

## Vector

Vector 实际上是对 std::vector<float> 进行了一些封装，添加了一些算数运算的功能。

头文件如下：

```cpp
#pragma once

#include <cstdint>
#include <ostream>
#include <vector>

#include "real.h"

namespace fasttext {

class Matrix;

class Vector {
 protected:
  std::vector<real> data_;

 public:
  explicit Vector(int64_t);
  Vector(const Vector&) = default;
  Vector(Vector&&) noexcept = default;
  Vector& operator=(const Vector&) = default;
  Vector& operator=(Vector&&) = default;

  inline real* data() {
    return data_.data();
  }
  inline const real* data() const {
    return data_.data();
  }
  inline real& operator[](int64_t i) {
    return data_[i];
  }
  inline const real& operator[](int64_t i) const {
    return data_[i];
  }

  inline int64_t size() const {
    return data_.size();
  }
  void zero(); // 所有数据置0
  void mul(real); // 数乘操作
  real norm() const;
  void addVector(const Vector& y); // x += y
  void addVector(const Vector& y, real s);  // x += y * s
  void addRow(const Matrix& m, int64_t j); // x += m[j]
  void addRow(const Matrix& m, int64_t j, real s); // x += m[j] * s
  void mul(const Matrix& A, const Vector& y); // x = A y
  int64_t argmax();
};

std::ostream& operator<<(std::ostream&, const Vector&);

} // namespace fasttext
```


##  Matrix

```cpp
#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

#include <assert.h>
#include "real.h"

namespace fasttext {

class Vector;

class Matrix {
 protected:
  int64_t m_;
  int64_t n_;

 public:
  Matrix();
  explicit Matrix(int64_t, int64_t);
  virtual ~Matrix() = default;

  int64_t size(int64_t dim) const;

  virtual real dotRow(const Vector& vec, int64_t i) const = 0; // matrix[i] = matrix[i] * vec
  virtual void addVectorToRow(const Vector& vec, int64_t i, real a) = 0; // matrix[i] += vec * a
  virtual void addRowToVector(Vector& x, int32_t i) const = 0; // x += matrix[i]
  virtual void addRowToVector(Vector& x, int32_t i, real a) const = 0; // x += matrix[i] * a
  virtual void save(std::ostream&) const = 0;
  virtual void load(std::istream&) = 0;
  virtual void dump(std::ostream&) const = 0;
};

} // namespace fasttext
```

## DenseMatrix

```cpp
#pragma once

#include <assert.h>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "matrix.h"
#include "real.h"

namespace fasttext {

class Vector;

class DenseMatrix : public Matrix {
 protected:
  std::vector<real> data_;
  void uniformThread(real, int, int32_t);

 public:
  DenseMatrix();
  explicit DenseMatrix(int64_t, int64_t);
  explicit DenseMatrix(int64_t m, int64_t n, real* dataPtr);
  DenseMatrix(const DenseMatrix&) = default;
  DenseMatrix(DenseMatrix&&) noexcept;
  DenseMatrix& operator=(const DenseMatrix&) = delete;
  DenseMatrix& operator=(DenseMatrix&&) = delete;
  virtual ~DenseMatrix() noexcept override = default;

  inline real* data() {
    return data_.data();
  }
  inline const real* data() const {
    return data_.data();
  }

  inline const real& at(int64_t i, int64_t j) const { // 取下标操作
    assert(i * n_ + j < data_.size());
    return data_[i * n_ + j];
  };
  inline real& at(int64_t i, int64_t j) {
    return data_[i * n_ + j];
  };

  inline int64_t rows() const { // 返回行数
    return m_;
  }
  inline int64_t cols() const { // 返回列数
    return n_;
  }
  void zero(); // 将矩阵置为全0
  void uniform(real, unsigned int, int32_t); // 将矩阵随机化为 uniform 分布

  void multiplyRow(const Vector& nums, int64_t ib = 0, int64_t ie = -1); // 对于 matrix 在 [ib, ie) 区间的行，都乘以 nums
  void divideRow(const Vector& denoms, int64_t ib = 0, int64_t ie = -1); // 对于 matrix 在 [ib, ie) 区间的行，都除以 nums

  real l2NormRow(int64_t i) const; // 计算第 i 行的 l2-norm
  void l2NormRow(Vector& norms) const; // 计算 m 行的 l2-norm，将结果写入 norms 中

  real dotRow(const Vector&, int64_t) const override;
  void addVectorToRow(const Vector&, int64_t, real) override;
  void addRowToVector(Vector& x, int32_t i) const override;
  void addRowToVector(Vector& x, int32_t i, real a) const override;
  void save(std::ostream&) const override;
  void load(std::istream&) override;
  void dump(std::ostream&) const override;

  class EncounteredNaNError : public std::runtime_error {
   public:
    EncounteredNaNError() : std::runtime_error("Encountered NaN.") {}
  };
};
} // namespace fasttext
```

####  uniformThread 和 uniform

```cpp
void DenseMatrix::uniformThread(real a, int block, int32_t seed) {
  std::minstd_rand rng(block + seed);
  std::uniform_real_distribution<> uniform(-a, a);
  int64_t blockSize = (m_ * n_) / 10; //分成 10 份？ 为什么是分成 10 份？不应该是分成的块数等于线程数吗？
  for (int64_t i = blockSize * block; // 填充 block 所在的那一份
       i < (m_ * n_) && i < blockSize * (block + 1);
       i++) {
    data_[i] = uniform(rng);
  }
}

void DenseMatrix::uniform(real a, unsigned int thread, int32_t seed) {
  if (thread > 1) {
    std::vector<std::thread> threads;
    for (int i = 0; i < thread; i++) {
      threads.push_back(std::thread([=]() { uniformThread(a, i, seed); }));
    }
    for (int32_t i = 0; i < threads.size(); i++) {
      threads[i].join();
    }
  } else {
    // webassembly can't instantiate `std::thread`
    uniformThread(a, 0, seed);
  }
}
```