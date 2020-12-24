#pragma once
#include <vector>
#include <iostream>
#include "real.h"

namespace fasttext
{
    class Matrix;
    class Vector
    {
    private:
        std::vector<real> data_;

    public:
        inline real *data() { return data_.data(); }
        inline const real *data() const { return data_.data(); }
        inline real &operator[](int64_t i) { return data_[i]; }
        inline const real &operator[](int64_t i) const { return data_[i]; }
        inline int64_t size() const { return data_.size(); }
        void zero();    // 将值置为0
        void mul(real); // 整个 vector 乘以一个常数
        void mul(const Matrix &, const Vector &);
        real norm() const;                          // 求vector 的norm
        void addVector(const Vector &);             // 加上一个向量
        void addVector(const Vector &, real);       // 加上一个向量的倍数
        void addRow(const Matrix &, int64_t);       // 加上 matrix 中的某行
        void addRow(const Matrix &, int64_t, real); // 加上 matrix 中某行的整数倍
        int64_t argmax();
    };
    std::ostream &operator<<(std::ostream &, const Vector &);
}; // namespace