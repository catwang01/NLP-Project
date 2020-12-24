#pragma once
#include <iostream>
#include <vector>
#include "real.h"

namespace fasttext {
    class Vector;
    class Matrix {
        private:
            int64_t m_, n_;
        public:
            Matrix();
            explicit Matrix(int64_t m, int64_t n);

            int64_t size(int64_t dim) const;
            virtual real dotRow(const Vector&, int64_t) const = 0;
            virtual void addVectorToRow(const Vector&, int64_t, real) = 0;
            virtual void addRowToVector(Vector& x, int32_t i) const = 0;
            virtual void addRowToVector(Vector& x, int32_t i, real a) const = 0;
            virtual void save(std::ostream&) const = 0;
            virtual void load(std::istream&) = 0;
            virtual void dump(std::ostream&) const = 0;
    };


};