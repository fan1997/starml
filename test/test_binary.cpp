#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/operators/binary_ops.h"
#include "gtest/gtest.h"

using namespace starml;

TEST(BINARY, test){
    int m = 4;
    int n = 3;
    Matrix origin_data({m, n}, kCPU, kFloat);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            origin_data.mutable_data<float>()[i * n + j] = i + 1;
        }
    }
    origin_data.print();
    Matrix res = add(origin_data, origin_data);
    res.print();

    Matrix res_sub = sub(origin_data, origin_data);
    res_sub.print();

    res_sub = sub(origin_data, res);
    res_sub.print();

    res_sub = sub(1, origin_data);
    res_sub.print();

    res_sub = div(1.0, origin_data);
    res_sub.print();
    res_sub = div(origin_data, 5);
    res_sub.print();


    Matrix origin_data_cuda = origin_data.to(kCUDA);
    Matrix res_sub_cuda = sub(origin_data_cuda, origin_data_cuda);
    res_sub_cuda = sub(1.0, origin_data_cuda);
    res_sub_cuda.print();
}
