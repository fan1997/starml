#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/operators/concat.h"
#include "starml/operators/transpose.h"
#include "gtest/gtest.h"
#include <Eigen/Core>
#include <Eigen/Dense>

#include <ctime>
using namespace starml;
using std::cout;
using std::endl;
TEST(CONCAT, test){
  int m = 4;
  int n = 3;
  Matrix origin_data({m, n}, kCPU, kFloat);
  for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
          origin_data.data<float>()[i * n + j] = i + 1;
      }
  }
  origin_data.print();
  Matrix origin_data_concat = concat(origin_data, origin_data, 1);
  Matrix origin_data_concat1 = concat(origin_data, origin_data);
  origin_data_concat.print();
  origin_data_concat1.print();

  //gpu
  // Matrix origin_data_cuda = origin_data.to(kCUDA);
  // origin_data_cuda.print();
  // Matrix origin_data_concat_cuda =  concat(origin_data_cuda, origin_data_cuda, 1);
  // origin_data_concat_cuda.print();
  // origin_data_cuda.print();
  // Matrix origin_data_concat_cuda1 =  concat(origin_data_cuda, origin_data_cuda);
  // origin_data_concat_cuda1.print();
  //test wrong
  // Matrix origin_data_concat_wrong = concat(origin_data, transpose(origin_data), 1);
  // Matrix origin_data_concat_wrong1 = concat(origin_data, transpose(origin_data), 1);

//test eigen
  // #define MATRIX_SIZE 100
  //
  //     Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > A1;
  //     A1 = Eigen::MatrixXd::Random( MATRIX_SIZE, MATRIX_SIZE );
  //
  //     Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > b1;
  //     b1 = Eigen::MatrixXd::Random( MATRIX_SIZE, 2 );
  //
  //     clock_t time_stt = clock(); // 计时
  //     // 直接求逆
  //     Eigen::Matrix<double,MATRIX_SIZE,1> x = A1.inverse()*b1;
  //     cout <<"time use in normal inverse is " << 1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms"<< endl;
  //     cout<<x<<endl;
  //     // QR分解colPivHouseholderQr()
  //     time_stt = clock();
  //     x = A1.colPivHouseholderQr().solve(b1);
  //     cout <<"time use in Qr decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
  //     cout <<x<<endl;
  //     //QR分解fullPivHouseholderQr()
  //     time_stt = clock();
  //     x = A1.fullPivHouseholderQr().solve(b1);
  //     cout <<"time use in Qr decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
  //     cout <<x<<endl;
  //     /* //llt分解 要求矩阵A正定
  //     time_stt = clock();
  //     x = A1.llt().solve(b1);
  //     cout <<"time use in llt decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
  //     cout <<x<<endl;*/
  //     /*//ldlt分解  要求矩阵A正或负半定
  //     time_stt = clock();
  //     x = A1.ldlt().solve(b1);
  //     cout <<"time use in ldlt decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
  //     cout <<x<<endl;*/
  //     //lu分解 partialPivLu()
  //     time_stt = clock();
  //     x = A1.partialPivLu().solve(b1);
  //     cout <<"time use in lu decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
  //     cout <<x<<endl;
  //     //lu分解（fullPivLu()
  //     time_stt = clock();
  //     x = A1.fullPivLu().solve(b1);
  //     cout <<"time use in lu decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
  //     cout <<x<<endl;


}
