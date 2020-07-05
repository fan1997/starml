#include <iostream>
#include <memory>
#include "gtest/gtest.h"
#include "starml/basic/matrix.h"
#include "starml/models/linear_regression/linear_regression.h"
#include "starml/dataloader/dataloader.h"

using namespace starml;
using namespace starml::regression;

class matrix_shared{
public:
    std::shared_ptr<float> ptr;
    matrix_shared(){
        ptr.reset(new float[100]);
        float* p_ptr = ptr.get();
        for (int i = 0; i < 100; i++) {
            p_ptr[i] = i;
        }
    }
};


class matrix_unique{
public:
    std::unique_ptr<float> ptr;
    matrix_unique(){
        ptr.reset(new float[100]);
        float* p_ptr = ptr.get();
        for (int i = 0; i < 100; i++) {
            p_ptr[i] = i;
        }
    }
};

matrix_shared try_matrix_shared(){
    matrix_shared a;
    return a;
}

matrix_unique try_matrix_unique(){
    matrix_unique a;
    return a;
}

std::shared_ptr<int> try_smart_ptr(){
   std::shared_ptr<int> p(new int[100]);
   int* p_ptr = p.get();
   for (int i = 0; i < 100; i++) {
       p_ptr[i] = i;
   }
   return p;
}

std::unique_ptr<int> try_unique_ptr(){
   std::unique_ptr<int> p(new int[100]);
   int* p_ptr = p.get();
   for (int i = 0; i < 100; i++) {
       p_ptr[i] = i;
   }
   return p;
}


int main(int argc, char **argv) {
  // ::testing::InitGoogleTest(&argc, argv);
  // starml::Matrix b;
  starml::Matrix a(2, 3, kCPU);
  float *data = a.data();
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      data[i * 3 + j] = i + 1;
    }
  }
  // starml::Matrix b(3, 3, kCPU);
  // b = a;
  // std::cout << data[0] << '\n';
  // std::cout << b.data()[0] << '\n';
  matrix_shared temp;
  matrix_shared temp1;
  temp1 = temp;
  std::cout << "temp.ptr.use_count() " << temp.ptr.use_count() <<'\n';
  std::cout << "temp1.ptr.use_count() " << temp1.ptr.use_count() <<'\n';

  matrix_shared* temp_ptr = new matrix_shared;
  matrix_shared* temp1_ptr = new matrix_shared;
  *temp1_ptr = *temp_ptr;
  std::cout << "temp.ptr.use_count() " << (*temp_ptr).ptr.use_count() <<'\n';
  std::cout << "temp1.ptr.use_count() " << (*temp1_ptr).ptr.use_count() <<'\n';
  delete temp_ptr;
  std::cout << "temp1.ptr.use_count() " << (*temp1_ptr).ptr.use_count() <<'\n';


  matrix_unique temp2;
  // matrix_unique temp3;
  // temp2 = temp3;

  temp = try_matrix_shared();
  temp2 = try_matrix_unique();


 #if 0
  LinearRegression model;
  LinearRegression model1(5.0);
  starml::Matrix train_data(2, 3, kCPU);
  starml::Matrix label(2, 3, kCPU);
  LinearRegression model2(a, label, 6.0);
  model.train(train_data, label);
  std::cout << "model lambda: " << model.get_lambda() << '\n';
  std::cout << "model1 lambda: " << model1.get_lambda() << '\n';
  std::cout << "model2 lambda: " << model2.get_lambda() << '\n';


  // DataLoader data_loader("./example", kLIBSVM);

  DataLoader data_loader("/gpfs/share/home/1901213502/dataset/dataset-all/a1a", kLIBSVM);
  // DataLoader data_loader;
  // data_loader.load_from_file("./example", kLIBSVM);
  long int total_size = data_loader.get_total_size();
  int num_instances = data_loader.get_num_instances();
  int num_features = data_loader.get_num_features();
  std::cout << "num_features: " << num_features << '\n';
  std::cout << "num_instances: " << num_instances << '\n';
  std::cout << "total_size: "<< total_size << '\n';
  Matrix label_1 = data_loader.get_label();
  Matrix data_1 = data_loader.get_data();
  // return RUN_ALL_TESTS();
  for (int i = 0; i < num_instances; i++) {
    std::cout << label_1.data()[i] << ' ' << '\n';
  }
  for (size_t i = 0; i < num_instances; i++) {
    for (size_t j = 0; j < num_features; j++) {
       std::cout << data_1.data()[i * num_features + j] << ' ';
    }
    std::cout << '\n';
  }
#endif

#if 0
  std::shared_ptr<int> p(new int[100]);  // pointer-like class
  int x = 42;
  int* x_ptr = &(x);

  std::cout << " p " << p << " *p " << *p << '\n'; // override *
  std::cout << " x_ptr " << x_ptr << " *x_ptr " << *x_ptr << '\n';
  std::cout << "p.get " << p.get()  << '\n'; // int*

  std::cout << "p.use_count() " << p.use_count() << '\n';
  std::shared_ptr<int> q(p);
  // q = p;
  std::cout << "p.use_count() " << p.use_count() << '\n';  // 2
  std::cout << "q.use_count() " << q.use_count() << '\n';  // 2

  p.reset();
  std::cout << "p.use_count() " << p.use_count() << '\n'; // 0
  std::cout << "q.use_count() " << q.use_count() << '\n'; // 1

  std::shared_ptr<int> xx = try_smart_ptr();
  std::cout << "xx.use_count() " << xx.use_count() << '\n';
  for (size_t i = 0; i < 10; i++) {
      std::cout << "xx" << xx.get()[i] << '\n';
  }
  xx.reset();

  std::unique_ptr<int> xxx = try_unique_ptr();
  // std::cout << "xx.use_count() " << xx.use_count() << '\n';
  for (size_t i = 0; i < 10; i++) {
      std::cout << "xxx " << xxx.get()[i] << '\n';
  }
  // xx.reset();

  // std::cout << "xx.use_count() " << xx.use_count() << '\n';
  // for (size_t i = 0; i < 10; i++) {
  //     std::cout << "xx" << xx.get()[i] << '\n';
  // }
#endif

}
