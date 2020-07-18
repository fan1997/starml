#include <iostream>
#include <string>
#include <string.h>
#include <omp.h>
#include "starml/basic/matrix.h"
#include "starml/dataloader/dataloader.h"

namespace starml {

  DataLoader::DataLoader(std::string file_path, DatasetType ds_type) {
    std::cout << "construct DataLoader, loading dataset from  "
              << file_path << " DatasetType : " << static_cast<int>(ds_type) << '\n';
    load_from_file(file_path, ds_type);
  }

  void DataLoader::load_from_file(std::string file_path, DatasetType ds_type) {
    std::cout << "loading dataset from  " << file_path
              << " DatasetType : " << static_cast<int>(ds_type) << '\n';
    int type_id = static_cast<int>(ds_type);
    switch (type_id) {
      case 0:
         load_from_libsvm(file_path);
         break;
      case 1:
         load_from_uci(file_path);
         break;
      case 2:
         load_from_csv(file_path);
         break;
      default: printf("Not supported Dataset file type...\n");
    }
  }

  void DataLoader::group_class(){
    std::cout << "grouping class..." << '\n';
  }

  Matrix DataLoader::get_data() const{
    std::cout << "getting data ..." << '\n';
    int m = this->num_instances_;
    int n = this->num_features_;
    Matrix data({m, n}, kCPU, kFloat);
    memset(data.data<float>(), 0.0, m * n * sizeof(float));
#pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < this->instances_[i].size(); j++) {
        int col_id = this->instances_[i][j].index - 1;
        float v = this->instances_[i][j].value;
        data.data<float>()[i * n + col_id] = v;
      }
    }
    return data;
  }

  Matrix DataLoader::get_label() const{
    std::cout << "getting label ..." << '\n';
    Matrix label({this->num_instances_, 1}, kCPU, kFloat);
    for (size_t i = 0; i < this->num_instances_; i++) {
      label.data<float>()[i] = this->label_[i];
    }
    return label;
  }

} // namespace starml
