#pragma once
#include <string>
#include <vector>
#include "starml/basic/matrix.h"

namespace starml {


enum class DatasetType : int { LIBSVM = 0, UCI = 1, CSV = 2 };
constexpr DatasetType kLIBSVM = DatasetType::LIBSVM;
constexpr DatasetType kUCI = DatasetType::UCI;
constexpr DatasetType kCSV = DatasetType::CSV;

class DataLoader {
 public:
   struct node{
     node(int index, float value) : index(index), value(value) {}
     int index;
     float value;
   };
   typedef std::vector<DataLoader::node> sample;   //样本
   typedef std::vector<sample> sample_set;  //稀疏 省内存

   DataLoader(): num_instances_(0), num_features_(0), total_size_(0), num_classes_(0) {};
   DataLoader(std::string file_path, DatasetType ds_type);  //load dataset when construct DataLoader
   void load_from_file(std::string file_path, DatasetType ds_type);
   void group_class();
   int get_num_instances() const {return this->num_instances_;}
   int get_num_features() const {return this->num_features_;}
   long int get_total_size() const {return this->total_size_;}
   const Matrix get_data() const;  // 返回引用(可能需要智能指针) 
   const Matrix get_label() const; //
   // const Matrix get_csr_data() const; // 稀疏

 protected:
   void load_from_libsvm(std::string file_path);
   void load_from_uci(std::string file_path);
   void load_from_csv(std::string file_path);

 private:
   int num_instances_;
   int num_features_;
   long int total_size_;
   int num_classes_;

   sample_set instances_;
   std::vector<float> label_;

};

}  // namespace starml
