#include "starml/basic/matrix.h"
#include "starml/dataloader/dataloader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <omp.h>
#include <cmath>

using std::fstream;
using std::stringstream;

namespace starml{

inline char *findlastline(char *ptr, char *begin) {
  while (ptr != begin && *ptr != '\n') --ptr;
  return ptr;
}
void DataLoader::load_from_libsvm(std::string file_path){
  std::cout << "loading from libsvm file..." << '\n';
  // developing

  // init
  label_.clear();
  instances_.clear();
  total_size_ = 0;
  num_instances_ = 0;
  num_features_ = 0;
  num_classes_ = 0;

  // open file
  std::ifstream fin(file_path, std::ifstream::binary); //check
  int buffer_size = 16 << 20; // 16MB
  char *buffer = (char *)malloc(buffer_size * sizeof(char)); // 16M B MEMORY
  const int nthread = omp_get_max_threads();
  std::cout << nthread << '\n';
  while (fin) {
    char *head = buffer;
    fin.read(buffer, buffer_size); //get next 16M char in buffer (containing ' ' and '/n')
    std::cout << "buffer[0]: " << buffer[0] << '\n';
    size_t size = fin.gcount();
    std::cout << "size:" << size << '\n';
    std::vector<std::vector<float>> label_thread(nthread); // label_thread[tid] is a float vector
    std::vector<sample_set> instances_thread(nthread); // instances_thread[tid] is a sample_set
    std::vector<int> local_max_feature(nthread, 0); //local_feature[tid] means the largeset num_features local
#pragma omp parallel num_threads(nthread)
    {
       int tid = omp_get_thread_num(); // local id
       //calc local working area
       //total size = size * (char)
       size_t local_size = (size + nthread - 1) / nthread; // size per thread
       size_t size_begin = std::min(tid * local_size, size - 1);
       size_t size_end = std::min((tid + 1) * local_size, size - 1);
       // head : buffer start
       char* line_begin = findlastline(head + size_begin, head); //fisrt place one thread to read
       char* line_end = findlastline(head + size_end, head);  // last place one thread to read
       //虽然目前 读了 gcount个字节进buffer， 但是由于按行处理，并不一定能处理完所有的字节, 所以
       // 不能fin.seekg(gcount), 这样就缺失了一部分
       //其中指针相减 得到 个数
       //这里 fin.read()之后，fin已经指向gcount之后个位置，因此需要向前偏移一些
       if (tid == nthread - 1) fin.seekg(line_end - head - size_end, std::ios_base::cur); //由 当前位置 偏移 多少 字节

       //each thread process line('/n') by line('/n') in its working area
       char* lbegin = line_begin;
       char* lend = lbegin;
       while (lend != line_end) {
         lend = lbegin + 1;
         while (lend != line_end && *lend != '\n') {
           ++lend;
         }
         std::string line(lbegin, lend);
         std::stringstream ss(line);
         // read label
         label_thread[tid].emplace_back();
         ss >> label_thread[tid].back();
         // read features
         instances_thread[tid].emplace_back();
         std::string tuple;
         while (ss >> tuple) {
             int i;
             float v;
             sscanf(tuple.c_str(), "%d:%f", &i, &v);
             instances_thread[tid].back().emplace_back(i, v);
             if (i > local_max_feature[tid]) local_max_feature[tid] = i;
         };
         //read next line
         lbegin = lend;
       }
   }  //pragma
    for (int i = 0; i < nthread; i++) {
      if (local_max_feature[i] > num_features_)
        num_features_ = local_max_feature[i];
      num_instances_ += instances_thread[i].size();
    }
    for (int i = 0; i < nthread; i++) {
      this->label_.insert(label_.end(), label_thread[i].begin(), label_thread[i].end());
      this->instances_.insert(instances_.end(), instances_thread[i].begin(), instances_thread[i].end());
    }
  } //while
  this->total_size_ = this->num_features_ * this->num_instances_;
}  //func




} //namespace starml
