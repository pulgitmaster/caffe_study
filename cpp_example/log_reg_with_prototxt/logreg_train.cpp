#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"
#include "google/protobuf/text_format.h"
#include "caffe/sgd_solvers.hpp"

using namespace caffe;
using std::cout;
using std::endl;
typedef double Dtype;

const int num_required_args = 3; // net, type, arch

// refernce : https://kezunlin.me/post/5898412/
// bool ReadProtoFromTextFile(const char* filename, Message* proto) {
//     Encryption encryption;
//   int fd = open(filename, O_RDONLY);
//   CHECK_NE(fd, -1) << "File not found: " << filename;
//   std::FileInputStream* input = new std::FileInputStream(fd);
//   bool success = google::protobuf::TextFormat::Parse(input, proto);
//   delete input;
//   close(fd);
//   return success;
// }

// bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
//   int fd = open(filename, O_RDONLY);
//   CHECK_NE(fd, -1) << "File not found: " << filename;
//   ZeroCopyInputStream* raw_input = new FileInputStream(fd);
//   CodedInputStream* coded_input = new CodedInputStream(raw_input);
//   coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

//   bool success = proto->ParseFromCodedStream(coded_input);

//   delete coded_input;
//   delete raw_input;
//   close(fd);
//   return success;
// }


int main(int argc, char** argv) {
    // set net
    string proto =
        "name: 'LogReg_train' "
        "layer { "
        "  name: 'data' "
        "  type: 'Data' "
        "  top: 'data' "
        "  top: 'label' "
        "  data_param { "
        "    source: 'train_leveldb' "
        "    batch_size: 200 "
        "  } "
        "} "
        "layer { "
        "  name: 'ip' "
        "  type: 'InnerProduct' "
        "  bottom: 'data' "
        "  top: 'ip' "
        "  inner_product_param { "
        "    num_output: 2 "
        "  } "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'SoftmaxWithLoss' "
        "  bottom: 'ip' "
        "  bottom: 'label' "
        "  top: 'loss' "
        "} ";

    NetParameter param_net;
    google::protobuf::TextFormat::ParseFromString(proto, &param_net);

    // if (argc < num_required_args) {
    //     return 1;
    // }

    // for(int i=0; i<num_required_args; i++){
    //     if(strcmp(argv[i], "GPU") == 0){
    //         Caffe::set_mode(Caffe::GPU);
    //     }
    //     else if(strcmp(argv[i], "CPU") == 0){
    //         Caffe::set_mode(Caffe::CPU);
    //     }
    // }

    google::protobuf::Message *message;


    SolverParameter param_solver;
    param_solver.set_allocated_net_param(&param_net);
    param_solver.set_base_lr(0.001);
    param_solver.set_max_iter(2000);
    param_solver.set_lr_policy("inv");
    param_solver.set_momentum(0.9);
    param_solver.set_gamma(0.0001);
    param_solver.set_snapshot(1000);
	param_solver.set_snapshot_prefix("logreg"); // file name prefix
    param_solver.set_display(50);
    param_solver.set_solver_mode(SolverParameter_SolverMode_GPU);
    
    // training
    SGDSolver<Dtype> solver(param_solver);
    solver.Solve();


    

    //std::string pretrained_caffemodel(argv[])

}
