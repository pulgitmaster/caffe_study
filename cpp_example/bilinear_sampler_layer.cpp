#include <algorithm>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/layers/bilinear_sampler_layer.hpp"

namespace caffe {
    template <typename Dtype>
    void BilinearSamplerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    //  validate number of bottom and top blobs
    //  CHECK_EQ(3,bottom.size())<< "We need 3 bottoms: Image, horizontal flow and verical flow";
    //  CHECK_EQ(1,top.size())<< "Output is only warp image";
    }

    template <typename Dtype>
    void BilinearSamplerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
            << "corresponding to (num, channels, height, width)";
        // num = bottom[0]->num(); // 0
        // channels_ = bottom[0]->channels(); // 1
        // height_ = bottom[0]->height(); // 2
        // width_ = bottom[0]->width(); // 3
        top[0]->Reshape(bottom[0]->num(),bottom[0]->channels() ,bottom[0]->height(),bottom[0]->width());

    }
}