#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.center_loss_param().num_output();  
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.center_loss_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> center_shape(2);
    center_shape[0] = N_;
    center_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(center_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(
        this->layer_param_.center_loss_param().center_filler()));
    center_filler->Fill(this->blobs_[0].get());

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // read 
  vector<int> mult_dims(1, N_);
  num_.Reshape(mult_dims);
  Dtype* num_data = num_.mutable_cpu_data();
  const string& source = this->layer_param_.center_loss_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  int num;
  int label;
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    num = atoi(line.substr(pos + 1).c_str());
    label = atoi(line.substr(0, pos).c_str());
    caffe_set(1, Dtype(num), num_data + label);
    LOG(INFO)<< label << " " << num << " " << num_data[label];
  }
  ca_loss_weight_ = this->layer_param_.center_loss_param().ca_loss_weight();
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  M_ = bottom[0]->num();
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  LossLayer<Dtype>::Reshape(bottom, top);
  distance_.ReshapeLike(*bottom[0]);
  variation_sum_.ReshapeLike(*this->blobs_[0]);


  vector<int> lambda_dims = bottom[0]->shape();
  lambda_dims[1] = 1;
  lambda_.Reshape(lambda_dims);

}

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* center = this->blobs_[0]->cpu_data();
  Dtype* distance_data = distance_.mutable_cpu_data();

  // for center aligned
  Dtype* lambda_data = lambda_.mutable_cpu_data(); 
  Dtype center_norm = Dtype(0); 
  for (int i = 0; i < N_; i++) {
    Dtype class_i_norm = caffe_cpu_dot(K_, center + i * K_, center + i * K_);
    center_norm += class_i_norm / N_ ;
  }
  tau_ = center_norm;


  // the i-th distance_data
  for (int i = 0; i < M_; i++) {
    const int label_value = static_cast<int>(label[i]);
    // D(i,:) = X(i,:) - C(y(i),:)
    caffe_sub(K_, bottom_data + i * K_, center + label_value * K_, distance_data + i * K_);

    // for ca loss
    Dtype lambda = caffe_cpu_dot(K_, center + label_value * K_, center + label_value * K_);
    lambda_data[i]  =  lambda - tau_;
  }

  // center loss
  Dtype dot = caffe_cpu_dot(M_ * K_, distance_.cpu_data(), distance_.cpu_data());
  Dtype loss_center = dot / M_ / Dtype(2);
  // center aligned loss
  Dtype dot_ca = caffe_cpu_dot(M_, lambda_data, lambda_data);
  Dtype loss_ca = dot_ca / M_ / Dtype(4);


  // cia only
  top[0]->mutable_cpu_data()[0] = loss_ca;

  // center + cia
  // top[0]->mutable_cpu_data()[0] = loss_center + loss_ca;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

const Dtype* lambda_data = lambda_.mutable_cpu_data();
const Dtype* label = bottom[1]->cpu_data();
const Dtype* center = this->blobs_[0]->cpu_data();
const Dtype* num_data = num_.mutable_cpu_data();
const Dtype* distance_data = distance_.cpu_data();


  // Gradient with respect to centers
  if (this->param_propagate_down_[0]) {
    // const Dtype* label = bottom[1]->cpu_data();
    Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* variation_sum_data = variation_sum_.mutable_cpu_data();

    // \sum_{y_i==j}
    caffe_set(N_ * K_, (Dtype)0., variation_sum_.mutable_cpu_data());
    for (int n = 0; n < N_; n++) {
      int count = 0;
      for (int m = 0; m < M_; m++) {
        const int label_value = static_cast<int>(label[m]);
        if (label_value == n) {
          count++;
          caffe_sub(K_, variation_sum_data + n * K_, distance_data + m * K_, variation_sum_data + n * K_);
        }
      }
      caffe_axpy(K_, (Dtype)1./(count + (Dtype)1.), variation_sum_data + n * K_, center_diff + n * K_);
    }
  }
  // Gradient with respect to bottom data 
  if (propagate_down[0]) {
    Dtype* bottom_diff_data = bottom[0]->mutable_cpu_diff();
     for (int i = 0; i < M_; i++){
        const int label_value = static_cast<int>(label[i]);
        Dtype scale = Dtype(0);
        const int num = static_cast<int> (num_data[label_value]);
        scale =  ca_loss_weight_ * lambda_data[i] * ((Dtype)1. - (Dtype)1. / N_) / num;
	// ca gradients
        caffe_axpy(K_, scale, center + label_value * K_, bottom_diff_data + i * K_);
        // center gradients
	// caffe_add(K_, distance_data + i * K_, bottom_diff_data + i * K_, bottom_diff_data + i * K_);
    }    

    // caffe_copy(M_ * K_, distance_.cpu_data(), bottom[0]->mutable_cpu_diff());

    caffe_scal(M_ * K_, top[0]->cpu_diff()[0] / M_, bottom[0]->mutable_cpu_diff());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(CenterLossLayer);
#endif

INSTANTIATE_CLASS(CenterLossLayer);
REGISTER_LAYER_CLASS(CenterLoss);

}  // namespace caffe
