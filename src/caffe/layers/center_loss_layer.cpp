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
  //vector<int> mult_dims(1, bottom[0]->shape(1));
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
    LOG(INFO)<< label << " " << num;
  }

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


// TODO
// optimize computation cost for lambda
// o(lambda) = min(#classes, #samples)


template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* center = this->blobs_[0]->cpu_data();
  Dtype* distance_data = distance_.mutable_cpu_data();
  Dtype* lambda_data = lambda_.mutable_cpu_data();

  Dtype center_norm = Dtype(0);
  // center norm 
  for (int i = 0; i < N_; i++) {
    // Dtype class_i_norm = sqrt(caffe_cpu_dot(K_, center + i * K_, center + i * K_));
    Dtype class_i_norm = caffe_cpu_dot(K_, center + i * K_, center + i * K_);
    // std::cout << "cneter_norm_" << i << " " << class_i_norm << std::endl; 
    

    // mean
    center_norm += class_i_norm / N_ ;
    // center_norm += caffe_cpu_dot(K_, center + i * K_, center + i * K_) / N_ ;

    // max
    // if (center_norm < class_i_norm ){
    //	center_norm = class_i_norm;
    // }
  }
  tau_ = center_norm;
  std::cout << "cneter_norm " << center_norm << std::endl;

  // the i-th distance_data
  for (int i = 0; i < M_; i++) {
    const int label_value = static_cast<int>(label[i]);
    // D(i,:) = X(i,:) - C(y(i),:)
    caffe_sub(K_, bottom_data + i * K_, center + label_value * K_, distance_data + i * K_);
    // 
    // Dtype lambda = sqrt(caffe_cpu_dot(K_, center + label_value * K_, center + label_value * K_));
    Dtype lambda = caffe_cpu_dot(K_, center + label_value * K_, center + label_value * K_);
    lambda_data[i]  =  lambda - tau_;
    // lambda_data[i]  = lambda / center_norm - 1;
    // lambda_data[i]  =  1 - lambda/center_norm;
    // lambda_data[i]  =   center_norm/lambda - 1;
    // if (label_value < 5){
    //   std::cout << "lambda " << lambda <<  " " << tau_ << " "<< lambda_data[i] <<std::endl;
    // }
    // std::cout << "lambda " << lambda << std::endl;
  }


  // Dtype dot = caffe_cpu_dot(M_ * K_, distance_.cpu_data(), distance_.cpu_data());
  Dtype dot = caffe_cpu_dot(M_, lambda_data, lambda_data);
  // Dtype dot = caffe_cpu_dot(M_, bottom_data, bottom_data);
  Dtype loss = dot / M_ / Dtype(4);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  // const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* lambda_data = lambda_.mutable_cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
  const Dtype* center = this->blobs_[0]->cpu_data();
const Dtype* num_data = num_.mutable_cpu_data();
  // Gradient with respect to centers
  if (this->param_propagate_down_[0]) {
    Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* variation_sum_data = variation_sum_.mutable_cpu_data();
    const Dtype* distance_data = distance_.cpu_data();

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
	// Dtype scale = (Dtype(1) - lambda_data[i]) * (Dtype(1) - lambda_data[i]);
	// Dtype scale =  Dtype(-1) + lambda_data[i];
        const int label_value = static_cast<int>(label[i]);
	Dtype scale = Dtype(0);
        // if (Dtype(1)  > lambda_data[i]){
	// scale =  lambda_data[i]  / num_data[label_value] ;
	// scale =  lambda_data[i] / tau_ / num_data[label_value] * ((Dtype)1. - (Dtype)1. / N_);
	scale =  lambda_data[i] * ((Dtype)1. - (Dtype)1. / N_) / num_data[label_value] ;
	// }
	
	caffe_axpy(K_, scale, center + label_value * K_, bottom_diff_data + i * K_);
	// caffe_axpy(K_, scale, bottom_data + i * K_, bottom_diff_data + i * K_);
    }
// std::cout << top[0]->cpu_diff()[0] << std::endl;
    caffe_scal(M_ * K_, top[0]->cpu_diff()[0] / M_, bottom_diff_data);

    // caffe_copy(M_ * K_, distance_.cpu_data(), bottom[0]->mutable_cpu_diff());
    // caffe_copy(M_ * K_, distance_.cpu_data(), bottom[0]->mutable_cpu_diff());
    // caffe_scal(M_ * K_, top[0]->cpu_diff()[0] / M_, bottom[0]->mutable_cpu_diff());
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
