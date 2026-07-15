/**
 * @file residual_engine.hpp
 * @brief TensorRT residual multi-layer perceptron between encoder and decoder.
 *
 * Contract (three-input / single-output ONNX):
 *   - Input obs_dict or obs: token + delta history + decoder his_* (no height map)
 *   - Input decoder_history: decoder his_* only (930)
 *   - Input height_map_2d_history5 or height_map_3d: (T, H, W) elevation history
 *   - Output delta_token or actions: float32, typically 64
 *
 * The deployment adds tanh(output) * 0.2 to the encoder token and stores the same
 * scaled delta in obs_dict history before filling policy observations.
 */

#ifndef RESIDUAL_ENGINE_HPP
#define RESIDUAL_ENGINE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <TRTInference/InferenceEngine.h>

/**
 * @class ResidualEngine
 * @brief Loads model_residual.onnx (via TensorRT) and runs one inference per control step.
 */
class ResidualEngine {
 public:
  ResidualEngine() = default;
  ~ResidualEngine() { destroy(); }

  bool initialize(const std::string& model_path, bool use_fp16 = false) {
    if (model_path.empty()) {
      std::cerr << "ResidualEngine::initialize: model path is empty" << std::endl;
      return false;
    }

    configuration_.model_path = model_path;
    configuration_.use_fp16 = use_fp16;

    try {
      std::cout << "Loading residual model..." << std::endl;
      inference_engine_ = std::make_unique<TRTInferenceEngine>();

      Options options;
      options.deviceID = configuration_.device_id;  // same field name as EncoderEngine
      std::string cache_prefix("residual_");
      if (use_fp16) {
        options.precision = Precision::FP16;
        cache_prefix += "fp16_";
      }

      std::string cached_tensorrt_path;
      if (!ConvertONNXToTRT(options, model_path, cached_tensorrt_path, cache_prefix, false)) {
        std::cerr << "ResidualEngine::initialize: TensorRT conversion failed for " << model_path << std::endl;
        inference_engine_.reset();
        return false;
      }

      if (!inference_engine_->Initialize(cached_tensorrt_path, options.deviceID, options.dynamic_axes_names)) {
        std::cerr << "ResidualEngine::initialize: TensorRT load failed: " << cached_tensorrt_path << std::endl;
        inference_engine_.reset();
        return false;
      }

      if (!inference_engine_->InitInputs({})) {
        std::cerr << "ResidualEngine::initialize: InitInputs failed" << std::endl;
        inference_engine_.reset();
        return false;
      }

      auto input_names = inference_engine_->GetInputTensorNames();
      if (input_names.size() != 3) {
        std::cerr << "ResidualEngine::initialize: expected exactly three inputs, found "
                  << input_names.size() << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (!resolve_input_tensor_name(input_names, {"obs_dict", "obs"}, obs_dict_tensor_name_)) {
        std::cerr << "ResidualEngine::initialize: missing obs input (obs_dict or obs). Available: ";
        for (const auto& name : input_names) {
          std::cerr << name << " ";
        }
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (!resolve_input_tensor_name(input_names, {"decoder_history"}, decoder_history_tensor_name_)) {
        std::cerr << "ResidualEngine::initialize: missing decoder_history input. Available: ";
        for (const auto& name : input_names) {
          std::cerr << name << " ";
        }
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (!resolve_input_tensor_name(input_names,
                                     {"height_map_2d_history5", "height_map_2d_history5_noise", "height_map_3d_noise"},
                                     height_map_tensor_name_)) {
        std::cerr << "ResidualEngine::initialize: missing height map input (height_map_2d_history5 or "
                     "height_map_3d). Available: ";
        for (const auto& name : input_names) {
          std::cerr << name << " ";
        }
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      for (const char* name :
           {obs_dict_tensor_name_.c_str(), decoder_history_tensor_name_.c_str(), height_map_tensor_name_.c_str()}) {
        if (inference_engine_->GetTensorDataType(name) != DataType::FLOAT) {
          std::cerr << "ResidualEngine::initialize: " << name << " must be float32" << std::endl;
          inference_engine_.reset();
          return false;
        }
      }

      std::vector<int64_t> obs_dict_dimensions;
      if (!inference_engine_->GetTensorShape(obs_dict_tensor_name_, obs_dict_dimensions)) {
        std::cerr << "ResidualEngine::initialize: failed to read obs_dict shape" << std::endl;
        inference_engine_.reset();
        return false;
      }
      configuration_.obs_dict_element_count =
          static_cast<size_t>(std::accumulate(obs_dict_dimensions.begin(), obs_dict_dimensions.end(),
                                              static_cast<int64_t>(1), std::multiplies<int64_t>()));
      obs_dict_input_buffer_.resize(configuration_.obs_dict_element_count, 0.0f);
      inference_engine_->SetInputData(obs_dict_tensor_name_, obs_dict_input_buffer_);

      std::vector<int64_t> decoder_history_dimensions;
      if (!inference_engine_->GetTensorShape(decoder_history_tensor_name_, decoder_history_dimensions)) {
        std::cerr << "ResidualEngine::initialize: failed to read decoder_history shape" << std::endl;
        inference_engine_.reset();
        return false;
      }
      configuration_.decoder_history_element_count =
          static_cast<size_t>(std::accumulate(decoder_history_dimensions.begin(), decoder_history_dimensions.end(),
                                              static_cast<int64_t>(1), std::multiplies<int64_t>()));
      decoder_history_input_buffer_.resize(configuration_.decoder_history_element_count, 0.0f);
      inference_engine_->SetInputData(decoder_history_tensor_name_, decoder_history_input_buffer_);

      std::vector<int64_t> height_map_dimensions;
      if (!inference_engine_->GetTensorShape(height_map_tensor_name_, height_map_dimensions)) {
        std::cerr << "ResidualEngine::initialize: failed to read height map shape" << std::endl;
        inference_engine_.reset();
        return false;
      }
      configuration_.height_map_element_count =
          static_cast<size_t>(std::accumulate(height_map_dimensions.begin(), height_map_dimensions.end(),
                                                static_cast<int64_t>(1), std::multiplies<int64_t>()));
      height_map_input_buffer_.resize(configuration_.height_map_element_count, 0.0f);
      inference_engine_->SetInputData(height_map_tensor_name_, height_map_input_buffer_);

      auto output_names = inference_engine_->GetOutputTensorNames();
      if (output_names.size() != 1) {
        std::cerr << "ResidualEngine::initialize: expected exactly one output, found " << output_names.size()
                  << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (std::find(output_names.begin(), output_names.end(), std::string("delta_token")) != output_names.end()) {
        output_tensor_name_ = "delta_token";
      } else if (std::find(output_names.begin(), output_names.end(), std::string("actions")) != output_names.end()) {
        output_tensor_name_ = "actions";
      } else {
        std::cerr << "ResidualEngine::initialize: output tensor delta_token or actions is missing" << std::endl;
        inference_engine_.reset();
        return false;
      }
      std::vector<int64_t> output_dimensions;
      if (!inference_engine_->GetTensorShape(output_tensor_name_, output_dimensions)) {
        std::cerr << "ResidualEngine::initialize: failed to read output shape" << std::endl;
        inference_engine_.reset();
        return false;
      }
      configuration_.output_element_count =
          static_cast<size_t>(std::accumulate(output_dimensions.begin(), output_dimensions.end(), static_cast<int64_t>(1),
                                              std::multiplies<int64_t>()));
      delta_token_buffer_.resize(configuration_.output_element_count, 0.0f);

      cudaError_t cuda_status = cudaStreamCreate(&cuda_stream_);
      if (cuda_status != cudaSuccess) {
        std::cerr << "ResidualEngine::initialize: cudaStreamCreate failed: " << cudaGetErrorString(cuda_status)
                  << std::endl;
        inference_engine_.reset();
        return false;
      }

      is_initialized_ = true;
      std::cout << "Residual model initialized." << std::endl;
      std::cout << "  Path: " << model_path << std::endl;
      std::cout << "  Input " << obs_dict_tensor_name_ << " elements: " << configuration_.obs_dict_element_count
                << std::endl;
      std::cout << "  Input " << decoder_history_tensor_name_
                << " elements: " << configuration_.decoder_history_element_count << std::endl;
      std::cout << "  Input " << height_map_tensor_name_
                << " elements: " << configuration_.height_map_element_count << std::endl;
      std::cout << "  Output (" << output_tensor_name_ << ") elements: " << configuration_.output_element_count
                << std::endl;
      return true;
    } catch (const std::exception& exception) {
      std::cerr << "ResidualEngine::initialize: exception: " << exception.what() << std::endl;
      inference_engine_.reset();
      return false;
    }
  }

  bool infer(cudaStream_t stream = nullptr) {
    if (!is_initialized_ || !inference_engine_) {
      std::cerr << "ResidualEngine::infer: not initialized" << std::endl;
      return false;
    }
    cudaStream_t inference_stream = (stream != nullptr) ? stream : cuda_stream_;
    inference_engine_->SetInputData(obs_dict_tensor_name_, obs_dict_input_buffer_);
    inference_engine_->SetInputData(decoder_history_tensor_name_, decoder_history_input_buffer_);
    inference_engine_->SetInputData(height_map_tensor_name_, height_map_input_buffer_);
    if (!inference_engine_->Enqueue(inference_stream)) {
      std::cerr << "ResidualEngine::infer: enqueue failed" << std::endl;
      return false;
    }
    inference_engine_->GetOutputDataAsync(output_tensor_name_, delta_token_buffer_, inference_stream);
    cudaStreamSynchronize(inference_stream);
    return true;
  }

  size_t get_obs_dict_element_count() const { return configuration_.obs_dict_element_count; }
  size_t get_decoder_history_element_count() const { return configuration_.decoder_history_element_count; }
  size_t get_height_map_element_count() const { return configuration_.height_map_element_count; }
  size_t get_output_element_count() const { return configuration_.output_element_count; }

  TPinnedVector<float>& get_obs_dict_buffer() { return obs_dict_input_buffer_; }
  TPinnedVector<float>& get_decoder_history_buffer() { return decoder_history_input_buffer_; }
  TPinnedVector<float>& get_height_map_buffer() { return height_map_input_buffer_; }
  TPinnedVector<float>& get_delta_token_buffer() { return delta_token_buffer_; }

  bool is_initialized() const { return is_initialized_; }

  void destroy() {
    if (!is_initialized_) {
      return;
    }
    if (cuda_stream_ != nullptr) {
      cudaStreamDestroy(cuda_stream_);
      cuda_stream_ = nullptr;
    }
    if (inference_engine_) {
      inference_engine_->Destroy();
      inference_engine_.reset();
    }
    is_initialized_ = false;
  }

 private:
  static bool resolve_input_tensor_name(const std::vector<std::string>& input_names,
                                        std::initializer_list<const char*> candidates, std::string& resolved) {
    for (const char* candidate : candidates) {
      if (std::find(input_names.begin(), input_names.end(), std::string(candidate)) != input_names.end()) {
        resolved = candidate;
        return true;
      }
    }
    return false;
  }

  struct Configuration {
    std::string model_path;
    int device_id = 0;
    size_t obs_dict_element_count = 0;
    size_t decoder_history_element_count = 0;
    size_t height_map_element_count = 0;
    size_t output_element_count = 0;
    bool use_fp16 = false;
  };
  Configuration configuration_;

  std::unique_ptr<TRTInferenceEngine> inference_engine_;
  std::string obs_dict_tensor_name_;
  std::string decoder_history_tensor_name_;
  std::string height_map_tensor_name_;
  std::string output_tensor_name_;
  cudaStream_t cuda_stream_ = nullptr;

  TPinnedVector<float> obs_dict_input_buffer_;
  TPinnedVector<float> decoder_history_input_buffer_;
  TPinnedVector<float> height_map_input_buffer_;
  TPinnedVector<float> delta_token_buffer_;

  bool is_initialized_ = false;
};

#endif  // RESIDUAL_ENGINE_HPP
