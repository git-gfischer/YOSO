/**
 * yoso_infer_trt.cpp  –  YOSO TensorRT C++ Inference
 *
 * Build
 * -----
 *   cd cpp && mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make -j$(nproc)
 *
 * Run
 * ---
 *   ./yoso_infer_trt --engine yoso_res50.engine \
 *                    --image  /path/to/image.jpg \
 *                    [--score-thresh 0.5]         \
 *                    [--mask-thresh  0.5]          \
 *                    [--out-dir      ./results]
 *
 * Dependencies
 * ------------
 *   TensorRT ≥ 8.6,  CUDA,  OpenCV ≥ 4
 */

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include "coco_vis.hpp"
#include "infer_result.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// ─── constants ──────────────────────────────────────────────────
static const int   NUM_KERNELS   = 100;
static const int   NUM_CLASSES   = 134;  // 133 foreground + 1 background
static const int   NUM_FG_CLASSES = 133;  // exclude background index 133
static const int   TOPK_DETECTIONS = 100;
static const float YOSO_TEMPERATURE = 0.05f;  // cfg MODEL.YOSO.TEMPERATIRE (YOSO-R50.yaml)
// Detectron2-style normalization used by YOSO configs (RGB, 0..255 scale):
//   PIXEL_MEAN: [123.675, 116.280, 103.530]
//   PIXEL_STD : [58.395, 57.120, 57.375]
static const float PIXEL_MEAN[3] = {123.675f, 116.280f, 103.530f};
static const float PIXEL_STD[3]  = {58.395f, 57.120f, 57.375f};
// Alternate normalization used by some standalone exports (RGB in [0,1]).
static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

// ─── COCO panoptic class names in cocovis::class_names() ────────

// ─── TRT logger ─────────────────────────────────────────────────
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity sev, const char* msg) noexcept override {
        if (sev <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << '\n';
    }
};

// ─── GPU buffer helper ──────────────────────────────────────────
struct GpuBuffer {
    void*  ptr  = nullptr;
    size_t bytes = 0;

    void alloc(size_t n) {
        bytes = n;
        cudaMalloc(&ptr, n);
    }
    void free()  { cudaFree(ptr); ptr = nullptr; }
    ~GpuBuffer() { free(); }
};

// ─── Detection result ───────────────────────────────────────────
struct Detection {
    int           class_id;
    float         score;
    std::string   class_name;
    cv::Mat       mask;          // CV_8UC1, original image size
};

// ─── YOSO TRT inference class ───────────────────────────────────
class YOSOInference {
public:
    explicit YOSOInference(const std::string& engine_path,
                           float score_thresh = 0.9f,
                           float mask_thresh  = 0.7f,
                           bool things_only   = false,
                           float min_area_ratio = 0.0f,
                           float max_area_ratio = 0.98f,
                           bool debug_preds = false,
                           bool input_bgr = false,
                           bool no_norm = false)
        : score_thresh_(score_thresh), mask_thresh_(mask_thresh),
          things_only_(things_only),
          min_area_ratio_(min_area_ratio),
          max_area_ratio_(max_area_ratio),
          debug_preds_(debug_preds),
          input_bgr_(input_bgr),
          no_norm_(no_norm)
    {
        load_engine(engine_path);
        alloc_buffers();
    }

    ~YOSOInference() {
        for (auto& b : gpu_bufs_) b.free();
    }

    // Infer on a single image.
    InferResult<Detection> infer(const cv::Mat& bgr_image) {
        InferResult<Detection> result;
        auto t0 = now_ms();

        // ── 1. pre-process ───────────────────────────────────────
        auto [input_h, input_w] = get_input_dims();
        cv::Mat resized;
        cv::resize(bgr_image, resized, {input_w, input_h});

        std::vector<float> input_data = preprocess(resized, input_h, input_w);
        if (input_dtype_ == nvinfer1::DataType::kFLOAT) {
            cudaMemcpy(gpu_bufs_[input_idx_].ptr,
                       input_data.data(),
                       input_data.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
        } else if (input_dtype_ == nvinfer1::DataType::kHALF) {
            std::vector<uint16_t> input_fp16(input_data.size());
            for (size_t i = 0; i < input_data.size(); ++i) input_fp16[i] = fp32_to_fp16(input_data[i]);
            cudaMemcpy(gpu_bufs_[input_idx_].ptr,
                       input_fp16.data(),
                       input_fp16.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice);
        } else {
            throw std::runtime_error("Unsupported input dtype: " + std::string(dtype_name(input_dtype_)));
        }

        // ── 2. inference ────────────────────────────────────────
        context_->executeV2(bindings_.data());
        cudaStreamSynchronize(stream_);

        auto t1 = now_ms();

        // ── 3. copy outputs to host ──────────────────────────────
        const size_t logits_numel = static_cast<size_t>(num_kernels_logits_) *
                                    static_cast<size_t>(num_classes_logits_);
        const int mask_h = input_h / 4;
        const int mask_w = input_w / 4;
        const size_t masks_numel = static_cast<size_t>(num_kernels_masks_) *
                                   static_cast<size_t>(mask_h) *
                                   static_cast<size_t>(mask_w);
        std::vector<float> logits_host;
        std::vector<float> masks_host;
        copy_output_to_float(logits_name_, gpu_bufs_[logits_idx_].ptr,
                             logits_numel, logits_dtype_, logits_host);
        copy_output_to_float(masks_name_, gpu_bufs_[masks_idx_].ptr,
                             masks_numel, masks_dtype_, masks_host);
        if (debug_preds_) {
            debug_print_predictions(logits_host, masks_host, mask_h, mask_w);
        }

        // ── 4. post-process ──────────────────────────────────────
        result.detections = postprocess(logits_host, masks_host,
                                        bgr_image.rows, bgr_image.cols,
                                        mask_h, mask_w);

        auto t2 = now_ms();
        result.preprocess_ms  = t1 - t0;
        result.postprocess_ms = t2 - t1;
        result.infer_ms       = 0.0;
        result.total_ms       = t2 - t0;
        std::cout << "[YOSO] pre=" << result.preprocess_ms
                  << "ms  post=" << result.postprocess_ms
                  << "ms  total=" << result.total_ms
                  << "ms  fps=" << std::fixed << std::setprecision(1)
                  << result.fps() << '\n';
        return result;
    }

private:
    enum class LogitsLayout {
        KERNELS_CLASSES,  // [B, K, C]
        CLASSES_KERNELS   // [B, C, K]
    };

    static bool has_dynamic_dim(const nvinfer1::Dims& dims) {
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] < 0) return true;
        }
        return false;
    }

    static const char* dtype_name(nvinfer1::DataType t) {
        switch (t) {
            case nvinfer1::DataType::kFLOAT: return "FP32";
            case nvinfer1::DataType::kHALF:  return "FP16";
            case nvinfer1::DataType::kINT8:  return "INT8";
            case nvinfer1::DataType::kINT32: return "INT32";
            case nvinfer1::DataType::kBOOL:  return "BOOL";
            default: return "UNKNOWN";
        }
    }

    static size_t dtype_size(nvinfer1::DataType t) {
        switch (t) {
            case nvinfer1::DataType::kFLOAT: return 4;
            case nvinfer1::DataType::kHALF:  return 2;
            case nvinfer1::DataType::kINT8:  return 1;
            case nvinfer1::DataType::kINT32: return 4;
            case nvinfer1::DataType::kBOOL:  return 1;
            default: return 0;
        }
    }

    static float fp16_to_fp32(uint16_t h) {
        const uint32_t sign = (h & 0x8000u) << 16;
        uint32_t exp = (h & 0x7C00u) >> 10;
        uint32_t mant = h & 0x03FFu;
        uint32_t out = 0;
        if (exp == 0) {
            if (mant == 0) {
                out = sign;
            } else {
                exp = 1;
                while ((mant & 0x0400u) == 0) {
                    mant <<= 1;
                    --exp;
                }
                mant &= 0x03FFu;
                out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
            }
        } else if (exp == 0x1Fu) {
            out = sign | 0x7F800000u | (mant << 13);
        } else {
            out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &out, sizeof(float));
        return f;
    }

    static uint16_t fp32_to_fp16(float f) {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(uint32_t));
        const uint32_t sign = (x >> 16) & 0x8000u;
        int32_t exp = int32_t((x >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = x & 0x7FFFFFu;
        if (exp <= 0) {
            if (exp < -10) return static_cast<uint16_t>(sign);
            mant = (mant | 0x800000u) >> (1 - exp);
            return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
        }
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(sign | (uint32_t(exp) << 10) | ((mant + 0x1000u) >> 13));
    }

    void copy_output_to_float(const std::string& name,
                              void* device_ptr,
                              size_t numel,
                              nvinfer1::DataType dtype,
                              std::vector<float>& out) const {
        out.resize(numel);
        if (dtype == nvinfer1::DataType::kFLOAT) {
            cudaMemcpy(out.data(), device_ptr, numel * sizeof(float), cudaMemcpyDeviceToHost);
            return;
        }
        if (dtype == nvinfer1::DataType::kHALF) {
            std::vector<uint16_t> tmp(numel);
            cudaMemcpy(tmp.data(), device_ptr, numel * sizeof(uint16_t), cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < numel; ++i) out[i] = fp16_to_fp32(tmp[i]);
            return;
        }
        throw std::runtime_error("Unsupported output dtype for '" + name + "': " + dtype_name(dtype));
    }

    // ── engine loading ──────────────────────────────────────────
    void load_engine(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open engine: " + path);

        file.seekg(0, std::ios::end);
        auto size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> data(size);
        file.read(data.data(), size);

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
        if (!engine_) throw std::runtime_error("Failed to deserialize engine");
        context_.reset(engine_->createExecutionContext());

        std::cout << "[YOSO] Engine loaded: " << path
                  << "  (" << size/1e6 << " MB)\n";

        // Map tensor names to binding indices (robust against renamed outputs).
        int output_count = 0;
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* tname = engine_->getIOTensorName(i);
            std::string name = tname;
            auto mode = engine_->getTensorIOMode(tname);
            auto dims = engine_->getTensorShape(tname);

            std::cout << "  binding[" << i << "] " << name << " mode="
                      << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
                      << " dims=";
            for (int d = 0; d < dims.nbDims; ++d) std::cout << dims.d[d] << " ";
            std::cout << "\n";

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                input_idx_ = i;
                input_name_ = name;
                continue;
            }

            ++output_count;
            if (dims.nbDims == 3) {
                logits_idx_ = i;
                logits_name_ = name;
            } else if (dims.nbDims == 4) {
                masks_idx_ = i;
                masks_name_ = name;
            }
        }

        // Fallback heuristics by name if dims-based mapping did not resolve.
        if (logits_name_.empty() || masks_name_.empty()) {
            for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
                const char* tname = engine_->getIOTensorName(i);
                std::string name = tname;
                auto mode = engine_->getTensorIOMode(tname);
                if (mode != nvinfer1::TensorIOMode::kOUTPUT) continue;
                if (logits_name_.empty() && name.find("logit") != std::string::npos) {
                    logits_idx_ = i;
                    logits_name_ = name;
                } else if (masks_name_.empty() && name.find("mask") != std::string::npos) {
                    masks_idx_ = i;
                    masks_name_ = name;
                }
            }
        }

        if (input_name_.empty() || logits_name_.empty() || masks_name_.empty() || output_count < 2) {
            throw std::runtime_error(
                "Failed to resolve input/logits/masks tensor mapping from TensorRT engine.");
        }
        std::cout << "[YOSO] Tensor map: input='" << input_name_
                  << "' logits='" << logits_name_
                  << "' masks='" << masks_name_ << "'\n";

        auto logits_dims = engine_->getTensorShape(logits_name_.c_str());
        if (logits_dims.nbDims != 3) {
            throw std::runtime_error("Unexpected logits rank; expected 3D [B,*,*].");
        }
        int l1 = logits_dims.d[1];
        int l2 = logits_dims.d[2];
        if (l1 == NUM_KERNELS && l2 == NUM_CLASSES) {
            logits_layout_ = LogitsLayout::KERNELS_CLASSES;
        } else if (l1 == NUM_CLASSES && l2 == NUM_KERNELS) {
            logits_layout_ = LogitsLayout::CLASSES_KERNELS;
        } else {
            // Conservative fallback: assume [B,K,C] and cap by known constants.
            logits_layout_ = LogitsLayout::KERNELS_CLASSES;
            std::cerr << "[YOSO][WARN] Unexpected logits dims (" << l1 << "," << l2
                      << "), falling back to [B,K,C] decoding.\n";
        }

        auto masks_dims = engine_->getTensorShape(masks_name_.c_str());
        if (masks_dims.nbDims != 4) {
            throw std::runtime_error("Unexpected masks rank; expected 4D [B,K,H,W].");
        }
        input_dtype_ = engine_->getTensorDataType(input_name_.c_str());
        logits_dtype_ = engine_->getTensorDataType(logits_name_.c_str());
        masks_dtype_ = engine_->getTensorDataType(masks_name_.c_str());
        num_kernels_logits_ = (logits_layout_ == LogitsLayout::KERNELS_CLASSES) ? l1 : l2;
        num_classes_logits_ = (logits_layout_ == LogitsLayout::KERNELS_CLASSES) ? l2 : l1;
        num_kernels_masks_ = masks_dims.d[1];

        std::cout << "[YOSO] Logits layout: "
                  << (logits_layout_ == LogitsLayout::KERNELS_CLASSES ? "[B,K,C]" : "[B,C,K]")
                  << "  kernels(logits)=" << num_kernels_logits_
                  << " classes=" << num_classes_logits_
                  << " kernels(masks)=" << num_kernels_masks_ << "\n";
        std::cout << "[YOSO] Tensor dtypes: input=" << dtype_name(input_dtype_)
                  << " logits=" << dtype_name(logits_dtype_)
                  << " masks=" << dtype_name(masks_dtype_) << "\n";
        cudaStreamCreate(&stream_);
    }

    // ── buffer allocation ────────────────────────────────────────
    void alloc_buffers() {
        int n = engine_->getNbIOTensors();
        gpu_bufs_.resize(n);
        bindings_.resize(n);

        for (int i = 0; i < n; ++i) {
            const char* tname = engine_->getIOTensorName(i);
            auto dims = engine_->getTensorShape(tname);
            auto dtype = engine_->getTensorDataType(tname);
            size_t vol = 1;
            for (int d = 0; d < dims.nbDims; ++d)
                vol *= (dims.d[d] > 0 ? dims.d[d] : 1);
            const size_t elem_bytes = dtype_size(dtype);
            if (elem_bytes == 0) {
                throw std::runtime_error("Unsupported tensor dtype for buffer alloc: " + std::string(tname));
            }
            gpu_bufs_[i].alloc(vol * elem_bytes);
            bindings_[i] = gpu_bufs_[i].ptr;
            context_->setTensorAddress(tname, gpu_bufs_[i].ptr);
        }
    }

    // ── pre-processing ───────────────────────────────────────────
    std::vector<float> preprocess(const cv::Mat& bgr, int H, int W) const {
        // Convert input to requested channel order, then optional normalization.
        std::vector<float> out(3 * H * W);
        const int area = H * W;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                auto px = bgr.at<cv::Vec3b>(y, x);
                int idx = y * W + x;
                float c0 = input_bgr_ ? static_cast<float>(px[0]) : static_cast<float>(px[2]);
                float c1 = static_cast<float>(px[1]);
                float c2 = input_bgr_ ? static_cast<float>(px[2]) : static_cast<float>(px[0]);
                if (!no_norm_) {
                    c0 = (c0 - PIXEL_MEAN[0]) / PIXEL_STD[0];
                    c1 = (c1 - PIXEL_MEAN[1]) / PIXEL_STD[1];
                    c2 = (c2 - PIXEL_MEAN[2]) / PIXEL_STD[2];
                } else {
                    c0 /= 255.f;
                    c1 /= 255.f;
                    c2 /= 255.f;
                }
                out[0 * area + idx] = c0;
                out[1 * area + idx] = c1;
                out[2 * area + idx] = c2;
            }
        }
        return out;
    }

    // ── sigmoid helper ────────────────────────────────────────────
    static float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

    // YOSO instance_inference-style post-processing (Detectron2 segmentator.py).
    std::vector<Detection> postprocess(
        const std::vector<float>& logits,
        const std::vector<float>& masks,
        int orig_h, int orig_w,
        int mask_h, int mask_w) const
    {
        std::vector<Detection> results;
        const int K = std::min({NUM_KERNELS, num_kernels_logits_, num_kernels_masks_});
        const int C = std::min(NUM_CLASSES, num_classes_logits_);
        if (K <= 0 || C <= 1) return results;

        struct Cand {
            int kernel;
            int cls;
            float cls_score;
        };
        std::vector<Cand> cands;
        cands.reserve(static_cast<size_t>(K) * NUM_FG_CLASSES);

        // Global top-k over all (kernel, class) pairs (exclude background class).
        for (int k = 0; k < K; ++k) {
            std::vector<float> row(C);
            if (logits_layout_ == LogitsLayout::KERNELS_CLASSES) {
                const float* src = logits.data() + k * num_classes_logits_;
                for (int c = 0; c < C; ++c) row[c] = src[c] / YOSO_TEMPERATURE;
            } else {
                for (int c = 0; c < C; ++c)
                    row[c] = logits[c * num_kernels_logits_ + k] / YOSO_TEMPERATURE;
            }

            float max_v = *std::max_element(row.begin(), row.end());
            float sum = 0.f;
            std::vector<float> probs(C);
            for (int c = 0; c < C; ++c) {
                probs[c] = std::exp(row[c] - max_v);
                sum += probs[c];
            }
            for (auto& p : probs) p /= std::max(sum, 1e-8f);

            const int cls_limit = things_only_ ? std::min(NUM_FG_CLASSES, 80) : NUM_FG_CLASSES;
            for (int c = 0; c < std::min(cls_limit, C - 1); ++c)
                cands.push_back({k, c, probs[c]});
        }

        const size_t top_n = std::min<size_t>(TOPK_DETECTIONS, cands.size());
        std::partial_sort(cands.begin(), cands.begin() + top_n, cands.end(),
                          [](const Cand& a, const Cand& b) { return a.cls_score > b.cls_score; });

        for (size_t i = 0; i < top_n; ++i) {
            const auto& cand = cands[i];
            if (cand.cls_score < score_thresh_) continue;

            const float* mask_ptr = masks.data() + cand.kernel * mask_h * mask_w;
            cv::Mat low_mask(mask_h, mask_w, CV_32FC1);
            cv::Mat binary_low(mask_h, mask_w, CV_8UC1);
            float mask_sum = 0.f;
            int mask_count = 0;
            for (int y = 0; y < mask_h; ++y) {
                for (int x = 0; x < mask_w; ++x) {
                    const float logit = mask_ptr[y * mask_w + x];
                    const float prob = sigmoid(logit);
                    low_mask.at<float>(y, x) = prob;
                    const bool on = logit > 0.f;  // official YOSO uses logit > 0
                    binary_low.at<uchar>(y, x) = on ? 255 : 0;
                    if (on) {
                        mask_sum += prob;
                        ++mask_count;
                    }
                }
            }
            if (mask_count == 0) continue;

            const float mask_score = mask_sum / float(mask_count);
            const float final_score = cand.cls_score * mask_score;
            if (final_score < score_thresh_) continue;

            cv::Mat full_mask;
            cv::resize(low_mask, full_mask, {orig_w, orig_h}, 0, 0, cv::INTER_LINEAR);
            cv::Mat binary_mask;
            cv::threshold(full_mask, binary_mask, mask_thresh_, 255, cv::THRESH_BINARY);
            binary_mask.convertTo(binary_mask, CV_8UC1);

            const float area_ratio =
                float(cv::countNonZero(binary_mask)) / float(std::max(1, orig_h * orig_w));
            if (area_ratio < min_area_ratio_ || area_ratio > max_area_ratio_) continue;

            const auto& names = cocovis::class_names();
            std::string name = (cand.cls < (int)names.size())
                               ? names[cand.cls] : std::to_string(cand.cls);
            results.push_back({cand.cls, final_score, name, binary_mask});
        }

        std::sort(results.begin(), results.end(),
                  [](auto& a, auto& b){ return a.score > b.score; });
        return results;
    }

    void debug_print_predictions(const std::vector<float>& logits,
                                 const std::vector<float>& masks,
                                 int mask_h, int mask_w) const {
        const int K = std::min({NUM_KERNELS, num_kernels_logits_, num_kernels_masks_});
        const int C = std::min(NUM_CLASSES, num_classes_logits_);
        if (K <= 0 || C <= 1) return;

        struct Cand {
            float score;
            int kernel;
            int cls;
        };
        std::vector<Cand> best;
        best.reserve(K);

        float logits_min = std::numeric_limits<float>::infinity();
        float logits_max = -std::numeric_limits<float>::infinity();
        int logits_nan = 0;
        for (float v : logits) {
            if (std::isnan(v)) { ++logits_nan; continue; }
            logits_min = std::min(logits_min, v);
            logits_max = std::max(logits_max, v);
        }

        float masks_min = std::numeric_limits<float>::infinity();
        float masks_max = -std::numeric_limits<float>::infinity();
        int masks_nan = 0;
        for (float v : masks) {
            if (std::isnan(v)) { ++masks_nan; continue; }
            masks_min = std::min(masks_min, v);
            masks_max = std::max(masks_max, v);
        }

        const int class_limit = things_only_ ? std::min(C, 80) : C;
        const int search_end = std::max(1, std::min(class_limit, C - 1));
        for (int k = 0; k < K; ++k) {
            std::vector<float> row(C);
            if (logits_layout_ == LogitsLayout::KERNELS_CLASSES) {
                const float* src = logits.data() + k * num_classes_logits_;
                std::copy(src, src + C, row.begin());
            } else {
                for (int c = 0; c < C; ++c) row[c] = logits[c * num_kernels_logits_ + k];
            }

            float max_v = row[0] / YOSO_TEMPERATURE;
            for (int c = 1; c < C; ++c)
                max_v = std::max(max_v, row[c] / YOSO_TEMPERATURE);
            float sum = 0.f;
            std::vector<float> probs(C);
            for (int c = 0; c < C; ++c) {
                probs[c] = std::exp(row[c] / YOSO_TEMPERATURE - max_v);
                sum += probs[c];
            }
            for (auto& p : probs) p /= std::max(sum, 1e-8f);

            const int cls = int(std::max_element(probs.begin(), probs.begin() + search_end) - probs.begin());
            best.push_back({probs[cls], k, cls});
        }

        const size_t top_n = std::min<size_t>(3, best.size());
        std::partial_sort(best.begin(), best.begin() + top_n, best.end(),
                          [](const Cand& a, const Cand& b) { return a.score > b.score; });

        std::cout << "[YOSO][DBG] logits range=[" << logits_min << "," << logits_max
                  << "] nan=" << logits_nan
                  << " masks range=[" << masks_min << "," << masks_max
                  << "] nan=" << masks_nan << "\n";
        std::cout << "[YOSO][DBG] top predictions:";
        for (size_t i = 0; i < top_n; ++i) {
            const auto& c = best[i];
            const auto& names = cocovis::class_names();
            std::string name = (c.cls < (int)names.size()) ? names[c.cls] : std::to_string(c.cls);
            const float* mask_ptr = masks.data() + c.kernel * mask_h * mask_w;
            int active = 0;
            float mean = 0.f;
            for (int p = 0; p < mask_h * mask_w; ++p) {
                const float v = sigmoid(mask_ptr[p]);
                mean += v;
                if (v > mask_thresh_) ++active;
            }
            mean /= float(mask_h * mask_w);
            const float active_ratio = float(active) / float(mask_h * mask_w);
            std::cout << " [k=" << c.kernel
                      << " cls=" << name
                      << " score=" << std::fixed << std::setprecision(4) << c.score
                      << " mask_mean=" << mean
                      << " mask>th=" << active_ratio << "]";
        }
        std::cout << "\n";
    }

    // ── dims helper ───────────────────────────────────────────────
    std::pair<int,int> get_input_dims() const {
        auto dims = context_->getTensorShape(input_name_.c_str());
        if (has_dynamic_dim(dims)) {
            dims = engine_->getTensorShape(input_name_.c_str());
        }
        return {dims.d[2], dims.d[3]};   // H, W
    }

    static double now_ms() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    Logger                                          logger_;
    std::unique_ptr<nvinfer1::IRuntime>             runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>          engine_;
    std::unique_ptr<nvinfer1::IExecutionContext>    context_;
    cudaStream_t                                    stream_{};
    std::vector<GpuBuffer>                          gpu_bufs_;
    std::vector<void*>                              bindings_;
    int   input_idx_  = 0;
    int   logits_idx_ = 1;
    int   masks_idx_  = 2;
    std::string input_name_, logits_name_, masks_name_;
    LogitsLayout logits_layout_{LogitsLayout::KERNELS_CLASSES};
    int num_kernels_logits_{NUM_KERNELS};
    int num_classes_logits_{NUM_CLASSES};
    int num_kernels_masks_{NUM_KERNELS};
    nvinfer1::DataType input_dtype_{nvinfer1::DataType::kFLOAT};
    nvinfer1::DataType logits_dtype_{nvinfer1::DataType::kFLOAT};
    nvinfer1::DataType masks_dtype_{nvinfer1::DataType::kFLOAT};
    float score_thresh_;
    float mask_thresh_;
    bool  things_only_;
    bool  debug_preds_;
    bool  input_bgr_;
    bool  no_norm_;
    float min_area_ratio_;
    float max_area_ratio_;
};

// ─── main ────────────────────────────────────────────────────────
static void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " --engine ENGINE.engine --image IMAGE.jpg [options]\n"
              << "  " << prog << " --engine ENGINE.engine --webcam [options]\n\n"
              << "Options:\n"
              << "  --score-thresh X           Score threshold (default: 0.3)\n"
              << "  --mask-thresh X            Mask threshold (default: 0.5)\n"
              << "  --things-only              Use only 80 thing classes (default: all classes)\n"
              << "  --min-area X               Min mask area ratio in [0,1] (default: 0.0)\n"
              << "  --max-area X               Max mask area ratio in [0,1] (default: 0.98)\n"
              << "  --out-dir DIR              Output directory (default: ./results)\n"
              << "  --camera-id N              Webcam device index (default: 0)\n"
              << "  --cam-width N              Requested webcam width\n"
              << "  --cam-height N             Requested webcam height\n"
              << "  --cam-fps N                Requested webcam FPS\n"
              << "  --max-frames N             Stop after N frames in webcam mode\n"
              << "  --debug-preds              Print per-frame prediction stats\n"
              << "  --input-bgr                Keep BGR channel order (disable BGR->RGB swap)\n"
              << "  --no-norm                  Disable mean/std normalization (use 0..1 only)\n";
}

int main(int argc, char** argv) {
    std::string engine_path, image_path, out_dir = "./results";
    float score_thresh = 0.8f, mask_thresh = 0.7f;
    float min_area_ratio = 0.0f, max_area_ratio = 0.98f;
    bool things_only = false;
    bool debug_preds = false;
    bool input_bgr = false;
    bool no_norm = false;
    bool use_webcam = false;
    int camera_id = 0;
    int cam_width = 640, cam_height = 480, cam_fps = 30;
    int max_frames = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--engine" && i+1 < argc)       engine_path  = argv[++i];
        else if (a == "--image" && i+1 < argc)   image_path   = argv[++i];
        else if (a == "--webcam")                use_webcam   = true;
        else if (a == "--out-dir" && i+1 < argc) out_dir      = argv[++i];
        else if (a == "--score-thresh" && i+1 < argc) score_thresh = std::stof(argv[++i]);
        else if (a == "--mask-thresh"  && i+1 < argc) mask_thresh  = std::stof(argv[++i]);
        else if (a == "--things-only")               things_only = true;
        else if (a == "--min-area" && i+1 < argc)   min_area_ratio = std::stof(argv[++i]);
        else if (a == "--max-area" && i+1 < argc)   max_area_ratio = std::stof(argv[++i]);
        else if (a == "--camera-id" && i+1 < argc) camera_id = std::stoi(argv[++i]);
        else if (a == "--cam-width" && i+1 < argc) cam_width = std::stoi(argv[++i]);
        else if (a == "--cam-height" && i+1 < argc) cam_height = std::stoi(argv[++i]);
        else if (a == "--cam-fps" && i+1 < argc) cam_fps = std::stoi(argv[++i]);
        else if (a == "--max-frames" && i+1 < argc) max_frames = std::stoi(argv[++i]);
        else if (a == "--debug-preds")              debug_preds = true;
        else if (a == "--input-bgr")                input_bgr = true;
        else if (a == "--no-norm")                  no_norm = true;
        else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Require engine and exactly one input mode: --image xor --webcam
    if (engine_path.empty() || (image_path.empty() != use_webcam)) {
        print_usage(argv[0]);
        return 1;
    }

    min_area_ratio = std::max(0.0f, std::min(1.0f, min_area_ratio));
    max_area_ratio = std::max(0.0f, std::min(1.0f, max_area_ratio));
    if (min_area_ratio > max_area_ratio) std::swap(min_area_ratio, max_area_ratio);
    std::cout << "[YOSO] Preprocess: channel_order=" << (input_bgr ? "BGR" : "RGB")
              << " normalization=" << (no_norm ? "none(0..1)" : "detectron2") << "\n";
    YOSOInference yoso(engine_path, score_thresh, mask_thresh,
                       things_only, min_area_ratio, max_area_ratio,
                       debug_preds, input_bgr, no_norm);

    if (!image_path.empty()) {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "[ERROR] Cannot read image: " << image_path << '\n';
            return 1;
        }
        std::cout << "[YOSO] Image: " << image.cols << "×" << image.rows << '\n';

        auto result = yoso.infer(image);
        std::cout << "[YOSO] Detected " << result.detections.size() << " instances:\n";
        for (size_t i = 0; i < result.detections.size(); ++i)
            std::cout << "  [" << i << "] " << result.detections[i].class_name
                      << "  score=" << result.detections[i].score << '\n';

        std::filesystem::create_directories(out_dir);
        std::string stem  = std::filesystem::path(image_path).stem().string();
        std::string out_path = out_dir + "/" + stem + "_yoso.jpg";
        cv::Mat vis = cocovis::visualise(image, result.detections);
        cv::imwrite(out_path, vis);
        std::cout << "[YOSO] Result saved → " << out_path << '\n';
    } else {
        cv::VideoCapture cap(camera_id);
        if (!cap.isOpened()) {
            std::cerr << "[ERROR] Cannot open webcam device: " << camera_id << '\n';
            return 1;
        }
        if (cam_width > 0)  cap.set(cv::CAP_PROP_FRAME_WIDTH, cam_width);
        if (cam_height > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, cam_height);
        if (cam_fps > 0)    cap.set(cv::CAP_PROP_FPS, cam_fps);

        std::cout << "[YOSO] Webcam opened (device=" << camera_id
                  << ", " << int(cap.get(cv::CAP_PROP_FRAME_WIDTH))
                  << "x" << int(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
                  << ", fps=" << cap.get(cv::CAP_PROP_FPS) << ")\n";
        std::cout << "[YOSO] Thresholds: score=" << score_thresh
                  << " mask=" << mask_thresh << '\n';
        std::cout << "[YOSO] Press 'q' or ESC to quit.\n";

        cv::Mat frame, last_vis;
        int frame_count = 0;
        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            auto result = yoso.infer(frame);
            last_vis = cocovis::visualise(frame, result.detections);
            std::ostringstream hud;
            hud << "FPS: " << std::fixed << std::setprecision(1) << result.fps()
                << "  det=" << result.detections.size()
                << "  total=" << std::setprecision(0) << result.total_ms << "ms";
            cv::putText(last_vis, hud.str(),
                        {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        {0, 255, 0}, 2);
            cv::putText(last_vis, "Press q/ESC to quit",
                        {10, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        {255, 255, 255}, 2);
            cv::imshow("YOSO TRT Webcam", last_vis);

            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) break;

            ++frame_count;
            if (max_frames > 0 && frame_count >= max_frames) break;
        }
        cv::destroyAllWindows();

        if (!last_vis.empty()) {
            std::filesystem::create_directories(out_dir);
            std::string out_path = out_dir + "/webcam_yoso_trt.jpg";
            cv::imwrite(out_path, last_vis);
            std::cout << "[YOSO] Saved last webcam frame → " << out_path << '\n';
        }
    }

    return 0;
}
