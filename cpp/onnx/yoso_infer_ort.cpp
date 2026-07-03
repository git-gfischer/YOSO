/**
 * yoso_infer_ort.cpp  –  YOSO ONNX Runtime C++ Inference
 *
 * Works on CPU or CUDA without TensorRT.
 * Easier to deploy; recommended when TRT is unavailable.
 *
 * Build (see CMakeLists.txt)
 * --------------------------
 *   cd cpp && mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release \
 *             -DORT_ROOT=/path/to/onnxruntime
 *   make -j$(nproc)
 *
 * Run
 * ---
 *   ./yoso_infer_ort --model yoso_res50.onnx  \
 *                    --image /path/to/image.jpg \
 *                    [--gpu]                    \
 *                    [--score-thresh 0.5]        \
 *                    [--mask-thresh  0.5]         \
 *                    [--out-dir ./results]
 *
 * Dependencies
 * ------------
 *   ONNX Runtime ≥ 1.16  (https://github.com/microsoft/onnxruntime/releases)
 *   OpenCV ≥ 4
 */

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "coco_vis.hpp"
#include "infer_result.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ─── constants ──────────────────────────────────────────────────
static const int   NUM_KERNELS   = 100;
static const int   NUM_CLASSES   = 134;
static const int   NUM_FG_CLASSES = 133;
static const int   TOPK_DETECTIONS = 100;
static const float YOSO_TEMPERATURE = 0.05f;
static const float PIXEL_MEAN[3] = {123.675f, 116.280f, 103.530f};
static const float PIXEL_STD[3]  = {58.395f, 57.120f, 57.375f};

// COCO class names in cocovis::class_names()

// ─── Detection ──────────────────────────────────────────────────
struct Detection {
    int         class_id;
    float       score;
    std::string class_name;
    cv::Mat     mask;
};

// ─── Timing util ─────────────────────────────────────────────────
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// ─── YOSO ORT Inference ─────────────────────────────────────────
class YOSOInferenceORT {
public:
    YOSOInferenceORT(const std::string& model_path,
                     int input_h = 512, int input_w = 512,
                     float score_thresh = 0.5f,
                     float mask_thresh  = 0.5f,
                     bool  use_gpu      = false)
        : input_h_(input_h), input_w_(input_w),
          score_thresh_(score_thresh), mask_thresh_(mask_thresh)
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        bool using_cuda_ep = false;
        if (use_gpu) {
            // Try CUDA EP first; if it fails at runtime library load, fallback to CPU.
            try {
                OrtCUDAProviderOptions cuda_opts;
                cuda_opts.device_id = 0;
                opts.AppendExecutionProvider_CUDA(cuda_opts);
                std::cout << "[ORT] Attempting CUDA execution provider\n";
                session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
                using_cuda_ep = true;
            } catch (const Ort::Exception& e) {
                std::cerr << "[ORT][WARN] CUDA EP initialization failed: " << e.what() << "\n";
                std::cerr << "[ORT][WARN] Falling back to CPU execution provider.\n";
            }
        }
        if (!session_) {
            Ort::SessionOptions cpu_opts;
            cpu_opts.SetIntraOpNumThreads(4);
            cpu_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), cpu_opts);
        }
        std::cout << "[ORT] Execution provider: " << (using_cuda_ep ? "CUDA" : "CPU") << "\n";

        // Query I/O names
        Ort::AllocatorWithDefaultOptions alloc;
        const size_t input_count = session_->GetInputCount();
        const size_t output_count = session_->GetOutputCount();
        input_names_store_.reserve(input_count);
        output_names_store_.reserve(output_count);
        input_names_.reserve(input_count);
        output_names_.reserve(output_count);
        for (size_t i = 0; i < input_count; ++i) {
            auto name = session_->GetInputNameAllocated(i, alloc);
            input_names_store_.push_back(name.get());
            auto shape = session_->GetInputTypeInfo(i)
                                    .GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "[ORT] Input  " << i << ": " << name.get()
                      << "  shape=";
            for (auto d : shape) std::cout << d << " ";
            std::cout << '\n';
            if (i == 0 && shape.size() == 4 && shape[2] > 0 && shape[3] > 0) {
                input_h_ = static_cast<int>(shape[2]);
                input_w_ = static_cast<int>(shape[3]);
                std::cout << "[ORT] Using static model input size: "
                          << input_w_ << "x" << input_h_ << "\n";
            }
        }
        for (size_t i = 0; i < output_count; ++i) {
            auto name = session_->GetOutputNameAllocated(i, alloc);
            output_names_store_.push_back(name.get());
            auto shape = session_->GetOutputTypeInfo(i)
                                    .GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "[ORT] Output " << i << ": " << name.get()
                      << "  shape=";
            for (auto d : shape) std::cout << d << " ";
            std::cout << '\n';
        }
        for (const auto& n : input_names_store_) input_names_.push_back(n.c_str());
        for (const auto& n : output_names_store_) output_names_.push_back(n.c_str());
    }

    InferResult<Detection> infer(const cv::Mat& bgr_image) {
        InferResult<Detection> result;
        auto t0 = now_ms();

        // ── 1. pre-process ───────────────────────────────────────
        cv::Mat resized;
        cv::resize(bgr_image, resized, {input_w_, input_h_});
        std::vector<float> input_data = preprocess(resized);

        // ── 2. build input tensor ────────────────────────────────
        std::vector<int64_t> shape = {1, 3, input_h_, input_w_};
        auto mem_info = Ort::MemoryInfo::CreateCpu(
            OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            input_data.data(), input_data.size(),
            shape.data(), shape.size());

        // ── 3. run ───────────────────────────────────────────────
        auto t1 = now_ms();
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), &input_tensor, 1,
            output_names_.data(), output_names_.size());
        auto t2 = now_ms();

        // ── 4. extract outputs ───────────────────────────────────
        float* logits_ptr = outputs[0].GetTensorMutableData<float>();
        float* masks_ptr  = outputs[1].GetTensorMutableData<float>();

        auto logits_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        auto masks_shape  = outputs[1].GetTensorTypeAndShapeInfo().GetShape();

        int mask_h = static_cast<int>(masks_shape[2]);
        int mask_w = static_cast<int>(masks_shape[3]);

        // ── 5. post-process ──────────────────────────────────────
        result.detections = postprocess(logits_ptr, masks_ptr,
                                        bgr_image.rows, bgr_image.cols,
                                        mask_h, mask_w);
        auto t3 = now_ms();

        result.preprocess_ms  = t1 - t0;
        result.infer_ms       = t2 - t1;
        result.postprocess_ms = t3 - t2;
        result.total_ms       = t3 - t0;
        std::cout << "[ORT] pre=" << int(result.preprocess_ms)
                  << "ms  infer=" << int(result.infer_ms)
                  << "ms  post=" << int(result.postprocess_ms)
                  << "ms  total=" << int(result.total_ms)
                  << "ms  fps=" << std::fixed << std::setprecision(1)
                  << result.fps() << '\n';
        return result;
    }

private:
    static std::vector<float> preprocess(const cv::Mat& bgr) {
        int H = bgr.rows, W = bgr.cols;
        std::vector<float> out(3 * H * W);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                auto px  = bgr.at<cv::Vec3b>(y, x);
                int  idx = y * W + x;
                out[0 * H*W + idx] = (px[2] - PIXEL_MEAN[0]) / PIXEL_STD[0];
                out[1 * H*W + idx] = (px[1] - PIXEL_MEAN[1]) / PIXEL_STD[1];
                out[2 * H*W + idx] = (px[0] - PIXEL_MEAN[2]) / PIXEL_STD[2];
            }
        }
        return out;
    }

    static float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

    std::vector<Detection> postprocess(
        float* logits, float* masks,
        int orig_h, int orig_w,
        int mask_h, int mask_w) const
    {
        std::vector<Detection> results;

        struct Cand { int kernel; int cls; float cls_score; };
        std::vector<Cand> cands;
        cands.reserve(NUM_KERNELS * NUM_FG_CLASSES);

        for (int k = 0; k < NUM_KERNELS; ++k) {
            float* row = logits + k * NUM_CLASSES;
            float max_v = row[0] / YOSO_TEMPERATURE;
            for (int c = 1; c < NUM_CLASSES; ++c)
                max_v = std::max(max_v, row[c] / YOSO_TEMPERATURE);

            float sum = 0.f;
            std::vector<float> probs(NUM_CLASSES);
            for (int c = 0; c < NUM_CLASSES; ++c) {
                probs[c] = std::exp(row[c] / YOSO_TEMPERATURE - max_v);
                sum += probs[c];
            }
            for (auto& p : probs) p /= std::max(sum, 1e-8f);

            for (int c = 0; c < NUM_FG_CLASSES; ++c)
                cands.push_back({k, c, probs[c]});
        }

        const size_t top_n = std::min<size_t>(TOPK_DETECTIONS, cands.size());
        std::partial_sort(cands.begin(), cands.begin() + top_n, cands.end(),
                          [](const Cand& a, const Cand& b){ return a.cls_score > b.cls_score; });

        for (size_t i = 0; i < top_n; ++i) {
            const auto& cand = cands[i];
            if (cand.cls_score < score_thresh_) continue;

            float* mptr = masks + cand.kernel * mask_h * mask_w;
            cv::Mat low(mask_h, mask_w, CV_32FC1);
            float mask_sum = 0.f;
            int mask_count = 0;
            for (int y = 0; y < mask_h; ++y) {
                for (int x = 0; x < mask_w; ++x) {
                    const float logit = mptr[y * mask_w + x];
                    const float prob = sigmoid(logit);
                    low.at<float>(y, x) = prob;
                    if (logit > 0.f) { mask_sum += prob; ++mask_count; }
                }
            }
            if (mask_count == 0) continue;

            const float final_score = cand.cls_score * (mask_sum / float(mask_count));
            if (final_score < score_thresh_) continue;

            cv::Mat full;
            cv::resize(low, full, {orig_w, orig_h}, 0, 0, cv::INTER_LINEAR);
            cv::Mat binary;
            cv::threshold(full, binary, mask_thresh_, 255, cv::THRESH_BINARY);
            binary.convertTo(binary, CV_8UC1);

            const auto& names = cocovis::class_names();
            std::string name = (cand.cls < (int)names.size())
                               ? names[cand.cls] : std::to_string(cand.cls);
            results.push_back({cand.cls, final_score, name, binary});
        }

        std::sort(results.begin(), results.end(),
                  [](auto& a, auto& b){ return a.score > b.score; });
        return results;
    }

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "YOSO"};
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string>      input_names_store_, output_names_store_;
    std::vector<const char*>      input_names_,       output_names_;
    int   input_h_, input_w_;
    float score_thresh_, mask_thresh_;
};

// ─── main ────────────────────────────────────────────────────────
static void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " --model yoso_res50.onnx --image img.jpg [options]\n"
              << "  " << prog << " --model yoso_res50.onnx --webcam [options]\n\n"
              << "Options:\n"
              << "  --gpu                      Use CUDA execution provider\n"
              << "  --height N                 Model input height (default: 512)\n"
              << "  --width N                  Model input width (default: 512)\n"
              << "  --score-thresh X           Score threshold (default: 0.5)\n"
              << "  --mask-thresh X            Mask threshold (default: 0.5)\n"
              << "  --out-dir DIR              Output directory (default: ./results)\n"
              << "  --camera-id N              Webcam device index (default: 0)\n"
              << "  --cam-width N              Requested webcam width\n"
              << "  --cam-height N             Requested webcam height\n"
              << "  --cam-fps N                Requested webcam FPS\n"
              << "  --max-frames N             Stop after N frames in webcam mode\n";
}

int main(int argc, char** argv) {
    std::string model_path, image_path, out_dir = "./results";
    float score_thresh = 0.5f, mask_thresh = 0.5f;
    bool  use_gpu = false;
    int   input_h = 512, input_w = 512;
    bool  use_webcam = false;
    int   camera_id = 4;
    int   cam_width = 640, cam_height = 480, cam_fps = 30;
    int   max_frames = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"  && i+1<argc) model_path  = argv[++i];
        else if (a == "--image"  && i+1<argc) image_path  = argv[++i];
        else if (a == "--webcam")              use_webcam  = true;
        else if (a == "--gpu")                 use_gpu     = true;
        else if (a == "--out-dir"&& i+1<argc) out_dir     = argv[++i];
        else if (a == "--height" && i+1<argc) input_h     = std::stoi(argv[++i]);
        else if (a == "--width"  && i+1<argc) input_w     = std::stoi(argv[++i]);
        else if (a == "--score-thresh" && i+1<argc) score_thresh = std::stof(argv[++i]);
        else if (a == "--mask-thresh"  && i+1<argc) mask_thresh  = std::stof(argv[++i]);
        else if (a == "--camera-id"  && i+1<argc) camera_id  = std::stoi(argv[++i]);
        else if (a == "--cam-width"  && i+1<argc) cam_width  = std::stoi(argv[++i]);
        else if (a == "--cam-height" && i+1<argc) cam_height = std::stoi(argv[++i]);
        else if (a == "--cam-fps"    && i+1<argc) cam_fps    = std::stoi(argv[++i]);
        else if (a == "--max-frames" && i+1<argc) max_frames = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Require model and exactly one input mode: --image xor --webcam
    if (model_path.empty() || (image_path.empty() != use_webcam)) {
        print_usage(argv[0]);
        return 1;
    }

    YOSOInferenceORT yoso(model_path, input_h, input_w,
                          score_thresh, mask_thresh, use_gpu);

    if (!image_path.empty()) {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) { std::cerr << "Cannot read: " << image_path << '\n'; return 1; }
        std::cout << "[ORT] Image: " << image.cols << "×" << image.rows << '\n';

        auto result = yoso.infer(image);
        std::cout << "[ORT] Detections: " << result.detections.size() << '\n';
        for (size_t i = 0; i < result.detections.size(); ++i)
            std::cout << "  [" << i << "] " << result.detections[i].class_name
                      << "  score=" << result.detections[i].score << '\n';

        std::filesystem::create_directories(out_dir);
        std::string stem = std::filesystem::path(image_path).stem().string();
        std::string out_path = out_dir + "/" + stem + "_yoso_ort.jpg";
        cv::imwrite(out_path, cocovis::visualise(image, result.detections));
        std::cout << "[ORT] Saved → " << out_path << '\n';
    } else {
        cv::VideoCapture cap(camera_id);
        if (!cap.isOpened()) {
            std::cerr << "[ORT] Cannot open webcam device " << camera_id << '\n';
            return 1;
        }
        if (cam_width > 0)  cap.set(cv::CAP_PROP_FRAME_WIDTH, cam_width);
        if (cam_height > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, cam_height);
        if (cam_fps > 0)    cap.set(cv::CAP_PROP_FPS, cam_fps);

        std::cout << "[ORT] Webcam opened (device=" << camera_id
                  << ", " << int(cap.get(cv::CAP_PROP_FRAME_WIDTH))
                  << "x" << int(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
                  << ", fps=" << cap.get(cv::CAP_PROP_FPS) << ")\n";
        std::cout << "[ORT] Press 'q' or ESC to quit.\n";

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
            cv::imshow("YOSO ORT Webcam", last_vis);

            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) break;

            ++frame_count;
            if (max_frames > 0 && frame_count >= max_frames) break;
        }
        cv::destroyAllWindows();

        if (!last_vis.empty()) {
            std::filesystem::create_directories(out_dir);
            std::string out_path = out_dir + "/webcam_yoso_ort.jpg";
            cv::imwrite(out_path, last_vis);
            std::cout << "[ORT] Saved last webcam frame → " << out_path << '\n';
        }
    }

    return 0;
}
