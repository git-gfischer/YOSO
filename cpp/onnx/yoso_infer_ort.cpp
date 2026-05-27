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
static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

static const std::vector<std::string> CLASS_NAMES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
    "banner","blanket","bridge","cardboard","counter","curtain","door-stuff",
    "floor-wood","flower","fruit","gravel","house","light","mirror-stuff",
    "net","pillow","platform","playingfield","railroad","river","road",
    "roof","sand","sea","shelf","snow","stairs","tent","towel",
    "wall-brick","wall-stone","wall-tile","wall-wood","water-other",
    "window-blind","window-other","tree-merged","fence-merged","ceiling-merged",
    "sky-other-merged","cabinet-merged","table-merged","floor-other-merged",
    "pavement-merged","mountain-merged","grass-merged","dirt-merged",
    "paper-merged","food-other-merged","building-other-merged",
    "rock-merged","wall-other-merged","rug-merged",
    "background"
};

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

    std::vector<Detection> infer(const cv::Mat& bgr_image) {
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
        auto dets = postprocess(logits_ptr, masks_ptr,
                                bgr_image.rows, bgr_image.cols,
                                mask_h, mask_w);
        auto t3 = now_ms();

        std::cout << "[ORT] pre=" << int(t1-t0) << "ms  infer=" << int(t2-t1)
                  << "ms  post=" << int(t3-t2) << "ms  total=" << int(t3-t0) << "ms\n";
        return dets;
    }

private:
    static std::vector<float> preprocess(const cv::Mat& bgr) {
        int H = bgr.rows, W = bgr.cols;
        std::vector<float> out(3 * H * W);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                auto px  = bgr.at<cv::Vec3b>(y, x);
                int  idx = y * W + x;
                out[0 * H*W + idx] = (px[2] / 255.f - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                out[1 * H*W + idx] = (px[1] / 255.f - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                out[2 * H*W + idx] = (px[0] / 255.f - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
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

        for (int k = 0; k < NUM_KERNELS; ++k) {
            float* row   = logits + k * NUM_CLASSES;
            float  max_v = *std::max_element(row, row + NUM_CLASSES);
            float  sum   = 0.f;
            std::vector<float> probs(NUM_CLASSES);
            for (int c = 0; c < NUM_CLASSES; ++c) {
                probs[c]  = std::exp(row[c] - max_v);
                sum      += probs[c];
            }
            for (auto& p : probs) p /= sum;

            int   best_c = int(std::max_element(probs.begin(),
                                                probs.end()-1) - probs.begin());
            float score  = probs[best_c];
            if (score < score_thresh_) continue;

            // Build low-res mask
            float* mptr = masks + k * mask_h * mask_w;
            cv::Mat low(mask_h, mask_w, CV_32FC1);
            for (int i = 0; i < mask_h * mask_w; ++i)
                low.at<float>(i / mask_w, i % mask_w) = sigmoid(mptr[i]);

            // Upsample
            cv::Mat full;
            cv::resize(low, full, {orig_w, orig_h}, 0, 0, cv::INTER_LINEAR);
            cv::Mat binary;
            cv::threshold(full, binary, mask_thresh_, 255, cv::THRESH_BINARY);
            binary.convertTo(binary, CV_8UC1);

            std::string name = (best_c < (int)CLASS_NAMES.size())
                               ? CLASS_NAMES[best_c] : std::to_string(best_c);
            results.push_back({best_c, score, name, binary});
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

// ─── visualisation ───────────────────────────────────────────────
static cv::Mat visualise(const cv::Mat& image,
                          const std::vector<Detection>& dets)
{
    static const std::vector<cv::Scalar> PALETTE = {
        {255,56,56},{255,157,151},{255,112,31},{255,178,29},{207,210,49},
        {72,249,10},{146,204,23},{61,219,134},{26,147,52},{0,212,187},
        {44,153,168},{0,194,255},{52,69,147},{100,115,255},{0,24,236},
        {132,56,255},{82,0,133},{203,56,255},{255,149,200},{255,55,199}
    };
    cv::Mat vis = image.clone();
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        cv::Scalar  c = PALETTE[i % PALETTE.size()];

        if (!d.mask.empty()) {
            cv::Mat colour(vis.size(), CV_32FC3, c);
            cv::Mat mask3; cv::cvtColor(d.mask, mask3, cv::COLOR_GRAY2BGR);
            cv::Mat nm; mask3.convertTo(nm, CV_32FC3, 1.f / 255.f);
            cv::Mat vis_f; vis.convertTo(vis_f, CV_32FC3);
            cv::addWeighted(vis_f, 1.0, colour.mul(nm), 0.4, 0.0, vis_f);
            vis_f.convertTo(vis, CV_8UC3);
        }

        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(d.mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (!cnts.empty()) {
            cv::Rect bbox = cv::boundingRect(cnts[0]);
            for (auto& cnt : cnts) bbox |= cv::boundingRect(cnt);
            cv::rectangle(vis, bbox, c, 2);

            std::ostringstream lbl;
            lbl << d.class_name << " " << std::fixed
                << std::setprecision(2) << d.score;
            int bl = 0;
            auto ts = cv::getTextSize(lbl.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &bl);
            cv::rectangle(vis, {bbox.x, bbox.y-ts.height-6},
                          {bbox.x+ts.width+4, bbox.y}, c, -1);
            cv::putText(vis, lbl.str(), {bbox.x+2, bbox.y-4},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1);
        }
    }
    return vis;
}

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

        auto dets = yoso.infer(image);
        std::cout << "[ORT] Detections: " << dets.size() << '\n';
        for (size_t i = 0; i < dets.size(); ++i)
            std::cout << "  [" << i << "] " << dets[i].class_name
                      << "  score=" << dets[i].score << '\n';

        std::filesystem::create_directories(out_dir);
        std::string stem = std::filesystem::path(image_path).stem().string();
        std::string out_path = out_dir + "/" + stem + "_yoso_ort.jpg";
        cv::imwrite(out_path, visualise(image, dets));
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

            auto dets = yoso.infer(frame);
            last_vis = visualise(frame, dets);

            cv::putText(last_vis, "Press q/ESC to quit",
                        {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
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
