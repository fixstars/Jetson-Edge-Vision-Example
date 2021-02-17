#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <NvInferRuntime.h>

#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define ENABLE_V4L2_CAMERA 1

using namespace std;

class Logger : public nvinfer1::ILogger {
 public:
     void log(Severity severity, const char* msg) override
     {
         cerr << msg << endl;
     }
};

vector<uint8_t> load(const string& filename) {
    vector<uint8_t> buffer;
    ifstream ifs(filename, ios::binary);
    if (!ifs.is_open()) {
        throw runtime_error("Failed to load " + filename);
    }
    auto begin = ifs.tellg();
    ifs.seekg(0, ios::end);
    auto end = ifs.tellg();
    ifs.seekg(0, ios::beg);
    buffer.resize(end-begin);
    ifs.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    return buffer;
}

typedef struct DetectionBox {
    int class_id;
    float confidence;
    float x1, x2, y1, y2;
} DetectionBox;

float area(const DetectionBox &b) {
    return (b.x2 - b.x1) * (b.y2 - b.y1);
}

float intersection(const DetectionBox &a, const DetectionBox &b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float w = x2 - x1;
    const float h = y2 - y1;
    if (w <= 0 || h <= 0) return 0;
    return w * h;
}

float union_(const DetectionBox &a, const DetectionBox &b) {
    const auto area1 = area(a);
    const auto area2 = area(b);
    const auto inter = intersection(a, b);
    return area1 + area2 - inter;
}

vector<DetectionBox> detectnet_v2_post_processing(const float *bboxes, const float *coverages, const int num_grid_x, const int num_grid_y, const int num_classes, const int width, const int height, const float coverage_thresh = 0.4, const float nms_thresh = 0.4) {
    vector<DetectionBox> all_boxes;

    float bbox_norm = 35.0;
    int gx = width / num_grid_x;
    int gy = height / num_grid_y;
    float cx[num_grid_x];
    float cy[num_grid_y];
    for (int i=0;i <num_grid_x; ++i) {
        cx[i] = static_cast<float>(i * gx + 0.5) / bbox_norm;
    }
    for (int i=0;i <num_grid_y; ++i) {
        cy[i] = static_cast<float>(i * gy + 0.5) / bbox_norm;
    }

    for (int c=0; c<num_classes; ++c) {
        for (int y=0; y<num_grid_y; ++y) {
            for (int x=0; x<num_grid_x; ++x) {
                int c_offset = num_grid_x * num_grid_y * c;
                int b_offset = num_grid_x * num_grid_y * c * 4;
                float coverage = coverages[c_offset + num_grid_x * y + x];
                if (coverage > coverage_thresh) {
                    int x1 = (bboxes[b_offset + num_grid_x * num_grid_y * 0 + num_grid_x * y + x] - cx[x]) * (-bbox_norm);
                    int y1 = (bboxes[b_offset + num_grid_x * num_grid_y * 1 + num_grid_x * y + x] - cy[y]) * (-bbox_norm);
                    int x2 = (bboxes[b_offset + num_grid_x * num_grid_y * 2 + num_grid_x * y + x] + cx[x]) * (+bbox_norm);
                    int y2 = (bboxes[b_offset + num_grid_x * num_grid_y * 3 + num_grid_x * y + x] + cy[y]) * (+bbox_norm);

                    x1 = min(max(x1, 0), width-1);
                    y1 = min(max(y1, 0), height-1);
                    x2 = min(max(x2, 0), width-1);
                    y2 = min(max(y2, 0), height-1);

                    // Prevent underflows
                    if ((x2 - x1 < 0) || (y2 - y1) < 0) {
                        continue;
                    }

                    DetectionBox b;
                    b.confidence = coverage;
                    b.class_id = c;
                    b.x1 = x1;
                    b.y1 = y1;
                    b.x2 = x2;
                    b.y2 = y2;
                    all_boxes.push_back(b);
                }
            }
        }
    }

    vector<bool> is_valid(all_boxes.size(), true);

    sort(all_boxes.begin(), all_boxes.end(), [](const DetectionBox& x, const DetectionBox& y){ return x.confidence < y.confidence; });

    for (int i = 0; i < all_boxes.size(); i++) {
        if (!is_valid[i]) continue;
        const auto main = all_boxes[i];
        for (int j = i + 1; j < all_boxes.size(); j++) {
            if (!is_valid[j]) continue;
            const auto other = all_boxes[j];
            const auto iou = intersection(main, other) / union_(main, other);
            is_valid[j] = iou <= nms_thresh;
        }
    }

    vector<DetectionBox> detected_boxes;
    for (int i = 0; i < all_boxes.size(); i++) {
        if (is_valid[i]) detected_boxes.push_back(all_boxes[i]);
    }

    return detected_boxes;
}

int main() {

    try {

#if ENABLE_V4L2_CAMERA
        const int width = 1280;
        const int height = 720;

        cv::VideoCapture cap(cv::CAP_V4L2);
        if (!cap.isOpened()) {
            throw runtime_error("Failed to find camera device");
        }

        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
#else
        const int width = 1920;
        const int height = 1080;

        cv::VideoCapture cap("sample_1080p_h265.mp4");
        if (!cap.isOpened()) {
            throw runtime_error("Failed to find camera device");
        }

#endif
        vector<uint8_t> model = load("trt.engine");

        Logger logger;
        nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
        if (runtime == nullptr) {
            throw runtime_error("Failed to create TensorRT runtime");
        }

        nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(model.data(), model.size(), nullptr);
        if (engine == nullptr) {
            throw runtime_error("Failed to create TensorRT inference engine");
        }

        nvinfer1::IExecutionContext *context = engine->createExecutionContext();
        if (context == nullptr) {
            throw runtime_error("Failed to create TensorRT inference context");
        }

        const int internal_width = 960;
        const int internal_height = 544;
        const float internal_ratio = static_cast<float>(internal_width) / static_cast<float>(internal_height);
        const float ratio = static_cast<float>(width) / static_cast<float>(height);
        const int channel_num = 3;
        const int grid_size = 16;
        const int num_classes = 3;
        const int num_grid_x = internal_width / grid_size;
        const int num_grid_y = internal_height / grid_size;

        vector<void*> bindings;
        float *input;
        if (cudaMallocManaged(&input, internal_width * internal_height * channel_num * sizeof(float))) {
            throw runtime_error("Failed to allocate I/O buffer");
        }
        bindings.push_back(input);

        float *output_bboxes;
        if (cudaMallocManaged(&output_bboxes, 4 * num_classes * num_grid_x * num_grid_y * sizeof(float))) {
            throw runtime_error("Failed to allocate I/O buffer");
        }
        bindings.push_back(output_bboxes);

        float *output_coverages;
        if (cudaMallocManaged(&output_coverages, num_classes * num_grid_x * num_grid_y * sizeof(float))) {
            throw runtime_error("Failed to allocate I/O buffer");
        }
        bindings.push_back(output_coverages);

        auto start = chrono::high_resolution_clock::now();

        const int n = 1000;

        cv::Mat frame(internal_height, internal_width, CV_8UC3);
        for (int i=0; i<n; ++i) {
            // BGR
            cap.read(frame);

            cv::Mat input_mat;
            cv::cvtColor(frame, input_mat, cv::COLOR_BGR2RGB);
            cv::normalize(input_mat, input_mat, 0, 1.0, cv::NORM_MINMAX, CV_32FC3);

            if (ratio > internal_ratio) {
                cv::resize(input_mat, input_mat, cv::Size(internal_width, internal_width / ratio));
            } else {
                cv::resize(input_mat, input_mat, cv::Size(ratio * internal_height, internal_height));
            }

            const float resize_ratio = static_cast<float>(width) / static_cast<float>(input_mat.cols);
            const int top  = std::max((internal_height - input_mat.rows) / 2, 0);
            const int bottom = internal_height - input_mat.rows - top;
            const int left = std::max((internal_width - input_mat.cols) / 2, 0);
            const int right = internal_width - input_mat.cols - left;

            cv::copyMakeBorder(input_mat, input_mat, top, bottom, left, right, cv::BORDER_CONSTANT, 0);

            input_mat = input_mat.reshape(1, internal_width*internal_height).t();

            memcpy(input, input_mat.ptr(), internal_width*internal_height*3*sizeof(float));

            const int32_t batch_size = 1;
            if (!context->execute(batch_size, bindings.data())) {
                throw runtime_error("Failed to execute TensorRT infererence");
            }

            auto boxes = detectnet_v2_post_processing(output_bboxes, output_coverages, num_grid_x, num_grid_y, num_classes, internal_width, internal_height, 0.4, 0.4);

            for (auto& b : boxes) {
                b.x1 -= left;
                b.y1 -= top;
                b.x2 -= left;
                b.y2 -= top;
                b.x1 *= resize_ratio;
                b.y1 *= resize_ratio;
                b.x2 *= resize_ratio;
                b.y2 *= resize_ratio;

                b.x1 = max(0.0f, min(static_cast<float>(width),  b.x1));
                b.y1 = max(0.0f, min(static_cast<float>(height), b.y1));
                b.x2 = max(0.0f, min(static_cast<float>(width),  b.x2));
                b.y2 = max(0.0f, min(static_cast<float>(height), b.y2));
            }

            const char* labels[] = {"Person", "Bag", "Face"};

            for (const auto& b : boxes) {
                const cv::Point2d p1(b.x1, b.y1);
                const cv::Point2d p2(b.x2, b.y2);
                const cv::Scalar color = cv::Scalar(0, 0, 255);
                cv::putText(frame, labels[b.class_id], cv::Point(b.x1, b.y1 - 3), cv::FONT_HERSHEY_COMPLEX, 0.5, color);
                cv::rectangle(frame, p1, p2, color);
            }

            cv::imshow("app", frame);
            cv::waitKey(1);
        }

        auto end = chrono::high_resolution_clock::now();

        cout << chrono::duration_cast<chrono::milliseconds>(end-start).count() << " ms" << endl;
        cout << (static_cast<float>(n) / chrono::duration_cast<chrono::milliseconds>(end-start).count()) * 1e3f << " FPS" << endl;

        cudaFree(input);
        cudaFree(output_bboxes);
        cudaFree(output_coverages);

    } catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
