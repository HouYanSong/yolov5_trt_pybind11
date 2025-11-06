#include "NvInfer.h"
#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "utils/preprocess.h"
#include "utils/postprocess.h"
#include "utils/types.h"
#include "utils/utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <mutex>

namespace py = pybind11;

// 将numpy数组转换为cv::Mat
cv::Mat numpy_to_mat(py::array_t<unsigned char>& input) {
    py::buffer_info buf_info = input.request();
    
    if (buf_info.ndim == 3) {
        // 彩色图像
        int height = buf_info.shape[0];
        int width = buf_info.shape[1];
        int channels = buf_info.shape[2];
        
        cv::Mat mat(height, width, CV_8UC3, (unsigned char*)buf_info.ptr);
        return mat.clone();
    } else if (buf_info.ndim == 2) {
        // 灰度图像
        int height = buf_info.shape[0];
        int width = buf_info.shape[1];
        
        cv::Mat mat(height, width, CV_8UC1, (unsigned char*)buf_info.ptr);
        return mat.clone();
    }
    
    throw std::runtime_error("Unsupported array dimensions");
}

// 将cv::Mat转换为numpy数组
py::array_t<unsigned char> mat_to_numpy(cv::Mat& mat) {
    if (mat.empty()) {
        return py::array_t<unsigned char>();
    }
    
    if (mat.channels() == 1) {
        // 灰度图像
        auto result = py::array_t<unsigned char>({mat.rows, mat.cols});
        auto buf = result.request();
        memcpy(buf.ptr, mat.data, sizeof(unsigned char) * mat.total());
        return result;
    } else {
        // 彩色图像
        auto result = py::array_t<unsigned char>({mat.rows, mat.cols, mat.channels()});
        auto buf = result.request();
        memcpy(buf.ptr, mat.data, sizeof(unsigned char) * mat.total() * mat.channels());
        return result;
    }
}

// 加载模型文件
std::vector<unsigned char> load_engine_file(const std::string &file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);
    assert(engine_file.is_open() && "Unable to load engine file.");
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
    return engine_data;
}

// YOLOv5推理器类
class YOLOv5Detector {
private:
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::unique_ptr<samplesCommon::BufferManager> buffers;
    bool initialized = false;
    int img_size;

public:
    YOLOv5Detector(const std::string& engine_file) {
        initialize(engine_file);
    }
    
    void initialize(const std::string& engine_file) {
        // ========== 1. 创建推理运行时runtime ==========
        runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
        if (!runtime) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        // ========== 2. 反序列化生成engine ==========
        auto plan = load_engine_file(engine_file);
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine) {
            throw std::runtime_error("Failed to deserialize engine");
        }

        // ========== 3. 创建执行上下文context ==========
        context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("Failed to create execution context");
        }

        // ========== 4. 创建输入输出缓冲区 ==========
        buffers = std::make_unique<samplesCommon::BufferManager>(engine);
        
        initialized = true;
    }
    
    py::list detect(py::array_t<unsigned char>& input_image, int input_w=kInputW, int input_h=kInputH, float conf_thresh=kConfThresh, float nms_thresh=kNmsThresh) {
        if (!initialized) {
            throw std::runtime_error("Detector not initialized");
        }
        
        // 将numpy数组转换为cv::Mat
        cv::Mat frame = numpy_to_mat(input_image);
        
        if (frame.empty()) {
            throw std::runtime_error("Invalid input image");
        }
        
        int width = frame.cols;
        int height = frame.rows;
        img_size = width * height;
        cuda_preprocess_init(img_size);
        
        // CUDA预处理
        process_input_gpu(frame, (float *)buffers->getDeviceBuffer(kInputTensorName), input_w, input_h);
        
        // ========== 5. 执行推理 ==========
        context->executeV2(buffers->getDeviceBindings().data());
        
        // 拷贝回host
        buffers->copyOutputToHost();
        
        // 从buffer manager中获取模型输出
        int32_t *num_det = (int32_t *)buffers->getHostBuffer(kOutNumDet);
        int32_t *cls = (int32_t *)buffers->getHostBuffer(kOutDetCls);
        float *conf = (float *)buffers->getHostBuffer(kOutDetScores);
        float *bbox = (float *)buffers->getHostBuffer(kOutDetBBoxes);
        
        // 执行nms（非极大值抑制）
        std::vector<Detection> bboxs;
        yolo_nms(bboxs, num_det, cls, conf, bbox, conf_thresh, nms_thresh);
        
        // 返回检测结果
        py::list result_list;
        for (size_t j = 0; j < bboxs.size(); j++) {
            cv::Rect r = get_rect(frame, bboxs[j].bbox);
            py::dict detection;
            detection["class_id"] = (int)bboxs[j].class_id;
            detection["confidence"] = (float)bboxs[j].conf;
            detection["bbox"] = py::cast(std::vector<int>{r.x, r.y, r.x + r.width, r.y + r.height});
            result_list.append(detection);
        }
        
        return result_list;
    }
};

// Python绑定代码
PYBIND11_MODULE(yolov5_trt, m) {
    m.doc() = "YOLOv5 TensorRT Python bindings";
    
    py::class_<YOLOv5Detector>(m, "YOLOv5Detector")
        .def(py::init<const std::string&>(), "Initialize detector with engine file")
        .def("detect", &YOLOv5Detector::detect, "Perform detection on input image",
            py::arg("input_image"), 
            py::arg("input_w") = kInputW, 
            py::arg("input_h") = kInputH,
            py::arg("conf_thresh") = kConfThresh, 
            py::arg("nms_thresh") = kNmsThresh); 
}