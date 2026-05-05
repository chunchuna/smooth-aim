#pragma once
#include <windows.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <dml_provider_factory.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <memory>
#include <cmath>
#include <ctime>
#include <random>
#include <map>


// SIMD支持检测
namespace SIMDSupport {
    inline bool& avx2Supported() { static bool v = false; return v; }
    inline bool& sseSupported()  { static bool v = false; return v; }
    inline bool& initialized()   { static bool v = false; return v; }
}


// 在头文件中添加LetterBoxInfo结构体声明
struct LetterBoxInfo {
    float scale;           // 缩放比例
    float pad_w;          // 水平padding
    float pad_h;          // 垂直padding
    int new_unpad_w;      // 缩放后未填充的宽度
    int new_unpad_h;      // 缩放后未填充的高度
};

// YOLO检测对象结构
struct DMLObject {
    float x;        // 边界框x坐标
    float y;        // 边界框y坐标
    float width;    // 边界框宽度
    float height;   // 边界框高度
    int label;      // 类别标签
    float prob;     // 置信度
};

// 网格和步长结构，用于YOLO特征图处理
struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};

#include "../DetectionTypes.h"

// DML YOLO 基础检测器类 - 实现IYoloDetector接口
class YoloBaseDetectorDML {
private:
    // 添加LetterBox预处理函数声明
    virtual bool LetterBoxPreProcess(const unsigned char* imageData, int width, int height, LetterBoxInfo& letterbox_info);


    // 添加坐标还原函数声明
    void ScaleBoxes(std::vector<DMLObject>& objects, const LetterBoxInfo& letterbox_info);


public:
    YoloBaseDetectorDML();
    ~YoloBaseDetectorDML();

    bool InitializePath(const char* weightsPath, int device, int numThreads);
    bool InitializeData(const unsigned char* modelData, size_t modelLen, int device, int numThreads);

    // 推理接口实现
    std::vector<DetectionObject> Detect(const unsigned char* imageData, int width, int height, float conf, float nms);
    void Detect(const unsigned char* imageData, int width, int height, float conf, float nms, std::vector<DetectionObject>& output);

    std::vector<DetectionObject> DetectBMP(const unsigned char* bmpData, size_t bmpSize, float conf, float nms);
    std::vector<DetectionObject> DetectBGR(const unsigned char* imageData, int width, int height, float conf, float nms);
    void DetectBGR(const unsigned char* imageData, int width, int height, float conf, float nms, std::vector<DetectionObject>& output);

    // 释放接口实现
    void ReleaseResources();
    void ResetState();

    // Read class names from ONNX model metadata {id: name}
    std::map<int, std::string> ReadClassNames();
    const std::map<int, std::string>& GetClassNames() const { return m_classNames; }

protected:

    std::map<int, std::string> m_classNames; // class id -> name from model metadata
    std::string deviceCode; // 设备代码
    std::string m_resultBuffer;

    // ==== DirectML和ONNX Runtime相关 ====
    OrtEnv* m_env = nullptr;
    OrtSessionOptions* m_session_options = nullptr;
    OrtSession* m_session = nullptr;
    OrtMemoryInfo* m_memory_info = nullptr;
    OrtAllocator* m_allocator = nullptr;
    OrtValue* m_input_tensor = nullptr;
    OrtValue* m_output_tensor = nullptr;
    const OrtApi* m_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // 模型和推理相关
    const char* m_input_name[1] = { "images" };
    const char* m_output_name[1] = { "output" };
    std::vector<int64_t> m_input_dims = {};
    std::vector<int64_t> m_output_dims = {};
    size_t m_input_tensor_size = 1;
    int m_input_dim = 0;
    int m_total_img_pixels = 0;
    float* m_blob = nullptr;
    unsigned char* m_resized_data = nullptr;
    float* m_output_data = nullptr;
    float m_normalized = 1.f / 255.0f;

    // 检测结果
    std::vector<DetectionObject> m_detectionResults;
    std::vector<DMLObject> m_rawDetections;
    std::vector<GridAndStride> m_grid_strides;

    // ==== 实现方法 ====
    bool CheckStatus(OrtStatus* status, int line);
    std::wstring String2WString(const std::string& s);
    bool ParseModelInfo();
    bool ParseInput();
    bool ParseOutput();
    bool AutoDetectIONames();


    // YOLO特定处理
    void GenerateGridsAndStride();
    virtual void GenerateProposals(float* output, std::vector<DMLObject>& proposals, float conf) = 0;
    float CalculateIOU(const DMLObject& a, const DMLObject& b);
    std::vector<DMLObject> NMSBoxes(std::vector<DMLObject>& objects, float threshold);


    // 记录图像缩放和填充信息
    float m_scale = 1.0f;
    float m_scale_x = 1.0f;
    float m_scale_y = 1.0f;
    int m_padding_top = 0;
    int m_padding_left = 0;
    int m_original_width = 0;
    int m_original_height = 0;

    // LUT缓存成员变量
    std::vector<int> m_src_x_lut;
    std::vector<int> m_src_y_lut;
    int m_last_width = 0;
    int m_last_height = 0;
    int m_last_resized_width = 0;
    int m_last_resized_height = 0;

    // 优化的清零函数
    void ClearOptimized(float* b_ptr, float* g_ptr, float* r_ptr,
        int resized_width, int resized_height) {
        const int total_size = m_input_dim * m_input_dim * sizeof(float);

        // 全图清零（最快）
        memset(b_ptr, 0, total_size);
        memset(g_ptr, 0, total_size);
        memset(r_ptr, 0, total_size);
    }
};
