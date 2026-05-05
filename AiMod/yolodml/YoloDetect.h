#ifndef YOLODETECT_H
#define YOLODETECT_H

#include "Yolo.h"
#include "YoloV5.h"
#include "YoloV7.h"
#include "YoloV8.h"
#include "YoloV10.h"
#include "YoloV11.h"
#include "YoloV12.h"
#include "YoloV26.h"
#include "YoloX.h"
#include <vector>
#include <cstdint>
#include <memory>
#include <string>

// YOLO检测器封装类
class YoloDetect {
public:
    YoloDetect();
    ~YoloDetect();

    // 初始化函数 - 从路径加载
    bool Init(const std::string& modelPath, int device, int mode);

    // 初始化函数 - 从内存数据加载
    bool Init(const unsigned char* modelData, size_t modelLen, int device, int mode);

    // 检测函数
    std::vector<DetectionObject> Detect(const unsigned char* imageData, int width, int height, float confThreshold, float nmsThreshold);
    void Detect(const unsigned char* imageData, int width, int height, float confThreshold, float nmsThreshold, std::vector<DetectionObject>& output);

    // 重置函数
    void Reset();

    // Read class names from loaded model metadata
    std::map<int, std::string> ReadClassNames();
    const std::map<int, std::string>& GetClassNames() const;

private:
    void Release();

    std::unique_ptr<YoloBaseDetectorDML> detector;

    int mode;  // 模式参数: 1=YOLOv5, 2=YOLOv7, 3=YOLOv8, 4=YOLOv10, 5=YOLOv11, 6=YOLOv12, 0=YOLOX
};

#endif // YOLODETECT_H
