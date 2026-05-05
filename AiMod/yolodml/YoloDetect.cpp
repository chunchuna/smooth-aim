#include "YoloDetect.h"

YoloDetect::YoloDetect()
    : mode(0) {
}

YoloDetect::~YoloDetect() {
    Release();
}

bool YoloDetect::Init(const std::string& modelPath, int device, int mode) {
    this->mode = mode;

    switch (mode) {
    case 0:
        detector = std::make_unique<DMLYoloXDetector>();
        break;
    case 1:
        detector = std::make_unique<DMLYoloV5Detector>();
        break;
    case 2:
        detector = std::make_unique<DMLYoloV7Detector>();
        break;
    case 3:
        detector = std::make_unique<DMLYoloV8Detector>();
        break;
    case 4:
        detector = std::make_unique<DMLYoloV10Detector>();
        break;
    case 5:
        detector = std::make_unique<DMLYoloV11Detector>();
        break;
    case 6:
        detector = std::make_unique<DMLYoloV12Detector>();
        break;
    case 7:
        detector = std::make_unique<DMLYoloV26Detector>();
        break;
    default:
        return false;
    }

    if (detector->InitializePath(modelPath.c_str(), device, 0)) {
        return true;
    }

    detector.reset();
    return false;
}

bool YoloDetect::Init(const unsigned char* modelData, size_t modelLen, int device, int mode) {
    Release();

    this->mode = mode;

    switch (mode) {
    case 0:
        detector = std::make_unique<DMLYoloXDetector>();
        break;
    case 1:
        detector = std::make_unique<DMLYoloV5Detector>();
        break;
    case 2:
        detector = std::make_unique<DMLYoloV7Detector>();
        break;
    case 3:
        detector = std::make_unique<DMLYoloV8Detector>();
        break;
    case 4:
        detector = std::make_unique<DMLYoloV10Detector>();
        break;
    case 5:
        detector = std::make_unique<DMLYoloV11Detector>();
        break;
    case 6:
        detector = std::make_unique<DMLYoloV12Detector>();
        break;
    case 7:
        detector = std::make_unique<DMLYoloV26Detector>();
        break;
    default:
        return false;
    }

    if (detector->InitializeData(modelData, modelLen, device, 0)) {
        return true;
    }

    detector.reset();
    return false;
}

std::vector<DetectionObject> YoloDetect::Detect(const unsigned char* imageData, int width, int height, float confThreshold, float nmsThreshold) {
    return detector->DetectBGR(imageData, width, height, confThreshold, nmsThreshold);
}

void YoloDetect::Detect(const unsigned char* imageData, int width, int height, float confThreshold, float nmsThreshold, std::vector<DetectionObject>& output) {
    detector->DetectBGR(imageData, width, height, confThreshold, nmsThreshold, output);
}

void YoloDetect::Reset() {
    Release();
}

void YoloDetect::Release() {
    if (detector) {
        detector->ReleaseResources();
        detector.reset();
    }
}
