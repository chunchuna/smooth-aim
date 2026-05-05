#include "ScreenCapture.h"
#include <chrono>

ScreenCapture::ScreenCapture()
    : width(0)
    , height(0)
    , channels(3)
    , captureTime(0.0f)
    , initialized(false)
    , method(1) // 默认使用 DXGI
    , obsIpAddress("0. 0.0.0")
    , obsPort(7788) {
}

ScreenCapture::~ScreenCapture() {
    Release();
}

bool ScreenCapture::Init(int width, int height, int mode) {
    Release();
    method = mode;

    this->width = width;
    this->height = height;

    if (method == 0) {
        gdiCapture = std::make_unique<GDIScreenCapture>();
        if (gdiCapture->Initialize(width, height)) {
            initialized = true;
            return true;
        }
        gdiCapture.reset();
    }
    else if (method == 1) {
        dxCapture = std::make_unique<DXScreenCapture>();
        if (dxCapture->Initialize(width, height)) {
            initialized = true;
            return true;
        }
        dxCapture.reset();
    }
    else if (method == 2) {
        wgcCapture = std::make_unique<WGCScreenCapture>();
        if (wgcCapture->Initialize(width, height)) {
            initialized = true;
            return true;
        }
        wgcCapture.reset();
    }
    else if (method == 3) {
        // OBS 模式
        obsCapture = std::make_unique<Obs>();
        if (obsCapture->Init(obsIpAddress, obsPort)) {
            initialized = true;
            return true;
        }
        obsCapture.reset();
    }

    return false;
}

bool ScreenCapture::SetObsNetwork(const std::string& ipAddress, int port) {
    obsIpAddress = ipAddress;
    obsPort = port;

    // 如果已经初始化为OBS模式，重新初始化
    if (initialized && method == 3) {
        return Init(width, height, method);
    }

    return true;
}

bool ScreenCapture::SetWindow(HWND hwnd) {
    if (!initialized) return false;

    switch (method) {
    case 0: // GDI
        return gdiCapture ? gdiCapture->SetWindow(hwnd) : false;
    case 1: // DXGI (不支持窗口捕获)
        return false;
    case 2: // WGC
        return wgcCapture ? wgcCapture->SetWindow(hwnd) : false;
    case 3: // OBS (不支持窗口捕获)
        return false;
    default:
        return false;
    }
}

bool ScreenCapture::SetRegion(int x, int y, int width, int height) {
    if (!initialized) return false;

    bool success = false;

    switch (method) {
    case 0: // GDI
        success = gdiCapture ? gdiCapture->SetRegion(x, y, width, height) : false;
        break;
    case 1: // DXGI
        success = dxCapture ? dxCapture->SetRegion(x, y, width, height) : false;
        break;
    case 2: // WGC
        success = wgcCapture ? wgcCapture->SetRegion(x, y, width, height) : false;
        break;
    case 3: // OBS (不支持区域设置)
        return false;
    default:
        return false;
    }

    if (success) {
        this->width = width;
        this->height = height;
    }

    return success;
}

void ScreenCapture::Reset() {
    Release();
    initialized = false;
    width = 0;
    height = 0;
    captureTime = 0.0f;
}

std::vector<uint8_t> ScreenCapture::Capture() {
    if (!initialized) return std::vector<uint8_t>();

    std::vector<uint8_t> result;

    switch (method) {
    case 0: // GDI
        result = gdiCapture ? gdiCapture->CaptureBGR() : std::vector<uint8_t>();
        break;
    case 1: // DXGI
        result = dxCapture ? dxCapture->CaptureBGR() : std::vector<uint8_t>();
        break;
    case 2: // WGC
        result = wgcCapture ? wgcCapture->CaptureBGR() : std::vector<uint8_t>();
        break;
    case 3: // OBS
        if (obsCapture) {
            cv::Mat frame;
            if (obsCapture->Capture(frame)) {
                if (!frame.empty()) {
                    // 更新宽高
                    this->width = frame.cols;
                    this->height = frame.rows;

                    // 转换 cv::Mat 到 std::vector<uint8_t>
                    size_t dataSize = frame.total() * frame.elemSize();
                    result.resize(dataSize);
                    std::memcpy(result.data(), frame.data, dataSize);
                }
            }
        }
        break;
    default:
        break;
    }

    return result;
}

bool ScreenCapture::Capture(uint8_t* buffer, size_t bufferSize) {
    if (!initialized || !buffer) return false;

    switch (method) {
    case 0:
        return gdiCapture ? gdiCapture->CaptureBGR(buffer, bufferSize) : false;
    case 1:
        return dxCapture ? dxCapture->CaptureBGR(buffer, bufferSize) : false;
    case 2:
        return wgcCapture ? wgcCapture->CaptureBGR(buffer, bufferSize) : false;
    case 3:
        if (obsCapture) {
            cv::Mat frame;
            if (obsCapture->Capture(frame) && !frame.empty()) {
                this->width = frame.cols;
                this->height = frame.rows;
                size_t dataSize = frame.total() * frame.elemSize();
                if (bufferSize < dataSize) return false;
                std::memcpy(buffer, frame.data, dataSize);
                return true;
            }
        }
        return false;
    default:
        return false;
    }
}

void ScreenCapture::Release() {
    if (gdiCapture) {
        gdiCapture->Release();
        gdiCapture.reset();
    }

    if (dxCapture) {
        dxCapture->Release();
        dxCapture.reset();
    }

    if (wgcCapture) {
        wgcCapture->Release();
        wgcCapture.reset();
    }

    if (obsCapture) {
        obsCapture->Release();
        obsCapture.reset();
    }
}
