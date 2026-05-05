#include "ScreenCapture.h"
#include "YoloDetect.h"
#include <memory>
#include <vector>
#include <string>
#include <atomic>
#include <algorithm>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <Windows.h>
#pragma comment(lib, "winmm.lib")
#include <timeapi.h>
#include <immintrin.h>  // _mm_pause
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "KalmanFilter.h"
#include "curve.h"
#include "pid.h"
#include "Win32Gui.h"

// ============================================================
// 模型文件扫描
// ============================================================
static std::vector<std::string> g_modelFiles;  // 模型文件路径列表(相对于exe)
static std::vector<std::string> g_modelNames;  // 模型显示名称(不含路径和后缀)

static void ScanModels(const std::string& folder = "models") {
    g_modelFiles.clear();
    g_modelNames.clear();
    namespace fs = std::filesystem;
    if (!fs::exists(folder)) {
        // 创建目录以便用户放模型
        fs::create_directories(folder);
        return;
    }
    for (auto& entry : fs::recursive_directory_iterator(folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".onnx") {
            std::string relPath = entry.path().string();
            // 统一用正斜杠
            std::replace(relPath.begin(), relPath.end(), '\\', '/');
            g_modelFiles.push_back(relPath);
            // 显示名: 只取文件名(不含目录和.onnx后缀)
            std::string name = entry.path().stem().string();
            g_modelNames.push_back(name);
        }
    }
}

// ============================================================
// 虚拟键名称映射 (用于GUI显示)
// ============================================================
static const char* VKeyNames[] = {
    "None",           // 0
    "Left Mouse",     // 1  VK_LBUTTON
    "Right Mouse",    // 2  VK_RBUTTON
    "Cancel",         // 3
    "Middle Mouse",   // 4  VK_MBUTTON
    "X1 Mouse",       // 5  VK_XBUTTON1
    "X2 Mouse",       // 6  VK_XBUTTON2
};

static const int AimKeyOptions[] = {
    VK_RBUTTON,    // 0 - 右键
    VK_LBUTTON,    // 1 - 左键
    VK_XBUTTON1,   // 2 - 鼠标侧键1
    VK_XBUTTON2,   // 3 - 鼠标侧键2
    VK_SHIFT,      // 4 - Shift
    VK_MENU,       // 5 - Alt
    VK_CAPITAL,    // 6 - CapsLock (按住)
};
static const char* AimKeyNames[] = {
    "Right Mouse",
    "Left Mouse",
    "X1 (Side)",
    "X2 (Side)",
    "Shift",
    "Alt",
    "CapsLock",
};
static const int AimKeyCount = sizeof(AimKeyOptions) / sizeof(AimKeyOptions[0]);

// YOLO模型类型名称
static const char* YoloModeNames[] = {
    "YOLOX",   // 0
    "YOLOv5",  // 1
    "YOLOv7",  // 2
    "YOLOv8",  // 3
    "YOLOv10", // 4
    "YOLOv11", // 5
    "YOLOv12", // 6
    "YOLOv26", // 7
};
static const int YoloModeCount = 8;

// 截图模式名称
static const char* CaptureModeNames[] = {
    "GDI",  // 0
    "DXGI", // 1
    "WGC",  // 2
    "OBS",  // 3
};
static const int CaptureModeCount = 4;

struct DetectionConfig {
    int captureMode = 0;            // 截图模式: 0=GDI, 1=DXGI, 2=WGC, 3=OBS
    int captureWidth = 320;         // 截图宽度
    int captureHeight = 320;        // 截图高度
    HWND targetWindow = nullptr;    // 目标窗口句柄 (GDI/WGC模式下有效)

    std::string obsIpAddress = "0.0.0.0"; // OBS网络地址 (仅captureMode=3)
    int obsPort = 7788;                    // OBS端口

    bool useRegion = false;         // 是否使用区域截图
    int regionX = 0;                // 区域起始X
    int regionY = 0;                // 区域起始Y
    int regionWidth = 0;            // 区域宽度
    int regionHeight = 0;           // 区域高度

    std::string modelPath = "yolo5n.onnx"; // ONNX模型路径
    int yoloDevice = 0;             // DirectML设备ID (0=默认GPU)
    int yoloMode = 1;               // 模型类型: 0=YOLOX, 1=v5, 2=v7, 3=v8, 4=v10, 5=v11, 6=v12, 7=v26
    float confThreshold = 0.6f;     // 置信度阈值
    float nmsThreshold = 0.15f;     // NMS阈值

    float KpX = 25;                 // X轴 PID 比例增益
    float KdX = 25;                 // X轴 PID 微分增益
    float PredictX = 0;             // X轴预测系数
    float RateX = 0.3f;             // X轴速率系数

    float KpY = 25;                 // Y轴 PID 比例增益
    float KdY = 25;                 // Y轴 PID 微分增益
    float PredictY = 0;             // Y轴预测系数
    float RateY = 0.3f;             // Y轴速率系数

    bool enableDisplay = true;      // 是否启用显示窗口

    // --- 自瞄相关 ---
    int aimKeyIndex = 0;            // 自瞄激活键索引 (AimKeyOptions[])
    bool aimEnabled = true;         // 是否启用自瞄
    float aimSmooth = 1.0f;         // 自瞄平滑度 (1.0=直接, 越大越平滑)
    float aimFovRadius = 160.0f;    // 自瞄FOV半径(像素), 超出不瞄
    float headOffset = 0.35f;       // 瞄头偏移 (0=中心, 0.5=顶部)
    int moveMode = 0;               // 鼠标移动模式: 0=曲线(MMousePredictor), 1=直接SendInput
};

// ============================================================
// 简易 JSON 配置保存/加载 (不依赖第三方库)
// ============================================================
static void SaveConfig(const DetectionConfig& cfg, const std::string& path = "aimod_config.json") {
    std::ofstream f(path);
    if (!f.is_open()) return;
    f << "{\n";
    f << "  \"captureMode\": " << cfg.captureMode << ",\n";
    f << "  \"captureWidth\": " << cfg.captureWidth << ",\n";
    f << "  \"captureHeight\": " << cfg.captureHeight << ",\n";
    f << "  \"modelPath\": \"" << cfg.modelPath << "\",\n";
    f << "  \"yoloDevice\": " << cfg.yoloDevice << ",\n";
    f << "  \"yoloMode\": " << cfg.yoloMode << ",\n";
    f << "  \"confThreshold\": " << cfg.confThreshold << ",\n";
    f << "  \"nmsThreshold\": " << cfg.nmsThreshold << ",\n";
    f << "  \"KpX\": " << cfg.KpX << ",\n";
    f << "  \"KdX\": " << cfg.KdX << ",\n";
    f << "  \"PredictX\": " << cfg.PredictX << ",\n";
    f << "  \"RateX\": " << cfg.RateX << ",\n";
    f << "  \"KpY\": " << cfg.KpY << ",\n";
    f << "  \"KdY\": " << cfg.KdY << ",\n";
    f << "  \"PredictY\": " << cfg.PredictY << ",\n";
    f << "  \"RateY\": " << cfg.RateY << ",\n";
    f << "  \"enableDisplay\": " << (cfg.enableDisplay ? "true" : "false") << ",\n";
    f << "  \"aimKeyIndex\": " << cfg.aimKeyIndex << ",\n";
    f << "  \"aimEnabled\": " << (cfg.aimEnabled ? "true" : "false") << ",\n";
    f << "  \"aimSmooth\": " << cfg.aimSmooth << ",\n";
    f << "  \"aimFovRadius\": " << cfg.aimFovRadius << ",\n";
    f << "  \"headOffset\": " << cfg.headOffset << ",\n";
    f << "  \"moveMode\": " << cfg.moveMode << "\n";
    f << "}\n";
    f.close();
}

// 极简JSON值提取 (不依赖第三方库)
static std::string ExtractJsonString(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    auto start = json.find('"', pos + 1);
    if (start == std::string::npos) return "";
    auto end = json.find('"', start + 1);
    if (end == std::string::npos) return "";
    return json.substr(start + 1, end - start - 1);
}

static double ExtractJsonNumber(const std::string& json, const std::string& key, double defaultVal) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return defaultVal;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return defaultVal;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    try { return std::stod(json.substr(pos)); }
    catch (...) { return defaultVal; }
}

static bool ExtractJsonBool(const std::string& json, const std::string& key, bool defaultVal) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return defaultVal;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return defaultVal;
    auto rest = json.substr(pos + 1);
    if (rest.find("true") < rest.find(',') || rest.find("true") < rest.find('}')) return true;
    if (rest.find("false") < rest.find(',') || rest.find("false") < rest.find('}')) return false;
    return defaultVal;
}

static DetectionConfig LoadConfig(const std::string& path = "aimod_config.json") {
    DetectionConfig cfg;
    std::ifstream f(path);
    if (!f.is_open()) return cfg;
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    cfg.captureMode = (int)ExtractJsonNumber(json, "captureMode", cfg.captureMode);
    cfg.captureWidth = (int)ExtractJsonNumber(json, "captureWidth", cfg.captureWidth);
    cfg.captureHeight = (int)ExtractJsonNumber(json, "captureHeight", cfg.captureHeight);
    auto mp = ExtractJsonString(json, "modelPath");
    if (!mp.empty()) cfg.modelPath = mp;
    cfg.yoloDevice = (int)ExtractJsonNumber(json, "yoloDevice", cfg.yoloDevice);
    cfg.yoloMode = (int)ExtractJsonNumber(json, "yoloMode", cfg.yoloMode);
    cfg.confThreshold = (float)ExtractJsonNumber(json, "confThreshold", cfg.confThreshold);
    cfg.nmsThreshold = (float)ExtractJsonNumber(json, "nmsThreshold", cfg.nmsThreshold);
    cfg.KpX = (float)ExtractJsonNumber(json, "KpX", cfg.KpX);
    cfg.KdX = (float)ExtractJsonNumber(json, "KdX", cfg.KdX);
    cfg.PredictX = (float)ExtractJsonNumber(json, "PredictX", cfg.PredictX);
    cfg.RateX = (float)ExtractJsonNumber(json, "RateX", cfg.RateX);
    cfg.KpY = (float)ExtractJsonNumber(json, "KpY", cfg.KpY);
    cfg.KdY = (float)ExtractJsonNumber(json, "KdY", cfg.KdY);
    cfg.PredictY = (float)ExtractJsonNumber(json, "PredictY", cfg.PredictY);
    cfg.RateY = (float)ExtractJsonNumber(json, "RateY", cfg.RateY);
    cfg.enableDisplay = ExtractJsonBool(json, "enableDisplay", cfg.enableDisplay);
    cfg.aimKeyIndex = (int)ExtractJsonNumber(json, "aimKeyIndex", cfg.aimKeyIndex);
    cfg.aimEnabled = ExtractJsonBool(json, "aimEnabled", cfg.aimEnabled);
    cfg.aimSmooth = (float)ExtractJsonNumber(json, "aimSmooth", cfg.aimSmooth);
    cfg.aimFovRadius = (float)ExtractJsonNumber(json, "aimFovRadius", cfg.aimFovRadius);
    cfg.headOffset = (float)ExtractJsonNumber(json, "headOffset", cfg.headOffset);
    cfg.moveMode = (int)ExtractJsonNumber(json, "moveMode", cfg.moveMode);
    return cfg;
}

// ============================================================
// SendInput 鼠标相对移动
// ============================================================
static void MoveMouse(int dx, int dy) {
    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dx = dx;
    input.mi.dy = dy;
    input.mi.dwFlags = MOUSEEVENTF_MOVE;
    SendInput(1, &input, sizeof(INPUT));
}

struct ImageData {
    std::vector<uint8_t> data;
    int width;
    int height;
    std::chrono::high_resolution_clock::time_point timestamp;
};

class DetectionSystem {
public:
    DetectionSystem() = default;
    ~DetectionSystem();

    bool Init(const DetectionConfig& config);
    bool LoadModel();  // 加载/重载模型(可在运行时调用)
    bool IsModelLoaded() const { return m_modelLoaded; }
    bool Run();
    void Stop();
    bool UpdateCaptureRegion(int x, int y, int width, int height);
    void Release();

    // 运行时参数热更新
    void UpdatePidParams();
    DetectionConfig& GetConfig() { return m_config; }
    const DetectionConfig& GetConfig() const { return m_config; }

private:
    void CaptureLoop();
    void DetectionLoop();
    void DisplayLoop();
    void GuiLoop();

private:
    ScreenCapture m_screenCapture;
    YoloDetect m_detector;
    KalmanP m_tracker;
    P_PID m_pidX;
    P_PID m_pidY;
    MMousePredictor m_mouseCurve;
    DetectionConfig m_config;

    // 推理线程复用缓冲区
    std::vector<DetectionObject> m_detectResults;
    std::vector<DetectionObject> m_trackedResults;

    // --- Cache line 隔离: 每个热 atomic 独占 cache line，消除 false sharing ---
    alignas(64) std::atomic<bool> m_running{false};
    std::thread m_captureThread;
    std::thread m_detectionThread;
    std::thread m_displayThread;
    std::thread m_guiThread;

    // 无锁帧/结果共享 — 各自独占 cache line
    alignas(64) std::atomic<std::shared_ptr<ImageData>> m_latestFrame;
    alignas(64) std::atomic<std::shared_ptr<std::vector<DetectionObject>>> m_latestResults;

    alignas(64) std::atomic<double> m_captureFPS{0.0};
    alignas(64) std::atomic<double> m_detectionFPS{0.0};
    alignas(64) std::atomic<double> m_avgDetectionTime{0.0};
    alignas(64) std::atomic<int> m_totalCaptureFrames{0};
    alignas(64) std::atomic<int> m_totalDetectionFrames{0};
    int m_noTargetFrames = 0;  // 连续无目标帧数

    // PID参数脏标志 (GUI线程设置, 推理线程消费)
    alignas(64) std::atomic<bool> m_pidDirty{false};
    // 模型重载标志 (GUI线程设置, 推理线程消费)
    alignas(64) std::atomic<bool> m_modelDirty{false};
    // 模型是否已加载
    alignas(64) std::atomic<bool> m_modelLoaded{false};

    // Win32 原生GUI面板
    Win32GuiPanel m_guiPanel;
};

DetectionSystem::~DetectionSystem() {
    Release();
}

bool DetectionSystem::Init(const DetectionConfig& config) {
    m_config = config;

    if (config.captureMode == 3) {
        if (!m_screenCapture.SetObsNetwork(config.obsIpAddress, config.obsPort)) {
            return false;
        }
    }

    if (!m_screenCapture.Init(config.captureWidth, config.captureHeight, config.captureMode)) {
        return false;
    }

    if (config.targetWindow != nullptr) {
        m_screenCapture.SetWindow(config.targetWindow);
    }

    if (config.useRegion) {
        m_screenCapture.SetRegion(config.regionX, config.regionY,
            config.regionWidth, config.regionHeight);
    }

    // 模型加载延迟到用户选择后
    if (!config.modelPath.empty() && std::filesystem::exists(config.modelPath)) {
        LoadModel();
    }

    m_tracker.init(2, 5);

    m_pidX.init(config.KpX, config.KdX, config.PredictX, config.RateX, 9900);
    m_pidY.init(config.KpY, config.KdY, config.PredictY, config.RateY, 9900);

    m_mouseCurve.init(config.captureWidth, config.captureHeight, 4, 4, 25);

    // 把 Windows 定时器精度压到 0.5ms（NtSetTimerResolution，单位 100ns）
    typedef LONG(NTAPI* PFN_NtSetTimerResolution)(ULONG, BOOLEAN, PULONG);
    static PFN_NtSetTimerResolution pNtSetTimerResolution =
        (PFN_NtSetTimerResolution)GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtSetTimerResolution");
    ULONG actualRes = 0;
    if (pNtSetTimerResolution) {
        pNtSetTimerResolution(5000, TRUE, &actualRes);  // 0.5ms
    } else {
        timeBeginPeriod(1);
    }

    return true;
}

bool DetectionSystem::LoadModel() {
    m_detector.Reset();
    m_modelLoaded = false;
    if (m_config.modelPath.empty() || !std::filesystem::exists(m_config.modelPath)) {
        std::cerr << "[AiMod] Model file not found: " << m_config.modelPath << std::endl;
        return false;
    }
    if (m_detector.Init(m_config.modelPath, m_config.yoloDevice, m_config.yoloMode)) {
        m_modelLoaded = true;
        std::cout << "[AiMod] Model loaded: " << m_config.modelPath << std::endl;
        return true;
    }
    std::cerr << "[AiMod] Failed to load model: " << m_config.modelPath << std::endl;
    return false;
}

void DetectionSystem::UpdatePidParams() {
    m_pidX.updateParams(m_config.KpX, m_config.KdX, m_config.PredictX, m_config.RateX, 9900);
    m_pidY.updateParams(m_config.KpY, m_config.KdY, m_config.PredictY, m_config.RateY, 9900);
}

bool DetectionSystem::Run() {
    if (m_running) {
        return false;
    }

    m_running = true;

    // 提升进程优先级
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // 启动截图线程
    m_captureThread = std::thread(&DetectionSystem::CaptureLoop, this);
    SetThreadPriority(m_captureThread.native_handle(), THREAD_PRIORITY_HIGHEST);

    // 启动推理线程
    m_detectionThread = std::thread(&DetectionSystem::DetectionLoop, this);
    SetThreadPriority(m_detectionThread.native_handle(), THREAD_PRIORITY_HIGHEST);

    // 启动显示线程
    if (m_config.enableDisplay) {
        m_displayThread = std::thread(&DetectionSystem::DisplayLoop, this);
        SetThreadPriority(m_displayThread.native_handle(), THREAD_PRIORITY_NORMAL);
    }

    // 启动GUI控制面板线程
    m_guiThread = std::thread(&DetectionSystem::GuiLoop, this);

    // 主线程等待
    while (m_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    return true;
}

void DetectionSystem::Stop() {
    if (!m_running) return;
    m_running = false;
    if (m_captureThread.joinable())   m_captureThread.join();
    if (m_detectionThread.joinable()) m_detectionThread.join();
    if (m_displayThread.joinable())   m_displayThread.join();
    if (m_guiThread.joinable())       m_guiThread.join();
}

void DetectionSystem::CaptureLoop() {
    // TIME_CRITICAL 优先级，减少 OS 唤醒抖动；DXGI 阻塞 wait 期间不耗 CPU
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    int frameCount = 0;
    auto lastTime = std::chrono::high_resolution_clock::now();
    const size_t bufferSize = (size_t)m_config.captureWidth * m_config.captureHeight * 3;

    // 预分配三缓冲区，避免每帧堆分配 (triple buffer: 最多2个reader + 1个writer)
    std::shared_ptr<ImageData> frames[3];
    for (int i = 0; i < 3; i++) {
        frames[i] = std::make_shared<ImageData>();
        frames[i]->data.resize(bufferSize);
    }
    int writeIdx = 0;

    while (m_running) {
        auto& frame = frames[writeIdx];

        // 如果当前缓冲区还在被读取 (refcount > 1)，跳到下一个
        if (frame.use_count() > 1) {
            writeIdx = (writeIdx + 1) % 3;
            continue;
        }

        frame->width = m_config.captureWidth;
        frame->height = m_config.captureHeight;

        // Capture() 内部已改为阻塞等待 DXGI 新帧（INFINITE，完全跟随游戏 Present）；
        // 返回 false = 资源丢失等异常 → 跳过，下一轮重试
        if (!m_screenCapture.Capture(frame->data.data(), bufferSize)) {
            continue;
        }

        frame->width = m_screenCapture.GetWidth();
        frame->height = m_screenCapture.GetHeight();
        frame->timestamp = std::chrono::high_resolution_clock::now();

        m_latestFrame.store(frame, std::memory_order_release);
        writeIdx = (writeIdx + 1) % 3;

        frameCount++;
        m_totalCaptureFrames++;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastTime).count();
        if (elapsed >= 1.0) {
            m_captureFPS = frameCount / elapsed;
            frameCount = 0;
            lastTime = now;
        }

    }
}

void DetectionSystem::DetectionLoop() {
    int frameCount = 0;
    double totalDetectionTime = 0.0;
    auto lastTime = std::chrono::high_resolution_clock::now();

    std::shared_ptr<ImageData> prevFrame;
    std::chrono::high_resolution_clock::time_point lastMoveTime{};

    while (m_running) {
        // 检查是否需要热重载模型
        if (m_modelDirty.exchange(false, std::memory_order_acquire)) {
            LoadModel();
            m_pidX.reset();
            m_pidY.reset();
        }

        // 模型未加载时等待
        if (!m_modelLoaded.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        auto frame = m_latestFrame.load(std::memory_order_acquire);
        if (!frame || frame == prevFrame) {
            // 自旋等新帧（不 sleep，避免 OS 调度抖动）
            _mm_pause();
            continue;
        }
        prevFrame = frame;


        const uint8_t* imgPtr = frame->data.data();
        int width = frame->width;
        int height = frame->height;
        if (!imgPtr || frame->data.empty()) continue;

        auto detectStart = std::chrono::high_resolution_clock::now();
        m_detectResults.clear();
        m_detector.Detect(
            imgPtr, width, height,
            m_config.confThreshold, m_config.nmsThreshold,
            m_detectResults
        );
        auto detectEnd = std::chrono::high_resolution_clock::now();
        double detectionTime = std::chrono::duration<double, std::milli>(detectEnd - detectStart).count();

        m_tracker.predict(m_detectResults, m_trackedResults);

        auto newResults = std::make_shared<std::vector<DetectionObject>>(std::move(m_trackedResults));
        m_latestResults.store(newResults, std::memory_order_release);

        // 使用卡尔曼追踪后的目标值
        auto trackedResults = newResults;
        bool hasTargets = trackedResults && !trackedResults->empty();

        // 检查PID参数是否需要热更新
        if (m_pidDirty.exchange(false, std::memory_order_acquire)) {
            UpdatePidParams();
        }

        if (!hasTargets) {
            // 短暂丢失目标不重置，宽容5帧
            m_noTargetFrames++;
            if (m_noTargetFrames > 5) {
                m_pidX.reset();
                m_pidY.reset();
            }
        } else {
            m_noTargetFrames = 0;

            float centerX = width / 2.0f;
            float centerY = height / 2.0f;
            float minDist = FLT_MAX;
            const DetectionObject* closest = nullptr;
            for (const auto& obj : *trackedResults) {
                float tcx = obj.bbox.x + obj.bbox.width / 2.0f;
                // 应用头部偏移: headOffset=0 瞄中心, 0.5 瞄顶部
                float tcy = obj.bbox.y + obj.bbox.height * (0.5f - m_config.headOffset);
                float dx = tcx - centerX;
                float dy = tcy - centerY;
                float dist = dx * dx + dy * dy;
                if (dist < minDist) {
                    minDist = dist;
                    closest = &obj;
                }
            }

            if (closest) {
                double tcx = closest->bbox.x + closest->bbox.width / 2.0;
                double tcy = closest->bbox.y + closest->bbox.height * (0.5 - m_config.headOffset);
                double dx = tcx - (width / 2.0);
                double dy = tcy - (height / 2.0);

                // FOV检查
                double dist = std::sqrt(dx * dx + dy * dy);
                bool inFov = dist <= m_config.aimFovRadius;

                // 检查自瞄激活键是否按下
                int aimVKey = AimKeyOptions[m_config.aimKeyIndex % AimKeyCount];
                bool aimKeyDown = (GetAsyncKeyState(aimVKey) & 0x8000) != 0;

                if (m_config.aimEnabled && aimKeyDown && inFov) {
                    double outX = m_pidX.update(dx);
                    double outY = m_pidY.update(dy);

                    if (m_config.moveMode == 0) {
                        // 曲线模式: 使用 MMousePredictor 生成人性化轨迹
                        auto path = m_mouseCurve.moveTo(outX, outY);
                        // 发送轨迹上的每一个点
                        float smooth = std::max(m_config.aimSmooth, 1.0f);
                        for (const auto& pt : path) {
                            int mx = (int)std::round(pt.first / smooth);
                            int my = (int)std::round(pt.second / smooth);
                            if (mx != 0 || my != 0) {
                                MoveMouse(mx, my);
                            }
                        }
                    } else {
                        // 直接模式: 一次SendInput移动
                        float smooth = std::max(m_config.aimSmooth, 1.0f);
                        int mx = (int)std::round(outX / smooth);
                        int my = (int)std::round(outY / smooth);
                        if (mx != 0 || my != 0) {
                            MoveMouse(mx, my);
                        }
                    }
                } else {
                    // 未按键时重置PID, 下次按键时从新鲜状态开始
                    m_pidX.reset();
                    m_pidY.reset();
                }
            } else {
                m_pidX.reset();
                m_pidY.reset();
            }
        }

        frameCount++;
        m_totalDetectionFrames++;
        totalDetectionTime += detectionTime;

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastTime).count();
        if (elapsed >= 1.0) {
            m_detectionFPS = frameCount / elapsed;
            m_avgDetectionTime = (frameCount > 0) ? totalDetectionTime / frameCount : 0.0;
            frameCount = 0;
            totalDetectionTime = 0.0;
            lastTime = now;
        }
    }
}

void DetectionSystem::DisplayLoop() {
    const std::string windowName = "Detection Result";
    std::shared_ptr<ImageData> prevFrame;
    int displayFrameCount = 0;
    double displayTotalFPS = 0.0;
    cv::Mat display;

    while (m_running) {
        auto frameStart = std::chrono::high_resolution_clock::now();

        auto frame = m_latestFrame.load(std::memory_order_acquire);
        if (!frame || frame == prevFrame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        prevFrame = frame;

        auto localResults = m_latestResults.load(std::memory_order_acquire);

        int width = frame->width;
        int height = frame->height;
        if (frame->data.empty()) continue;

        cv::Mat frameMat(height, width, CV_8UC3, const_cast<uint8_t*>(frame->data.data()));
        if (display.rows != height || display.cols != width) {
            display.create(height, width, CV_8UC3);
        }
        frameMat.copyTo(display);

        if (localResults && !localResults->empty()) {
            for (const auto& obj : *localResults) {
                cv::rectangle(display,
                    cv::Point(static_cast<int>(obj.bbox.x), static_cast<int>(obj.bbox.y)),
                    cv::Point(static_cast<int>(obj.bbox.x + obj.bbox.width),
                        static_cast<int>(obj.bbox.y + obj.bbox.height)),
                    cv::Scalar(0, 255, 0), 2);

                std::ostringstream label;
                label << "ID:" << obj.label << " " << std::fixed << std::setprecision(2) << obj.prob;

                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                int top = std::max(static_cast<int>(obj.bbox.y), labelSize.height);

                cv::rectangle(display,
                    cv::Point(static_cast<int>(obj.bbox.x), top - labelSize.height),
                    cv::Point(static_cast<int>(obj.bbox.x) + labelSize.width, top + baseLine),
                    cv::Scalar(0, 255, 0), cv::FILLED);

                cv::putText(display, label.str(),
                    cv::Point(static_cast<int>(obj.bbox.x), top),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }
        }

        // FPS信息
        auto frameEnd = std::chrono::high_resolution_clock::now();
        double frameDuration = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
        double currentFPS = (frameDuration > 0) ? 1000.0 / frameDuration : 0.0;

        displayFrameCount++;
        displayTotalFPS += currentFPS;
        double avgDisplayFPS = displayTotalFPS / displayFrameCount;

        std::ostringstream fpsText;
        fpsText << "Capture FPS: " << std::fixed << std::setprecision(1) << m_captureFPS.load();
        cv::putText(display, fpsText.str(), cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        std::ostringstream detFpsText;
        detFpsText << "Detection FPS: " << std::fixed << std::setprecision(1) << m_detectionFPS.load();
        cv::putText(display, detFpsText.str(), cv::Point(10, 60),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        std::ostringstream detTimeText;
        detTimeText << "Detection Time: " << std::fixed << std::setprecision(1) << m_avgDetectionTime.load() << " ms";
        cv::putText(display, detTimeText.str(), cv::Point(10, 90),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        std::ostringstream dispFpsText;
        dispFpsText << "Display FPS: " << std::fixed << std::setprecision(1) << avgDisplayFPS;
        cv::putText(display, dispFpsText.str(), cv::Point(10, 120),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        // 目标数量
        if (localResults) {
            std::ostringstream objText;
            objText << "Objects: " << localResults->size();
            cv::putText(display, objText.str(), cv::Point(10, 150),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        }

        cv::imshow(windowName, display);

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            m_running = false;
            break;
        }
    }

    cv::destroyWindow(windowName);
}

bool DetectionSystem::UpdateCaptureRegion(int x, int y, int width, int height) {
    return m_screenCapture.SetRegion(x, y, width, height);
}

// ============================================================
// GUI 控制面板 (OpenCV trackbar)
// ============================================================
void DetectionSystem::GuiLoop() {
    const std::string winName = "AiMod Control Panel";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);

    // --- Trackbar 整数代理变量 (trackbar只支持int) ---
    // PID X
    int tb_KpX = (int)(m_config.KpX * 10);       // 0.1精度, 范围0-1000 => 0.0-100.0
    int tb_KdX = (int)(m_config.KdX * 10);
    int tb_PredictX = (int)(m_config.PredictX * 100); // 0.01精度
    int tb_RateX = (int)(m_config.RateX * 100);
    // PID Y
    int tb_KpY = (int)(m_config.KpY * 10);
    int tb_KdY = (int)(m_config.KdY * 10);
    int tb_PredictY = (int)(m_config.PredictY * 100);
    int tb_RateY = (int)(m_config.RateY * 100);
    // 检测
    int tb_conf = (int)(m_config.confThreshold * 100);  // 0-100
    int tb_nms = (int)(m_config.nmsThreshold * 100);
    // 模型文件选择 (从g_modelFiles列表中选择)
    int tb_modelIdx = 0;
    // 尝试找到当前config中的模型在列表中的位置
    for (int i = 0; i < (int)g_modelFiles.size(); i++) {
        if (g_modelFiles[i] == m_config.modelPath) { tb_modelIdx = i; break; }
    }
    int tb_yoloMode = m_config.yoloMode;
    // 自瞄
    int tb_aimEnabled = m_config.aimEnabled ? 1 : 0;
    int tb_aimKeyIdx = m_config.aimKeyIndex;
    int tb_aimSmooth = (int)(m_config.aimSmooth * 10);  // 0.1精度, 10-100 => 1.0-10.0
    int tb_aimFov = (int)(m_config.aimFovRadius);
    int tb_headOffset = (int)(m_config.headOffset * 100); // 0.01精度, 0-50
    int tb_moveMode = m_config.moveMode;
    int tb_display = m_config.enableDisplay ? 1 : 0;

    // 创建trackbar
    int maxModelIdx = std::max(0, (int)g_modelFiles.size() - 1);
    cv::createTrackbar("Model File", winName, &tb_modelIdx, maxModelIdx);
    cv::createTrackbar("YOLO Type", winName, &tb_yoloMode, YoloModeCount - 1);
    cv::createTrackbar("Aim ON/OFF", winName, &tb_aimEnabled, 1);
    cv::createTrackbar("Aim Key", winName, &tb_aimKeyIdx, AimKeyCount - 1);
    cv::createTrackbar("Move Mode", winName, &tb_moveMode, 1);
    cv::createTrackbar("Smooth x10", winName, &tb_aimSmooth, 100);
    cv::createTrackbar("FOV Radius", winName, &tb_aimFov, 500);
    cv::createTrackbar("HeadOff x100", winName, &tb_headOffset, 50);
    cv::createTrackbar("KpX x10", winName, &tb_KpX, 1000);
    cv::createTrackbar("KdX x10", winName, &tb_KdX, 1000);
    cv::createTrackbar("PredX x100", winName, &tb_PredictX, 500);
    cv::createTrackbar("RateX x100", winName, &tb_RateX, 100);
    cv::createTrackbar("KpY x10", winName, &tb_KpY, 1000);
    cv::createTrackbar("KdY x10", winName, &tb_KdY, 1000);
    cv::createTrackbar("PredY x100", winName, &tb_PredictY, 500);
    cv::createTrackbar("RateY x100", winName, &tb_RateY, 100);
    cv::createTrackbar("Conf x100", winName, &tb_conf, 100);
    cv::createTrackbar("NMS x100", winName, &tb_nms, 100);
    cv::createTrackbar("Preview", winName, &tb_display, 1);

    while (m_running) {
        // 从trackbar读取并更新config
        bool pidChanged = false;

        float newKpX = tb_KpX / 10.0f;
        float newKdX = tb_KdX / 10.0f;
        float newPredictX = tb_PredictX / 100.0f;
        float newRateX = tb_RateX / 100.0f;
        float newKpY = tb_KpY / 10.0f;
        float newKdY = tb_KdY / 10.0f;
        float newPredictY = tb_PredictY / 100.0f;
        float newRateY = tb_RateY / 100.0f;

        if (newKpX != m_config.KpX || newKdX != m_config.KdX ||
            newPredictX != m_config.PredictX || newRateX != m_config.RateX ||
            newKpY != m_config.KpY || newKdY != m_config.KdY ||
            newPredictY != m_config.PredictY || newRateY != m_config.RateY) {
            pidChanged = true;
        }

        m_config.KpX = newKpX;
        m_config.KdX = newKdX;
        m_config.PredictX = newPredictX;
        m_config.RateX = newRateX;
        m_config.KpY = newKpY;
        m_config.KdY = newKdY;
        m_config.PredictY = newPredictY;
        m_config.RateY = newRateY;

        m_config.confThreshold = tb_conf / 100.0f;
        m_config.nmsThreshold = tb_nms / 100.0f;
        m_config.aimEnabled = (tb_aimEnabled != 0);
        m_config.aimKeyIndex = tb_aimKeyIdx;
        m_config.aimSmooth = std::max(tb_aimSmooth / 10.0f, 0.1f);
        m_config.aimFovRadius = (float)tb_aimFov;
        m_config.headOffset = tb_headOffset / 100.0f;
        m_config.moveMode = tb_moveMode;
        m_config.enableDisplay = (tb_display != 0);

        // 模型类型变更
        m_config.yoloMode = tb_yoloMode;

        // 模型文件选择更新
        if (!g_modelFiles.empty() && tb_modelIdx >= 0 && tb_modelIdx < (int)g_modelFiles.size()) {
            m_config.modelPath = g_modelFiles[tb_modelIdx];
        }

        if (pidChanged) {
            m_pidDirty.store(true, std::memory_order_release);
        }

        // 绘制状态信息面板
        cv::Mat panel(480, 500, CV_8UC3, cv::Scalar(30, 30, 30));
        int y = 25;
        auto putLine = [&](const std::string& text, cv::Scalar color = cv::Scalar(200, 200, 200)) {
            cv::putText(panel, text, cv::Point(15, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            y += 22;
        };

        putLine("=== AiMod Control Panel ===", cv::Scalar(0, 255, 255));

        // 模型状态
        std::string modelStatus = m_modelLoaded.load() ? "[LOADED]" : "[NOT LOADED]";
        cv::Scalar modelColor = m_modelLoaded.load() ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        putLine("Status: " + modelStatus, modelColor);

        // 当前选中的模型文件名
        std::string selectedModel = "(none)";
        if (!g_modelFiles.empty() && tb_modelIdx >= 0 && tb_modelIdx < (int)g_modelNames.size()) {
            selectedModel = g_modelNames[tb_modelIdx];
        }
        putLine("Model[" + std::to_string(tb_modelIdx) + "]: " + selectedModel, cv::Scalar(255, 200, 100));
        putLine("Type: " + std::string(YoloModeNames[m_config.yoloMode % YoloModeCount]), cv::Scalar(255, 200, 100));
        putLine("");
        putLine("Aim: " + std::string(m_config.aimEnabled ? "ON" : "OFF"),
                m_config.aimEnabled ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255));
        putLine("Aim Key: " + std::string(AimKeyNames[m_config.aimKeyIndex % AimKeyCount]),
                cv::Scalar(200, 200, 255));
        putLine("Move: " + std::string(m_config.moveMode == 0 ? "Curve" : "Direct"));
        putLine("Smooth: " + std::to_string(m_config.aimSmooth).substr(0, 4));
        putLine("FOV: " + std::to_string((int)m_config.aimFovRadius) + " px");
        putLine("HeadOffset: " + std::to_string(m_config.headOffset).substr(0, 4));
        putLine("");
        putLine("--- PID X ---", cv::Scalar(100, 255, 100));
        putLine("Kp=" + std::to_string(m_config.KpX).substr(0, 5) +
               " Kd=" + std::to_string(m_config.KdX).substr(0, 5));
        putLine("Predict=" + std::to_string(m_config.PredictX).substr(0, 5) +
               " Rate=" + std::to_string(m_config.RateX).substr(0, 4));
        putLine("--- PID Y ---", cv::Scalar(100, 255, 100));
        putLine("Kp=" + std::to_string(m_config.KpY).substr(0, 5) +
               " Kd=" + std::to_string(m_config.KdY).substr(0, 5));
        putLine("Predict=" + std::to_string(m_config.PredictY).substr(0, 5) +
               " Rate=" + std::to_string(m_config.RateY).substr(0, 4));
        putLine("");
        putLine("Conf: " + std::to_string(m_config.confThreshold).substr(0, 4) +
               "  NMS: " + std::to_string(m_config.nmsThreshold).substr(0, 4));
        putLine("Capture FPS: " + std::to_string(m_captureFPS.load()).substr(0, 5));
        putLine("Detection FPS: " + std::to_string(m_detectionFPS.load()).substr(0, 5));
        putLine("Det Time: " + std::to_string(m_avgDetectionTime.load()).substr(0, 5) + " ms");
        putLine("");
        putLine("[Ctrl+R] Load Model  [Ctrl+S] Save  [Ctrl+L] Load  [ESC] Quit", cv::Scalar(0, 200, 255));

        cv::imshow(winName, panel);

        int key = cv::waitKey(50);
        // ESC 只在OpenCV窗口有焦点时生效，防止游戏内按ESC误退出
        if (key == 27) {
            m_running = false;
            break;
        }

        // Ctrl+R/S/L 全局热键，不需要窗口焦点，不会跟游戏按键冲突
        static bool prevR = false, prevS = false, prevL = false;
        bool ctrlDown = (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0;
        bool curR = ctrlDown && (GetAsyncKeyState('R') & 0x8000) != 0;
        bool curS = ctrlDown && (GetAsyncKeyState('S') & 0x8000) != 0;
        bool curL = ctrlDown && (GetAsyncKeyState('L') & 0x8000) != 0;

        if (curS && !prevS) {
            SaveConfig(m_config);
            std::cout << "[AiMod] Config saved to aimod_config.json" << std::endl;
        } else if (curR && !prevR) {
            // 请求推理线程重载模型
            m_modelDirty.store(true, std::memory_order_release);
            std::cout << "[AiMod] Model load requested: " << m_config.modelPath << std::endl;
        } else if (curL && !prevL) {
            m_config = LoadConfig();
            // 同步trackbar
            tb_KpX = (int)(m_config.KpX * 10);
            tb_KdX = (int)(m_config.KdX * 10);
            tb_PredictX = (int)(m_config.PredictX * 100);
            tb_RateX = (int)(m_config.RateX * 100);
            tb_KpY = (int)(m_config.KpY * 10);
            tb_KdY = (int)(m_config.KdY * 10);
            tb_PredictY = (int)(m_config.PredictY * 100);
            tb_RateY = (int)(m_config.RateY * 100);
            tb_conf = (int)(m_config.confThreshold * 100);
            tb_nms = (int)(m_config.nmsThreshold * 100);
            tb_aimEnabled = m_config.aimEnabled ? 1 : 0;
            tb_aimKeyIdx = m_config.aimKeyIndex;
            tb_aimSmooth = (int)(m_config.aimSmooth * 10);
            tb_aimFov = (int)(m_config.aimFovRadius);
            tb_headOffset = (int)(m_config.headOffset * 100);
            tb_moveMode = m_config.moveMode;
            tb_display = m_config.enableDisplay ? 1 : 0;
            tb_yoloMode = m_config.yoloMode;
            // 找到模型在列表中的位置
            for (int i = 0; i < (int)g_modelFiles.size(); i++) {
                if (g_modelFiles[i] == m_config.modelPath) { tb_modelIdx = i; break; }
            }

            cv::setTrackbarPos("Model File", winName, tb_modelIdx);
            cv::setTrackbarPos("YOLO Type", winName, tb_yoloMode);
            cv::setTrackbarPos("KpX x10", winName, tb_KpX);
            cv::setTrackbarPos("KdX x10", winName, tb_KdX);
            cv::setTrackbarPos("PredX x100", winName, tb_PredictX);
            cv::setTrackbarPos("RateX x100", winName, tb_RateX);
            cv::setTrackbarPos("KpY x10", winName, tb_KpY);
            cv::setTrackbarPos("KdY x10", winName, tb_KdY);
            cv::setTrackbarPos("PredY x100", winName, tb_PredictY);
            cv::setTrackbarPos("RateY x100", winName, tb_RateY);
            cv::setTrackbarPos("Conf x100", winName, tb_conf);
            cv::setTrackbarPos("NMS x100", winName, tb_nms);
            cv::setTrackbarPos("Aim ON/OFF", winName, tb_aimEnabled);
            cv::setTrackbarPos("Aim Key", winName, tb_aimKeyIdx);
            cv::setTrackbarPos("Smooth x10", winName, tb_aimSmooth);
            cv::setTrackbarPos("FOV Radius", winName, tb_aimFov);
            cv::setTrackbarPos("HeadOff x100", winName, tb_headOffset);
            cv::setTrackbarPos("Move Mode", winName, tb_moveMode);
            cv::setTrackbarPos("Preview", winName, tb_display);

            m_pidDirty.store(true, std::memory_order_release);
            std::cout << "[AiMod] Config loaded from aimod_config.json" << std::endl;
        }

        prevR = curR; prevS = curS; prevL = curL;
    }

    cv::destroyWindow(winName);
}

void DetectionSystem::Release() {
    Stop();
    timeEndPeriod(1);
    m_detector.Reset();
    m_screenCapture.Reset();
    m_latestFrame.store(nullptr, std::memory_order_release);
    m_latestResults.store(nullptr, std::memory_order_release);
    m_totalCaptureFrames = 0;
    m_totalDetectionFrames = 0;
    m_captureFPS = 0.0;
    m_detectionFPS = 0.0;
    m_avgDetectionTime = 0.0;
}

int main() {
    // 设置工作目录为exe所在目录
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);
    std::string exeDir(exePath);
    exeDir = exeDir.substr(0, exeDir.find_last_of("\\/"));
    SetCurrentDirectoryA(exeDir.c_str());

    std::cout << "========================================" << std::endl;
    std::cout << "  AiMod - AI Aim Assist System" << std::endl;
    std::cout << "========================================" << std::endl;

    // 扫描模型文件夹
    ScanModels("models");
    std::cout << "Found " << g_modelFiles.size() << " models in ./models/" << std::endl;

    // 尝试从文件加载配置, 如果不存在则使用默认值
    DetectionConfig config = LoadConfig();
    std::cout << "Capture: " << CaptureModeNames[config.captureMode % CaptureModeCount]
              << " " << config.captureWidth << "x" << config.captureHeight << std::endl;
    std::cout << "Aim Key: " << AimKeyNames[config.aimKeyIndex % AimKeyCount] << std::endl;

    if (g_modelFiles.empty()) {
        std::cout << "[Warning] No .onnx models found in ./models/ folder!" << std::endl;
        std::cout << "Please put .onnx model files in: " << exeDir << "\\models\\" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "GUI Controls:" << std::endl;
    std::cout << "  [R] Load/Reload Model  [S] Save config  [L] Load config  [ESC] Quit" << std::endl;
    std::cout << "  Use trackbars to adjust parameters in real-time" << std::endl;
    std::cout << "  Select model in GUI then press [R] to load it" << std::endl;
    std::cout << "========================================" << std::endl;

    DetectionSystem detectionSystem;

    if (!detectionSystem.Init(config)) {
        std::cerr << "截图系统初始化失败!" << std::endl;
        system("pause");
        return -1;
    }

    detectionSystem.Run();

    // 退出时自动保存配置
    SaveConfig(detectionSystem.GetConfig());
    std::cout << "[AiMod] Config auto-saved on exit." << std::endl;

    return 0;
}
