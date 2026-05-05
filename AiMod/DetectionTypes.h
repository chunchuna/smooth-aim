#pragma once

// ============================================================
// 检测目标结构 —— YOLO / KalmanTracker / DLL 通用
// ============================================================

struct DetectionObject {
    struct { float x, y, width, height; } bbox;
    int   label;
    float prob;
    int   track_id;
};
