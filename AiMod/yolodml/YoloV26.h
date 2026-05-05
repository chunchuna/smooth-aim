#pragma once
#include "Yolo.h"

// YOLOv26: end2end NMS-free，输出 [batch, num_det, 6] = [x1,y1,x2,y2,score,classId]
// 与 YOLOv10 输出格式一致，后处理逻辑相同。
class DMLYoloV26Detector : public YoloBaseDetectorDML {
public:

protected:
    void GenerateProposals(float* output, std::vector<DMLObject>& proposals, float conf) override;
};
