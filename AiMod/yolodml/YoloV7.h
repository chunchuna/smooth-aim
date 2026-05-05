#pragma once
#include "Yolo.h"

class DMLYoloV7Detector : public YoloBaseDetectorDML {
public:

protected:
    void GenerateProposals(float* output, std::vector<DMLObject>& proposals, float conf) override;
};