#pragma once
#include "Yolo.h"

class DMLYoloXDetector : public YoloBaseDetectorDML {
public:
    bool LetterBoxPreProcess(const unsigned char* imageData, int width, int height, LetterBoxInfo& letterbox_info) override;

protected:
    void GenerateProposals(float* output, std::vector<DMLObject>& proposals, float conf) override;
};