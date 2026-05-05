#include "YoloV10.h"


void DMLYoloV10Detector::GenerateProposals(float* output, std::vector<DMLObject>& proposals, float conf) {
    const float* ptr = output;
    int numDetections = m_output_dims[1];

    for (int i = 0; i < numDetections; i++) {
        float x1 = ptr[i * 6 + 0];
        float y1 = ptr[i * 6 + 1];
        float x2 = ptr[i * 6 + 2];
        float y2 = ptr[i * 6 + 3];
        float confidence = ptr[i * 6 + 4];
        int classId = static_cast<int>(ptr[i * 6 + 5]);

        if (confidence < conf)
            continue;

        DMLObject obj;
        obj.label = classId;
        obj.prob = confidence;
        obj.x = x1;
        obj.y = y1;
        obj.width = x2 - x1;
        obj.height = y2 - y1;

        proposals.emplace_back(obj);
    }
}