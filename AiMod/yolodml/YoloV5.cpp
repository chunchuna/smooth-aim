#include "YoloV5.h"


void DMLYoloV5Detector::GenerateProposals(float* output, std::vector<DMLObject>& proposals, float conf) {

    // 遍历所有预测框
    for (int box_idx = 0; box_idx < m_output_dims[1]; ++box_idx) {
        const int base_pos = box_idx * m_output_dims[2];
        float objectness = output[base_pos + 4];

        // 初步筛选：objectness 需大于阈值
        if (objectness > conf) {
            // 解析框的几何参数
            float x_center = output[base_pos + 0];
            float y_center = output[base_pos + 1];
            float width = output[base_pos + 2];
            float height = output[base_pos + 3];

            // 遍历类别得分（从第5个元素开始）
            for (int cls_idx = 5; cls_idx < m_output_dims[2]; ++cls_idx) {
                float cls_prob = output[base_pos + cls_idx];
                float score = objectness * cls_prob;  // 综合置信度

                // 最终筛选：综合得分需大于阈值
                if (score > conf) {
                    DMLObject obj;
                    obj.x = x_center - width * 0.5f;    // 左上角x
                    obj.y = y_center - height * 0.5f;   // 左上角y
                    obj.width = width;
                    obj.height = height;
                    obj.label = cls_idx - 5;            // 类别索引
                    obj.prob = score;                   // 综合置信度
                    proposals.emplace_back(obj);
                }
            }
        }
    }
}

