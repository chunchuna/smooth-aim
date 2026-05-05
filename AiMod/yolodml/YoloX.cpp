#define NOMINMAX
#include "YoloX.h"


// 支持AVX2/SSE/Scalar三种模式
bool DMLYoloXDetector::LetterBoxPreProcess(const unsigned char* imageData, int width, int height, LetterBoxInfo& letterbox_info) {
    if (!imageData || width <= 0 || height <= 0 || m_input_dim <= 0 || !m_blob) {
        return false;
    }

    m_original_width = width;
    m_original_height = height;

    // 计算缩放比例和padding
    float scale = std::min(static_cast<float>(m_input_dim) / width,
        static_cast<float>(m_input_dim) / height);

    int new_unpad_w = static_cast<int>(std::round(width * scale));
    int new_unpad_h = static_cast<int>(std::round(height * scale));

    float dw = m_input_dim - new_unpad_w;
    float dh = m_input_dim - new_unpad_h;
    dw /= 2.0f;
    dh /= 2.0f;

    int pad_left = static_cast<int>(std::round(dw - 0.1f));
    int pad_top = static_cast<int>(std::round(dh - 0.1f));

    letterbox_info.scale = scale;
    letterbox_info.pad_w = pad_left;
    letterbox_info.pad_h = pad_top;
    letterbox_info.new_unpad_w = new_unpad_w;
    letterbox_info.new_unpad_h = new_unpad_h;

    // 初始化为灰色背景
    const size_t total_pixels = m_input_dim * m_input_dim;
    float* r_ptr = m_blob;
    float* g_ptr = r_ptr + total_pixels;
    float* b_ptr = g_ptr + total_pixels;

    // 根据SIMD支持情况选择初始化方式
    // YOLOX: 不归一化，保持0-255范围，灰色值为114
    if (SIMDSupport::avx2Supported) {
        const __m256 gray_vec = _mm256_set1_ps(114.0f);  // YOLOX: 不归一化
        const size_t avx_size = total_pixels & ~7;

        for (size_t i = 0; i < avx_size; i += 8) {
            _mm256_storeu_ps(&r_ptr[i], gray_vec);
            _mm256_storeu_ps(&g_ptr[i], gray_vec);
            _mm256_storeu_ps(&b_ptr[i], gray_vec);
        }

        for (size_t i = avx_size; i < total_pixels; ++i) {
            r_ptr[i] = g_ptr[i] = b_ptr[i] = 114.0f;
        }
    }
    else if (SIMDSupport::sseSupported) {
        const __m128 gray_vec = _mm_set1_ps(114.0f);  // YOLOX: 不归一化
        const size_t sse_size = total_pixels & ~3;

        for (size_t i = 0; i < sse_size; i += 4) {
            _mm_storeu_ps(&r_ptr[i], gray_vec);
            _mm_storeu_ps(&g_ptr[i], gray_vec);
            _mm_storeu_ps(&b_ptr[i], gray_vec);
        }

        for (size_t i = sse_size; i < total_pixels; ++i) {
            r_ptr[i] = g_ptr[i] = b_ptr[i] = 114.0f;
        }
    }
    else {
        for (size_t i = 0; i < total_pixels; ++i) {
            r_ptr[i] = g_ptr[i] = b_ptr[i] = 114.0f;
        }
    }

    // 预计算坐标映射 - 最近邻
    const float x_ratio = static_cast<float>(width) / new_unpad_w;
    const float y_ratio = static_cast<float>(height) / new_unpad_h;

    std::vector<int> y_src_indices(new_unpad_h);
    for (int dst_y = 0; dst_y < new_unpad_h; ++dst_y) {
        float src_y = (dst_y + 0.5f) * y_ratio - 0.5f;
        int nearest_y = static_cast<int>(src_y + 0.5f);
        y_src_indices[dst_y] = std::max(0, std::min(nearest_y, height - 1));
    }

    std::vector<int> x_src_indices(new_unpad_w);
    for (int dst_x = 0; dst_x < new_unpad_w; ++dst_x) {
        float src_x = (dst_x + 0.5f) * x_ratio - 0.5f;
        int nearest_x = static_cast<int>(src_x + 0.5f);
        x_src_indices[dst_x] = std::max(0, std::min(nearest_x, width - 1));
    }

    if (SIMDSupport::avx2Supported) {
        // AVX2路径 - YOLOX版本（不归一化）
        for (int dst_y = 0; dst_y < new_unpad_h; ++dst_y) {
            const int src_y = y_src_indices[dst_y];
            const unsigned char* src_row = imageData + src_y * width * 3;
            const int dst_row_start = (dst_y + pad_top) * m_input_dim + pad_left;

            int dst_x = 0;
            const int avx_width = new_unpad_w & ~7;

            // AVX2处理 - 一次处理8个像素
            for (; dst_x < avx_width; dst_x += 8) {
                __m256 b_vals = _mm256_setr_ps(
                    src_row[x_src_indices[dst_x + 0] * 3 + 0],
                    src_row[x_src_indices[dst_x + 1] * 3 + 0],
                    src_row[x_src_indices[dst_x + 2] * 3 + 0],
                    src_row[x_src_indices[dst_x + 3] * 3 + 0],
                    src_row[x_src_indices[dst_x + 4] * 3 + 0],
                    src_row[x_src_indices[dst_x + 5] * 3 + 0],
                    src_row[x_src_indices[dst_x + 6] * 3 + 0],
                    src_row[x_src_indices[dst_x + 7] * 3 + 0]
                );

                __m256 g_vals = _mm256_setr_ps(
                    src_row[x_src_indices[dst_x + 0] * 3 + 1],
                    src_row[x_src_indices[dst_x + 1] * 3 + 1],
                    src_row[x_src_indices[dst_x + 2] * 3 + 1],
                    src_row[x_src_indices[dst_x + 3] * 3 + 1],
                    src_row[x_src_indices[dst_x + 4] * 3 + 1],
                    src_row[x_src_indices[dst_x + 5] * 3 + 1],
                    src_row[x_src_indices[dst_x + 6] * 3 + 1],
                    src_row[x_src_indices[dst_x + 7] * 3 + 1]
                );

                __m256 r_vals = _mm256_setr_ps(
                    src_row[x_src_indices[dst_x + 0] * 3 + 2],
                    src_row[x_src_indices[dst_x + 1] * 3 + 2],
                    src_row[x_src_indices[dst_x + 2] * 3 + 2],
                    src_row[x_src_indices[dst_x + 3] * 3 + 2],
                    src_row[x_src_indices[dst_x + 4] * 3 + 2],
                    src_row[x_src_indices[dst_x + 5] * 3 + 2],
                    src_row[x_src_indices[dst_x + 6] * 3 + 2],
                    src_row[x_src_indices[dst_x + 7] * 3 + 2]
                );

                // YOLOX: 不进行归一化处理，直接存储原始像素值
                // 存储 BGR -> RGB
                _mm256_storeu_ps(&r_ptr[dst_row_start + dst_x], r_vals);
                _mm256_storeu_ps(&g_ptr[dst_row_start + dst_x], g_vals);
                _mm256_storeu_ps(&b_ptr[dst_row_start + dst_x], b_vals);
            }

            // 处理剩余像素
            for (; dst_x < new_unpad_w; ++dst_x) {
                const int src_x = x_src_indices[dst_x];
                const unsigned char* src_pixel = src_row + src_x * 3;
                const int dst_idx = dst_row_start + dst_x;

                r_ptr[dst_idx] = static_cast<float>(src_pixel[2]); // BGR的R -> RGB的R (YOLOX: 不归一化)
                g_ptr[dst_idx] = static_cast<float>(src_pixel[1]); // BGR的G -> RGB的G (YOLOX: 不归一化)
                b_ptr[dst_idx] = static_cast<float>(src_pixel[0]); // BGR的B -> RGB的B (YOLOX: 不归一化)
            }
        }

        _mm256_zeroupper();
    }
    else if (SIMDSupport::sseSupported) {
        // SSE路径 - YOLOX版本（不归一化）
        for (int dst_y = 0; dst_y < new_unpad_h; ++dst_y) {
            const int src_y = y_src_indices[dst_y];
            const unsigned char* src_row = imageData + src_y * width * 3;
            const int dst_row_start = (dst_y + pad_top) * m_input_dim + pad_left;

            // 预取下一行
            if (dst_y + 1 < new_unpad_h) {
                const int next_src_y = y_src_indices[dst_y + 1];
                _mm_prefetch((const char*)(imageData + next_src_y * width * 3), _MM_HINT_T0);
            }

            int dst_x = 0;
            const int sse_width = new_unpad_w & ~3;

            // SSE处理 - 一次处理4个像素
            for (; dst_x < sse_width; dst_x += 4) {
                __m128 b_vals = _mm_setr_ps(
                    src_row[x_src_indices[dst_x + 0] * 3 + 0],
                    src_row[x_src_indices[dst_x + 1] * 3 + 0],
                    src_row[x_src_indices[dst_x + 2] * 3 + 0],
                    src_row[x_src_indices[dst_x + 3] * 3 + 0]
                );

                __m128 g_vals = _mm_setr_ps(
                    src_row[x_src_indices[dst_x + 0] * 3 + 1],
                    src_row[x_src_indices[dst_x + 1] * 3 + 1],
                    src_row[x_src_indices[dst_x + 2] * 3 + 1],
                    src_row[x_src_indices[dst_x + 3] * 3 + 1]
                );

                __m128 r_vals = _mm_setr_ps(
                    src_row[x_src_indices[dst_x + 0] * 3 + 2],
                    src_row[x_src_indices[dst_x + 1] * 3 + 2],
                    src_row[x_src_indices[dst_x + 2] * 3 + 2],
                    src_row[x_src_indices[dst_x + 3] * 3 + 2]
                );

                // YOLOX: 不进行归一化处理，直接存储原始像素值
                // 存储 BGR -> RGB
                _mm_storeu_ps(&r_ptr[dst_row_start + dst_x], r_vals);
                _mm_storeu_ps(&g_ptr[dst_row_start + dst_x], g_vals);
                _mm_storeu_ps(&b_ptr[dst_row_start + dst_x], b_vals);
            }

            // 处理剩余像素
            for (; dst_x < new_unpad_w; ++dst_x) {
                const int src_x = x_src_indices[dst_x];
                const unsigned char* src_pixel = src_row + src_x * 3;
                const int dst_idx = dst_row_start + dst_x;

                r_ptr[dst_idx] = static_cast<float>(src_pixel[2]); // BGR的R -> RGB的R (YOLOX: 不归一化)
                g_ptr[dst_idx] = static_cast<float>(src_pixel[1]); // BGR的G -> RGB的G (YOLOX: 不归一化)
                b_ptr[dst_idx] = static_cast<float>(src_pixel[0]); // BGR的B -> RGB的B (YOLOX: 不归一化)
            }
        }
    }
    else {
        // 标量路径 - YOLOX版本（不归一化）
        for (int dst_y = 0; dst_y < new_unpad_h; ++dst_y) {
            const int src_y = y_src_indices[dst_y];
            const unsigned char* src_row = imageData + src_y * width * 3;
            const int dst_row_start = (dst_y + pad_top) * m_input_dim + pad_left;

            for (int dst_x = 0; dst_x < new_unpad_w; ++dst_x) {
                const int src_x = x_src_indices[dst_x];
                const unsigned char* src_pixel = src_row + src_x * 3;
                const int dst_idx = dst_row_start + dst_x;

                r_ptr[dst_idx] = static_cast<float>(src_pixel[2]); // BGR的R -> RGB的R (YOLOX: 不归一化)
                g_ptr[dst_idx] = static_cast<float>(src_pixel[1]); // BGR的G -> RGB的G (YOLOX: 不归一化)
                b_ptr[dst_idx] = static_cast<float>(src_pixel[0]); // BGR的B -> RGB的B (YOLOX: 不归一化)
            }
        }
    }

    return true;
}


void DMLYoloXDetector::GenerateProposals(float* output, std::vector<DMLObject>& proposals, float conf) {
    GenerateGridsAndStride();

    for (int anchor_idx = 0; anchor_idx < m_grid_strides.size(); ++anchor_idx) {
        const int grid0 = m_grid_strides[anchor_idx].grid0; // H
        const int grid1 = m_grid_strides[anchor_idx].grid1; // W
        const int stride = m_grid_strides[anchor_idx].stride; // stride
        const int basic_pos = anchor_idx * m_output_dims[2];

        // boxes计算
        float x_center = (output[basic_pos + 0] + grid0) * stride;
        float y_center = (output[basic_pos + 1] + grid1) * stride;
        float w = exp(output[basic_pos + 2]) * stride;
        float h = exp(output[basic_pos + 3]) * stride;

        float box_objectness = output[basic_pos + 4];

        // num_class的置信度
        for (int class_idx = 0; class_idx < m_output_dims[2] - 5; class_idx++) {
            float box_cls_score = output[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;

            // 置信度筛选
            if (box_prob > conf) {
                DMLObject obj;
                obj.x = x_center - w * 0.5f;    // 左上角x
                obj.y = y_center - h * 0.5f;    // 左上角y
                obj.width = w;
                obj.height = h;
                obj.label = class_idx;          // 类别索引
                obj.prob = box_prob;            // 综合置信度
                proposals.emplace_back(obj);
            }
        }
    }
}
