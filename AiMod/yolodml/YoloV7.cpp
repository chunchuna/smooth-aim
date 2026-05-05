#include "YoloV7.h"
#include <immintrin.h>
#include <cfloat>

// V7 raw output [1, num_anchors, 5+nc], each anchor's fields are contiguous.
// Inner argmax over nc classes uses SIMD (AVX2 8/iter -> SSE4.1 4/iter -> Scalar).
void DMLYoloV7Detector::GenerateProposals(float* output, std::vector<DMLObject>& proposals, float conf_threshold) {
    proposals.clear();

    const int num_classes = static_cast<int>(m_output_dims[2]) - 5;
    const int num_proposals = static_cast<int>(m_output_dims[1]);
    const bool use_avx2 = SIMDSupport::avx2Supported();
    const bool use_sse  = SIMDSupport::sseSupported();

    for (int i = 0; i < num_proposals; i++) {
        float* current_pred = output + i * (num_classes + 5);

        float objectness = current_pred[4];
        if (objectness < conf_threshold) continue;

        float x_center = current_pred[0];
        float y_center = current_pred[1];
        float width    = current_pred[2];
        float height   = current_pred[3];
        if (width <= 0 || height <= 0) continue;

        float* class_scores = current_pred + 5;
        int   best_class_id  = 0;
        float max_class_score = -FLT_MAX;

        if (use_avx2 && num_classes >= 8) {
            __m256 max_v = _mm256_set1_ps(-FLT_MAX);
            __m256i idx_v = _mm256_setzero_si256();
            const __m256i base = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            int c = 0;
            for (; c + 8 <= num_classes; c += 8) {
                __m256 s = _mm256_loadu_ps(class_scores + c);
                __m256 m = _mm256_cmp_ps(s, max_v, _CMP_GT_OQ);
                max_v = _mm256_blendv_ps(max_v, s, m);
                __m256i cur_idx = _mm256_add_epi32(_mm256_set1_epi32(c), base);
                idx_v = _mm256_castps_si256(_mm256_blendv_ps(
                    _mm256_castsi256_ps(idx_v), _mm256_castsi256_ps(cur_idx), m));
            }
            alignas(32) float mx[8]; alignas(32) int ix[8];
            _mm256_store_ps(mx, max_v);
            _mm256_store_si256((__m256i*)ix, idx_v);
            for (int k = 0; k < 8; ++k) {
                if (mx[k] > max_class_score) { max_class_score = mx[k]; best_class_id = ix[k]; }
            }
            for (; c < num_classes; ++c) {
                if (class_scores[c] > max_class_score) { max_class_score = class_scores[c]; best_class_id = c; }
            }
        }
        else if (use_sse && num_classes >= 4) {
            __m128 max_v = _mm_set1_ps(-FLT_MAX);
            __m128i idx_v = _mm_setzero_si128();
            const __m128i base = _mm_setr_epi32(0, 1, 2, 3);
            int c = 0;
            for (; c + 4 <= num_classes; c += 4) {
                __m128 s = _mm_loadu_ps(class_scores + c);
                __m128 m = _mm_cmpgt_ps(s, max_v);
                max_v = _mm_blendv_ps(max_v, s, m);
                __m128i cur_idx = _mm_add_epi32(_mm_set1_epi32(c), base);
                idx_v = _mm_castps_si128(_mm_blendv_ps(
                    _mm_castsi128_ps(idx_v), _mm_castsi128_ps(cur_idx), m));
            }
            alignas(16) float mx[4]; alignas(16) int ix[4];
            _mm_store_ps(mx, max_v);
            _mm_store_si128((__m128i*)ix, idx_v);
            for (int k = 0; k < 4; ++k) {
                if (mx[k] > max_class_score) { max_class_score = mx[k]; best_class_id = ix[k]; }
            }
            for (; c < num_classes; ++c) {
                if (class_scores[c] > max_class_score) { max_class_score = class_scores[c]; best_class_id = c; }
            }
        }
        else {
            max_class_score = class_scores[0];
            best_class_id = 0;
            for (int c = 1; c < num_classes; ++c) {
                if (class_scores[c] > max_class_score) { max_class_score = class_scores[c]; best_class_id = c; }
            }
        }

        float final_confidence = objectness * max_class_score;
        if (final_confidence < conf_threshold) continue;

        DMLObject obj;
        obj.x      = x_center - width * 0.5f;
        obj.y      = y_center - height * 0.5f;
        obj.width  = width;
        obj.height = height;
        obj.label  = best_class_id;
        obj.prob   = final_confidence;
        proposals.push_back(obj);
    }
}
