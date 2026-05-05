#include "YoloV8.h"
#include <immintrin.h>
#include <cfloat>

// V8 raw output [1, 4+nc, num_anchors], anchors are contiguous within each channel.
// Three paths: AVX2 (8/iter) -> SSE4.1 (4/iter) -> Scalar.
void DMLYoloV8Detector::GenerateProposals(float* output, std::vector<DMLObject>& proposals, float conf) {
    const size_t num_features = m_output_dims[1];
    const size_t num_detections = m_output_dims[2];
    const int numClasses = static_cast<int>(num_features) - 4;

    if (num_detections == 0 || numClasses <= 0) return;

    const float* ptr = output;

    // ===== AVX2 path =====
    if (SIMDSupport::avx2Supported()) {
        const __m256 conf_v  = _mm256_set1_ps(conf);
        const __m256 half_v  = _mm256_set1_ps(0.5f);
        const __m256 neg_inf = _mm256_set1_ps(-FLT_MAX);
        const size_t avx_end = num_detections & ~size_t(7);
        size_t d = 0;
        alignas(32) float mx[8], cx[8], cy[8], wv[8], hv[8];
        alignas(32) int   cls[8];
        for (; d < avx_end; d += 8) {
            __m256 cx_v = _mm256_loadu_ps(ptr + 0 * num_detections + d);
            __m256 cy_v = _mm256_loadu_ps(ptr + 1 * num_detections + d);
            __m256 w_v  = _mm256_loadu_ps(ptr + 2 * num_detections + d);
            __m256 h_v  = _mm256_loadu_ps(ptr + 3 * num_detections + d);
            __m256 max_score = neg_inf;
            __m256i max_cls  = _mm256_setzero_si256();
            for (int c = 0; c < numClasses; ++c) {
                __m256 s = _mm256_loadu_ps(ptr + (4 + c) * num_detections + d);
                __m256 m = _mm256_cmp_ps(s, max_score, _CMP_GT_OQ);
                max_score = _mm256_blendv_ps(max_score, s, m);
                __m256i c_v = _mm256_set1_epi32(c);
                max_cls = _mm256_castps_si256(_mm256_blendv_ps(
                    _mm256_castsi256_ps(max_cls), _mm256_castsi256_ps(c_v), m));
            }
            __m256 pass = _mm256_cmp_ps(max_score, conf_v, _CMP_GT_OQ);
            int mask = _mm256_movemask_ps(pass);
            if (mask == 0) continue;
            __m256 left_v = _mm256_sub_ps(cx_v, _mm256_mul_ps(w_v, half_v));
            __m256 top_v  = _mm256_sub_ps(cy_v, _mm256_mul_ps(h_v, half_v));
            _mm256_store_ps(mx, max_score);
            _mm256_store_ps(cx, left_v);
            _mm256_store_ps(cy, top_v);
            _mm256_store_ps(wv, w_v);
            _mm256_store_ps(hv, h_v);
            _mm256_store_si256((__m256i*)cls, max_cls);
            for (int k = 0; k < 8; ++k) {
                if (mask & (1 << k)) {
                    DMLObject obj;
                    obj.label = cls[k]; obj.prob = mx[k];
                    obj.x = cx[k]; obj.y = cy[k];
                    obj.width = wv[k]; obj.height = hv[k];
                    proposals.emplace_back(obj);
                }
            }
        }
        for (; d < num_detections; ++d) {
            float maxScore = -FLT_MAX; int classId = 0;
            for (int c = 0; c < numClasses; ++c) {
                float s = ptr[(4 + c) * num_detections + d];
                if (s > maxScore) { maxScore = s; classId = c; }
            }
            if (maxScore > conf) {
                float w = ptr[2 * num_detections + d];
                float h = ptr[3 * num_detections + d];
                DMLObject obj;
                obj.label = classId; obj.prob = maxScore;
                obj.x = ptr[0 * num_detections + d] - w * 0.5f;
                obj.y = ptr[1 * num_detections + d] - h * 0.5f;
                obj.width = w; obj.height = h;
                proposals.emplace_back(obj);
            }
        }
        return;
    }

    // ===== SSE4.1 path =====
    if (SIMDSupport::sseSupported()) {
        const __m128 conf_v  = _mm_set1_ps(conf);
        const __m128 half_v  = _mm_set1_ps(0.5f);
        const __m128 neg_inf = _mm_set1_ps(-FLT_MAX);
        const size_t sse_end = num_detections & ~size_t(3);
        size_t d = 0;
        alignas(16) float mx[4], cx[4], cy[4], wv[4], hv[4];
        alignas(16) int   cls[4];
        for (; d < sse_end; d += 4) {
            __m128 cx_v = _mm_loadu_ps(ptr + 0 * num_detections + d);
            __m128 cy_v = _mm_loadu_ps(ptr + 1 * num_detections + d);
            __m128 w_v  = _mm_loadu_ps(ptr + 2 * num_detections + d);
            __m128 h_v  = _mm_loadu_ps(ptr + 3 * num_detections + d);
            __m128 max_score = neg_inf;
            __m128i max_cls  = _mm_setzero_si128();
            for (int c = 0; c < numClasses; ++c) {
                __m128 s = _mm_loadu_ps(ptr + (4 + c) * num_detections + d);
                __m128 m = _mm_cmpgt_ps(s, max_score);
                max_score = _mm_blendv_ps(max_score, s, m);
                __m128i c_v = _mm_set1_epi32(c);
                max_cls = _mm_castps_si128(_mm_blendv_ps(
                    _mm_castsi128_ps(max_cls), _mm_castsi128_ps(c_v), m));
            }
            __m128 pass = _mm_cmpgt_ps(max_score, conf_v);
            int mask = _mm_movemask_ps(pass);
            if (mask == 0) continue;
            __m128 left_v = _mm_sub_ps(cx_v, _mm_mul_ps(w_v, half_v));
            __m128 top_v  = _mm_sub_ps(cy_v, _mm_mul_ps(h_v, half_v));
            _mm_store_ps(mx, max_score);
            _mm_store_ps(cx, left_v);
            _mm_store_ps(cy, top_v);
            _mm_store_ps(wv, w_v);
            _mm_store_ps(hv, h_v);
            _mm_store_si128((__m128i*)cls, max_cls);
            for (int k = 0; k < 4; ++k) {
                if (mask & (1 << k)) {
                    DMLObject obj;
                    obj.label = cls[k]; obj.prob = mx[k];
                    obj.x = cx[k]; obj.y = cy[k];
                    obj.width = wv[k]; obj.height = hv[k];
                    proposals.emplace_back(obj);
                }
            }
        }
        for (; d < num_detections; ++d) {
            float maxScore = -FLT_MAX; int classId = 0;
            for (int c = 0; c < numClasses; ++c) {
                float s = ptr[(4 + c) * num_detections + d];
                if (s > maxScore) { maxScore = s; classId = c; }
            }
            if (maxScore > conf) {
                float w = ptr[2 * num_detections + d];
                float h = ptr[3 * num_detections + d];
                DMLObject obj;
                obj.label = classId; obj.prob = maxScore;
                obj.x = ptr[0 * num_detections + d] - w * 0.5f;
                obj.y = ptr[1 * num_detections + d] - h * 0.5f;
                obj.width = w; obj.height = h;
                proposals.emplace_back(obj);
            }
        }
        return;
    }

    // ===== Scalar fallback =====
    for (size_t d = 0; d < num_detections; ++d) {
        float maxScore = -FLT_MAX; int classId = 0;
        for (int c = 0; c < numClasses; ++c) {
            float s = ptr[(4 + c) * num_detections + d];
            if (s > maxScore) { maxScore = s; classId = c; }
        }
        if (maxScore > conf) {
            float w = ptr[2 * num_detections + d];
            float h = ptr[3 * num_detections + d];
            DMLObject obj;
            obj.label = classId; obj.prob = maxScore;
            obj.x = ptr[0 * num_detections + d] - w * 0.5f;
            obj.y = ptr[1 * num_detections + d] - h * 0.5f;
            obj.width = w; obj.height = h;
            proposals.emplace_back(obj);
        }
    }
}
