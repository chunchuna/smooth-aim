#define NOMINMAX
#include "Yolo.h"
#include <comdef.h>
#include <Wbemidl.h>
#pragma comment(lib, "wbemuuid.lib")

#include <immintrin.h>
#include <intrin.h>
#include <tlhelp32.h>
#include <thread>




void DetectCPUFeatures(bool& avx2Supported, bool& sseSupported, bool& initialized) {
    if (initialized) return;

    int cpuInfo[4] = { 0 };
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];

    if (nIds >= 1) {
        __cpuid(cpuInfo, 1);
        sseSupported = (cpuInfo[3] & (1 << 26)) != 0;  // SSE2

        if (nIds >= 7) {
            __cpuidex(cpuInfo, 7, 0);
            avx2Supported = (cpuInfo[1] & (1 << 5)) != 0;  // AVX2
        }
    }

    initialized = true;
}


// BMP结构体定义
#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};
struct BMPInfoHeader {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

bool isBMPFile(const unsigned char* data, size_t dataSize) {
    if (dataSize < sizeof(BMPFileHeader)) return false;
    const BMPFileHeader* header = reinterpret_cast<const BMPFileHeader*>(data);
    return header->bfType == 0x4D42;
}

// 优化的BMP像素数据提取 - 使用SSE加速
bool extractBMPPixelData(const unsigned char* bmpData, size_t bmpSize,
    std::vector<unsigned char>& pixelData, int& width, int& height) {
    if (bmpSize < sizeof(BMPFileHeader) + sizeof(BMPInfoHeader)) return false;

    const BMPFileHeader* fileHeader = reinterpret_cast<const BMPFileHeader*>(bmpData);
    const BMPInfoHeader* infoHeader = reinterpret_cast<const BMPInfoHeader*>(bmpData + sizeof(BMPFileHeader));

    if (fileHeader->bfType != 0x4D42 || infoHeader->biBitCount != 24) return false;

    width = infoHeader->biWidth;
    height = abs(infoHeader->biHeight);
    int rowSize = ((width * 3 + 3) / 4) * 4;
    const unsigned char* srcPixels = bmpData + fileHeader->bfOffBits;

    pixelData.resize(width * height * 3);
    bool isTopDown = infoHeader->biHeight < 0;

    if (SIMDSupport::sseSupported() && width >= 4) {
        // SSE优化路径
        for (int y = 0; y < height; y++) {
            int srcRow = isTopDown ? y : (height - 1 - y);
            const unsigned char* srcRowData = srcPixels + srcRow * rowSize;
            unsigned char* dstRowData = pixelData.data() + y * width * 3;

            // 预取下一行
            if (y + 1 < height) {
                int nextSrcRow = isTopDown ? (y + 1) : (height - 2 - y);
                _mm_prefetch((const char*)(srcPixels + nextSrcRow * rowSize), _MM_HINT_T0);
            }

            int x = 0;
            // SSE处理 - 每次处理4个像素(12字节)
            for (; x <= width - 4; x += 4) {
                __m128i pixels = _mm_loadu_si128((__m128i*)(srcRowData + x * 3));
                _mm_storeu_si128((__m128i*)(dstRowData + x * 3), pixels);
            }

            // 处理剩余像素
            for (; x < width; x++) {
                int srcIndex = x * 3;
                int dstIndex = x * 3;
                dstRowData[dstIndex] = srcRowData[srcIndex];
                dstRowData[dstIndex + 1] = srcRowData[srcIndex + 1];
                dstRowData[dstIndex + 2] = srcRowData[srcIndex + 2];
            }
        }
    }
    else {
        // 标量路径
        for (int y = 0; y < height; y++) {
            int srcRow = isTopDown ? y : (height - 1 - y);
            const unsigned char* srcRowData = srcPixels + srcRow * rowSize;
            unsigned char* dstRowData = pixelData.data() + y * width * 3;

            for (int x = 0; x < width; x++) {
                int srcIndex = x * 3;
                int dstIndex = x * 3;
                dstRowData[dstIndex] = srcRowData[srcIndex];
                dstRowData[dstIndex + 1] = srcRowData[srcIndex + 1];
                dstRowData[dstIndex + 2] = srcRowData[srcIndex + 2];
            }
        }
    }

    return true;
}


YoloBaseDetectorDML::YoloBaseDetectorDML() {
    // 初始化SIMD检测
	DetectCPUFeatures(SIMDSupport::avx2Supported(), SIMDSupport::sseSupported(), SIMDSupport::initialized());
}

YoloBaseDetectorDML::~YoloBaseDetectorDML() {
    ReleaseResources();
}


bool YoloBaseDetectorDML::InitializePath(const char* weightsPath, int device, int numThreads) {
    if (!CheckStatus(m_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "SuperResolutionA", &m_env), __LINE__)) {
        return false;
    }

    if (!CheckStatus(m_ort->CreateSessionOptions(&m_session_options), __LINE__)) {
        return false;
    }

    // ORT_ENABLE_ALL: enables Conv+BN+Activation fusion etc. ~10-30% faster on DML.
    if (!CheckStatus(m_ort->SetSessionGraphOptimizationLevel(m_session_options, ORT_ENABLE_ALL), __LINE__)) {
        return false;
    }

    if (!CheckStatus(m_ort->DisableMemPattern(m_session_options), __LINE__)) {
        return false;
    }

    m_ort->SetSessionExecutionMode(m_session_options, ORT_SEQUENTIAL);

    if (!CheckStatus(m_ort->SetIntraOpNumThreads(m_session_options, numThreads), __LINE__)) {
        return false;
    }

    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_DML(m_session_options, device);

    if (!CheckStatus(m_ort->CreateSession(m_env, String2WString(weightsPath).c_str(), m_session_options, &m_session), __LINE__)) {
        return false;
    }

    if (!CheckStatus(m_ort->GetAllocatorWithDefaultOptions(&m_allocator), __LINE__)) {
        return false;
    }

    if (!AutoDetectIONames()) {
        return false;
    }

    if (!ParseModelInfo()) {
        return false;
    }

    return true;
}


bool YoloBaseDetectorDML::InitializeData(const unsigned char* modelData, size_t modelLen, int device, int numThreads) {
    if (!CheckStatus(m_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "SuperResolutionA", &m_env), __LINE__)) {
        return false;
    }

    if (!CheckStatus(m_ort->CreateSessionOptions(&m_session_options), __LINE__)) {
        return false;
    }

    // ORT_ENABLE_ALL: enables Conv+BN+Activation fusion etc. ~10-30% faster on DML.
    if (!CheckStatus(m_ort->SetSessionGraphOptimizationLevel(m_session_options, ORT_ENABLE_ALL), __LINE__)) {
        return false;
    }

    if (!CheckStatus(m_ort->DisableMemPattern(m_session_options), __LINE__)) {
        return false;
    }

    m_ort->SetSessionExecutionMode(m_session_options, ORT_SEQUENTIAL);

    if (!CheckStatus(m_ort->SetIntraOpNumThreads(m_session_options, numThreads), __LINE__)) {
        return false;
    }

    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_DML(m_session_options, device);

    if (!CheckStatus(m_ort->CreateSessionFromArray(m_env, modelData, modelLen, m_session_options, &m_session), __LINE__)) {
        return false;
    }

    if (!CheckStatus(m_ort->GetAllocatorWithDefaultOptions(&m_allocator), __LINE__)) {
        return false;
    }

    if (!AutoDetectIONames()) {
        return false;
    }

    if (!ParseModelInfo()) {
        return false;
    }

    return true;
}

std::wstring YoloBaseDetectorDML::String2WString(const std::string& s) {
    if (s.empty()) return {};
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    if (len <= 0) return {};
    std::wstring result(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &result[0], len);
    return result;
}

bool YoloBaseDetectorDML::CheckStatus(OrtStatus* status, int line) {
    if (status != NULL) {
        const char* msg = m_ort->GetErrorMessage(status);
        m_ort->ReleaseStatus(status);
        return false;
    }
    return true;
}


bool YoloBaseDetectorDML::AutoDetectIONames() {
    if (!m_session || !m_allocator) {
        return false;
    }

    try {
        size_t num_input_nodes = 0;
        if (!CheckStatus(m_ort->SessionGetInputCount(m_session, &num_input_nodes), __LINE__)) {
            return false;
        }

        if (num_input_nodes > 0) {
            char* input_name_temp = nullptr;
            if (!CheckStatus(m_ort->SessionGetInputName(m_session, 0, m_allocator, &input_name_temp), __LINE__)) {
                return false;
            }

            if (input_name_temp) {
                static char input_name_buffer[256];
                strncpy(input_name_buffer, input_name_temp, sizeof(input_name_buffer) - 1);
                input_name_buffer[sizeof(input_name_buffer) - 1] = '\0';
                m_input_name[0] = input_name_buffer;
                m_ort->AllocatorFree(m_allocator, input_name_temp);
            }
        }

        size_t num_output_nodes = 0;
        if (!CheckStatus(m_ort->SessionGetOutputCount(m_session, &num_output_nodes), __LINE__)) {
            return false;
        }

        if (num_output_nodes > 0) {
            char* output_name_temp = nullptr;
            if (!CheckStatus(m_ort->SessionGetOutputName(m_session, 0, m_allocator, &output_name_temp), __LINE__)) {
                return false;
            }

            if (output_name_temp) {
                static char output_name_buffer[256];
                strncpy(output_name_buffer, output_name_temp, sizeof(output_name_buffer) - 1);
                output_name_buffer[sizeof(output_name_buffer) - 1] = '\0';
                m_output_name[0] = output_name_buffer;
                m_ort->AllocatorFree(m_allocator, output_name_temp);
            }
        }

        return true;
    }
    catch (const std::exception& e) {
        return false;
    }
}

std::map<int, std::string> YoloBaseDetectorDML::ReadClassNames() {
    m_classNames.clear();
    if (!m_session || !m_ort) return m_classNames;

    OrtModelMetadata* metadata = nullptr;
    if (m_ort->SessionGetModelMetadata(m_session, &metadata) != nullptr || !metadata) {
        return m_classNames;
    }

    // Try to read "names" key from custom metadata (YOLO convention: {0: 'cls0', 1: 'cls1', ...})
    OrtAllocator* alloc = nullptr;
    m_ort->GetAllocatorWithDefaultOptions(&alloc);

    char* value = nullptr;
    OrtStatus* st = m_ort->ModelMetadataLookupCustomMetadataMap(metadata, alloc, "names", &value);
    if (st == nullptr && value) {
        // Parse Python-dict-style string: {0: 'name0', 1: 'name1', ...}
        std::string raw(value);
        alloc->Free(alloc, value);

        // Simple parser for {int: 'string', ...}
        size_t pos = 0;
        while (pos < raw.size()) {
            // Find next digit
            while (pos < raw.size() && (raw[pos] < '0' || raw[pos] > '9')) pos++;
            if (pos >= raw.size()) break;

            // Read class id
            int cls_id = 0;
            while (pos < raw.size() && raw[pos] >= '0' && raw[pos] <= '9') {
                cls_id = cls_id * 10 + (raw[pos] - '0');
                pos++;
            }

            // Find next quote (single or double)
            while (pos < raw.size() && raw[pos] != '\'' && raw[pos] != '"') pos++;
            if (pos >= raw.size()) break;
            char quote = raw[pos];
            pos++; // skip opening quote

            // Read name until closing quote
            std::string name;
            while (pos < raw.size() && raw[pos] != quote) {
                name += raw[pos];
                pos++;
            }
            if (pos < raw.size()) pos++; // skip closing quote

            m_classNames[cls_id] = name;
        }

        std::cout << "[AiMod] Model has " << m_classNames.size() << " classes:";
        for (auto& kv : m_classNames) {
            std::cout << " " << kv.first << "=" << kv.second;
        }
        std::cout << std::endl;
    } else {
        if (st) m_ort->ReleaseStatus(st);
        // Fallback: try to infer class count from output shape
        std::cout << "[AiMod] No class names in model metadata" << std::endl;
    }

    m_ort->ReleaseModelMetadata(metadata);
    return m_classNames;
}

bool YoloBaseDetectorDML::ParseModelInfo() {
    if (!ParseInput()) {
        return false;
    }

    if (!ParseOutput()) {
        return false;
    }

    if (!CheckStatus(m_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m_memory_info), __LINE__)) {
        return false;
    }

    if (m_input_tensor) {
        m_ort->ReleaseValue(m_input_tensor);
        m_input_tensor = nullptr;
    }
    if (!CheckStatus(m_ort->CreateTensorWithDataAsOrtValue(m_memory_info, m_blob,
        m_input_tensor_size * sizeof(float), m_input_dims.data(), m_input_dims.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &m_input_tensor), __LINE__)) {
        return false;
    }

    return true;
}

bool YoloBaseDetectorDML::ParseInput() {
    size_t input_num = 0;
    size_t shape_size = 0;
    char* input_names_temp = nullptr;
    OrtTypeInfo* input_typeinfo = nullptr;
    const OrtTensorTypeAndShapeInfo* Input_tensor_info = nullptr;

    if (!CheckStatus(m_ort->SessionGetInputCount(m_session, &input_num), __LINE__)) {
        return false;
    }

    for (size_t i = 0; i < input_num; i++) {
        if (!CheckStatus(m_ort->SessionGetInputName(m_session, i, m_allocator, &input_names_temp), __LINE__)) {
            return false;
        }

        if (strcmp(*(m_input_name), input_names_temp) != 0) {
            continue;
        }

        if (!CheckStatus(m_ort->SessionGetInputTypeInfo(m_session, i, &input_typeinfo), __LINE__)) {
            return false;
        }

        if (!CheckStatus(m_ort->CastTypeInfoToTensorInfo(input_typeinfo, &Input_tensor_info), __LINE__)) {
            return false;
        }

        if (!CheckStatus(m_ort->GetDimensionsCount(Input_tensor_info, &shape_size), __LINE__)) {
            return false;
        }

        std::vector<int64_t> temp;
        temp.resize(shape_size);
        if (!CheckStatus(m_ort->GetDimensions(Input_tensor_info, temp.data(), shape_size), __LINE__)) {
            return false;
        }

        if (temp.empty()) {
            return false;
        }

        if (temp[2] != temp[3]) {
            return false;
        }

        m_input_tensor_size = 1;
        for (size_t j = 0; j < shape_size; j++) {
            m_input_tensor_size *= temp[j];
        }

        m_input_dims.assign(temp.begin(), temp.end());
        m_input_dim = temp[3];

        m_blob = new float[m_input_dim * m_input_dim * 3];
        m_resized_data = new unsigned char[m_input_dim * m_input_dim * 3];
        m_total_img_pixels = m_input_dim * m_input_dim;

        m_ort->AllocatorFree(m_allocator, input_names_temp);
        return true;
    }

    return false;
}

bool YoloBaseDetectorDML::ParseOutput() {
    size_t shape_size = 0;
    size_t num_output_nodes = 0;
    char* output_names_temp = nullptr;
    OrtTypeInfo* output_typeinfo = nullptr;
    const OrtTensorTypeAndShapeInfo* output_tensor_info = nullptr;

    if (!CheckStatus(m_ort->SessionGetOutputCount(m_session, &num_output_nodes), __LINE__)) {
        return false;
    }

    for (size_t i = 0; i < num_output_nodes; i++) {
        if (!CheckStatus(m_ort->SessionGetOutputName(m_session, i, m_allocator, &output_names_temp), __LINE__)) {
            return false;
        }

        if (strcmp(*(m_output_name), output_names_temp) != 0) {
            continue;
        }

        if (!CheckStatus(m_ort->SessionGetOutputTypeInfo(m_session, i, &output_typeinfo), __LINE__)) {
            return false;
        }

        if (!CheckStatus(m_ort->CastTypeInfoToTensorInfo(output_typeinfo, &output_tensor_info), __LINE__)) {
            return false;
        }

        if (!CheckStatus(m_ort->GetDimensionsCount(output_tensor_info, &shape_size), __LINE__)) {
            return false;
        }

        std::vector<int64_t> temp;
        temp.resize(shape_size);
        if (!CheckStatus(m_ort->GetDimensions(output_tensor_info, temp.data(), shape_size), __LINE__)) {
            return false;
        }

        if (temp.empty()) {
            return false;
        }

        m_output_dims.assign(temp.begin(), temp.end());
        m_ort->AllocatorFree(m_allocator, output_names_temp);
        return true;
    }

    return false;
}

// LetterBox预处理函数 - 支持AVX2/SSE/Scalar三种模式
bool YoloBaseDetectorDML::LetterBoxPreProcess(const unsigned char* imageData, int width, int height, LetterBoxInfo& letterbox_info) {
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
    if (SIMDSupport::avx2Supported()) {
        const __m256 gray_vec = _mm256_set1_ps(114.0f / 255.0f);
        const size_t avx_size = total_pixels & ~7;

        for (size_t i = 0; i < avx_size; i += 8) {
            _mm256_storeu_ps(&r_ptr[i], gray_vec);
            _mm256_storeu_ps(&g_ptr[i], gray_vec);
            _mm256_storeu_ps(&b_ptr[i], gray_vec);
        }

        for (size_t i = avx_size; i < total_pixels; ++i) {
            r_ptr[i] = g_ptr[i] = b_ptr[i] = 114.0f / 255.0f;
        }
    }
    else if (SIMDSupport::sseSupported()) {
        const __m128 gray_vec = _mm_set1_ps(114.0f / 255.0f);
        const size_t sse_size = total_pixels & ~3;

        for (size_t i = 0; i < sse_size; i += 4) {
            _mm_storeu_ps(&r_ptr[i], gray_vec);
            _mm_storeu_ps(&g_ptr[i], gray_vec);
            _mm_storeu_ps(&b_ptr[i], gray_vec);
        }

        for (size_t i = sse_size; i < total_pixels; ++i) {
            r_ptr[i] = g_ptr[i] = b_ptr[i] = 114.0f / 255.0f;
        }
    }
    else {
        for (size_t i = 0; i < total_pixels; ++i) {
            r_ptr[i] = g_ptr[i] = b_ptr[i] = 114.0f / 255.0f;
        }
    }

    // 预计算坐标映射 - 使用 LUT 缓存，避免每帧堆分配
    const float x_ratio = static_cast<float>(width) / new_unpad_w;
    const float y_ratio = static_cast<float>(height) / new_unpad_h;

    if (m_last_width != width || m_last_height != height ||
        m_last_resized_width != new_unpad_w || m_last_resized_height != new_unpad_h) {

        m_src_y_lut.resize(new_unpad_h);
        for (int dst_y = 0; dst_y < new_unpad_h; ++dst_y) {
            float src_y = (dst_y + 0.5f) * y_ratio - 0.5f;
            int nearest_y = static_cast<int>(src_y + 0.5f);
            m_src_y_lut[dst_y] = std::max(0, std::min(nearest_y, height - 1));
        }

        m_src_x_lut.resize(new_unpad_w);
        for (int dst_x = 0; dst_x < new_unpad_w; ++dst_x) {
            float src_x = (dst_x + 0.5f) * x_ratio - 0.5f;
            int nearest_x = static_cast<int>(src_x + 0.5f);
            m_src_x_lut[dst_x] = std::max(0, std::min(nearest_x, width - 1));
        }

        m_last_width = width;
        m_last_height = height;
        m_last_resized_width = new_unpad_w;
        m_last_resized_height = new_unpad_h;
    }

    const int* y_src_indices = m_src_y_lut.data();
    const int* x_src_indices = m_src_x_lut.data();

    // 主处理循环 - 根据SIMD支持情况选择处理方式
    const float norm = 1.0f / 255.0f;

    if (SIMDSupport::avx2Supported()) {
        // ===== AVX2 路径（重写：i32gather_epi32 + cvtepi32_ps，真正 SIMD 化 byte→float）=====
        // 一次 gather 8 个 BGR0 DWORD → mask/shift 拆 R/G/B → cvtepi32_ps → mul → store
        const __m256 norm_vec = _mm256_set1_ps(norm);
        const __m256i mask_ff = _mm256_set1_epi32(0xFF);
        const __m256i three_v = _mm256_set1_epi32(3);

        // 边界保护：i32gather_epi32 读 4 字节，最后像素 stride=3 时 +3 会越界 1 字节。
        // 留 1 个像素给标量回落（gather 上界改为 width-1 处对应索引）。
        const int row_bytes = width * 3;

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
            const int avx_width = new_unpad_w & ~7;

            for (; dst_x < avx_width; dst_x += 8) {
                // 8 个像素索引 -> 8 个字节偏移 (idx*3)
                __m256i x_idx = _mm256_loadu_si256((const __m256i*)(x_src_indices + dst_x));
                __m256i byte_off = _mm256_mullo_epi32(x_idx, three_v);

                // 检查最大索引是否安全（<= width-2，保证 +3 不越界）
                int max_idx = x_src_indices[dst_x + 7];
                if (max_idx + 1 < width) {
                    // 一次 gather 8 个 BGR0 DWORD（base 是 src_row 起始字节，scale=1）
                    __m256i bgr0 = _mm256_i32gather_epi32(
                        reinterpret_cast<const int*>(src_row), byte_off, 1);

                    // mask + shift 拆出 B / G / R 三个通道（uint8 → uint32）
                    __m256i b_i = _mm256_and_si256(bgr0, mask_ff);
                    __m256i g_i = _mm256_and_si256(_mm256_srli_epi32(bgr0, 8),  mask_ff);
                    __m256i r_i = _mm256_and_si256(_mm256_srli_epi32(bgr0, 16), mask_ff);

                    // SIMD cvt + 归一化
                    __m256 r_vals = _mm256_mul_ps(_mm256_cvtepi32_ps(r_i), norm_vec);
                    __m256 g_vals = _mm256_mul_ps(_mm256_cvtepi32_ps(g_i), norm_vec);
                    __m256 b_vals = _mm256_mul_ps(_mm256_cvtepi32_ps(b_i), norm_vec);

                    _mm256_storeu_ps(&r_ptr[dst_row_start + dst_x], r_vals);
                    _mm256_storeu_ps(&g_ptr[dst_row_start + dst_x], g_vals);
                    _mm256_storeu_ps(&b_ptr[dst_row_start + dst_x], b_vals);
                } else {
                    // 该 8 像素组中含 width-1 的尾像素，标量回落避免越界
                    for (int k = 0; k < 8; ++k) {
                        const int sx = x_src_indices[dst_x + k];
                        const unsigned char* p = src_row + sx * 3;
                        const int di = dst_row_start + dst_x + k;
                        r_ptr[di] = p[2] * norm;
                        g_ptr[di] = p[1] * norm;
                        b_ptr[di] = p[0] * norm;
                    }
                }
            }

            // 剩余 < 8 个像素的尾巴
            for (; dst_x < new_unpad_w; ++dst_x) {
                const int src_x = x_src_indices[dst_x];
                const unsigned char* p = src_row + src_x * 3;
                const int dst_idx = dst_row_start + dst_x;
                r_ptr[dst_idx] = p[2] * norm;
                g_ptr[dst_idx] = p[1] * norm;
                b_ptr[dst_idx] = p[0] * norm;
            }
        }

        _mm256_zeroupper();
    }
    else if (SIMDSupport::sseSupported()) {
        // ===== SSE4.1 路径（重写：cvtepu8_epi32 + cvtepi32_ps，4 像素一组）=====
        // 注：SSE 没有 gather；用 _mm_cvtsi32_si128 加载 4 字节 BGR0 后 cvtepu8_epi32 拓宽。
        // 索引非连续 → 必须每像素一次 4-byte load + insert，但 cvt 走 SIMD。
        const __m128 norm_vec = _mm_set1_ps(norm);
        const __m128i shuf_b = _mm_setr_epi8(0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1);
        const __m128i shuf_g = _mm_setr_epi8(1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1);
        const __m128i shuf_r = _mm_setr_epi8(2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1);

        for (int dst_y = 0; dst_y < new_unpad_h; ++dst_y) {
            const int src_y = y_src_indices[dst_y];
            const unsigned char* src_row = imageData + src_y * width * 3;
            const int dst_row_start = (dst_y + pad_top) * m_input_dim + pad_left;

            if (dst_y + 1 < new_unpad_h) {
                const int next_src_y = y_src_indices[dst_y + 1];
                _mm_prefetch((const char*)(imageData + next_src_y * width * 3), _MM_HINT_T0);
            }

            int dst_x = 0;
            const int sse_width = new_unpad_w & ~3;

            for (; dst_x < sse_width; dst_x += 4) {
                int max_idx = x_src_indices[dst_x + 3];
                if (max_idx + 1 < width) {
                    // 加载 4 个 BGR0 DWORD 进 __m128i (4×4=16 字节)
                    int p0 = *reinterpret_cast<const int*>(src_row + x_src_indices[dst_x + 0] * 3);
                    int p1 = *reinterpret_cast<const int*>(src_row + x_src_indices[dst_x + 1] * 3);
                    int p2 = *reinterpret_cast<const int*>(src_row + x_src_indices[dst_x + 2] * 3);
                    int p3 = *reinterpret_cast<const int*>(src_row + x_src_indices[dst_x + 3] * 3);
                    __m128i pix = _mm_setr_epi32(p0, p1, p2, p3);

                    // pshufb 一次拆出 4 个 B/G/R（拓宽到 32-bit，高 24 位置 0）
                    __m128i b_i = _mm_shuffle_epi8(pix, shuf_b);
                    __m128i g_i = _mm_shuffle_epi8(pix, shuf_g);
                    __m128i r_i = _mm_shuffle_epi8(pix, shuf_r);

                    __m128 r_vals = _mm_mul_ps(_mm_cvtepi32_ps(r_i), norm_vec);
                    __m128 g_vals = _mm_mul_ps(_mm_cvtepi32_ps(g_i), norm_vec);
                    __m128 b_vals = _mm_mul_ps(_mm_cvtepi32_ps(b_i), norm_vec);

                    _mm_storeu_ps(&r_ptr[dst_row_start + dst_x], r_vals);
                    _mm_storeu_ps(&g_ptr[dst_row_start + dst_x], g_vals);
                    _mm_storeu_ps(&b_ptr[dst_row_start + dst_x], b_vals);
                } else {
                    // 含尾像素，标量回落
                    for (int k = 0; k < 4; ++k) {
                        const int sx = x_src_indices[dst_x + k];
                        const unsigned char* p = src_row + sx * 3;
                        const int di = dst_row_start + dst_x + k;
                        r_ptr[di] = p[2] * norm;
                        g_ptr[di] = p[1] * norm;
                        b_ptr[di] = p[0] * norm;
                    }
                }
            }

            for (; dst_x < new_unpad_w; ++dst_x) {
                const int src_x = x_src_indices[dst_x];
                const unsigned char* p = src_row + src_x * 3;
                const int dst_idx = dst_row_start + dst_x;
                r_ptr[dst_idx] = p[2] * norm;
                g_ptr[dst_idx] = p[1] * norm;
                b_ptr[dst_idx] = p[0] * norm;
            }
        }
    }
    else {
        // 标量路径
        for (int dst_y = 0; dst_y < new_unpad_h; ++dst_y) {
            const int src_y = y_src_indices[dst_y];
            const unsigned char* src_row = imageData + src_y * width * 3;
            const int dst_row_start = (dst_y + pad_top) * m_input_dim + pad_left;

            for (int dst_x = 0; dst_x < new_unpad_w; ++dst_x) {
                const int src_x = x_src_indices[dst_x];
                const unsigned char* src_pixel = src_row + src_x * 3;
                const int dst_idx = dst_row_start + dst_x;

                r_ptr[dst_idx] = src_pixel[2] * norm;
                g_ptr[dst_idx] = src_pixel[1] * norm;
                b_ptr[dst_idx] = src_pixel[0] * norm;
            }
        }
    }

    return true;
}

float YoloBaseDetectorDML::CalculateIOU(const DMLObject& a, const DMLObject& b) {
    float cleft = (std::max)(a.x, b.x);
    float ctop = (std::max)(a.y, b.y);
    float cright = (std::min)(a.x + a.width, b.x + b.width);
    float cbottom = (std::min)(a.y + a.height, b.y + b.height);

    float c_area = (std::max)(cright - cleft, 0.0f) * (std::max)(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = (std::max)(0.0f, a.width) * (std::max)(0.0f, a.height);
    float b_area = (std::max)(0.0f, b.width) * (std::max)(0.0f, b.height);
    return c_area / (a_area + b_area - c_area);
}

// ScaleBoxes - AVX2 / SSE / Scalar three paths
void YoloBaseDetectorDML::ScaleBoxes(std::vector<DMLObject>& objects, const LetterBoxInfo& letterbox_info) {
    if (objects.empty()) return;

    if (SIMDSupport::avx2Supported()) {
        const __m256 pad_w_vec = _mm256_set1_ps(letterbox_info.pad_w);
        const __m256 pad_h_vec = _mm256_set1_ps(letterbox_info.pad_h);
        const __m256 inv_scale = _mm256_set1_ps(1.0f / letterbox_info.scale);
        const __m256 zero_vec  = _mm256_setzero_ps();
        const __m256 max_w_vec = _mm256_set1_ps(static_cast<float>(m_original_width));
        const __m256 max_h_vec = _mm256_set1_ps(static_cast<float>(m_original_height));

        size_t i = 0;
        const size_t avx_count = objects.size() & ~size_t(7);
        for (; i < avx_count; i += 8) {
            __m256 x_vec = _mm256_setr_ps(objects[i+0].x, objects[i+1].x, objects[i+2].x, objects[i+3].x,
                                          objects[i+4].x, objects[i+5].x, objects[i+6].x, objects[i+7].x);
            __m256 y_vec = _mm256_setr_ps(objects[i+0].y, objects[i+1].y, objects[i+2].y, objects[i+3].y,
                                          objects[i+4].y, objects[i+5].y, objects[i+6].y, objects[i+7].y);
            __m256 w_vec = _mm256_setr_ps(objects[i+0].width, objects[i+1].width, objects[i+2].width, objects[i+3].width,
                                          objects[i+4].width, objects[i+5].width, objects[i+6].width, objects[i+7].width);
            __m256 h_vec = _mm256_setr_ps(objects[i+0].height, objects[i+1].height, objects[i+2].height, objects[i+3].height,
                                          objects[i+4].height, objects[i+5].height, objects[i+6].height, objects[i+7].height);
            x_vec = _mm256_mul_ps(_mm256_sub_ps(x_vec, pad_w_vec), inv_scale);
            y_vec = _mm256_mul_ps(_mm256_sub_ps(y_vec, pad_h_vec), inv_scale);
            w_vec = _mm256_mul_ps(w_vec, inv_scale);
            h_vec = _mm256_mul_ps(h_vec, inv_scale);
            x_vec = _mm256_max_ps(zero_vec, _mm256_min_ps(x_vec, max_w_vec));
            y_vec = _mm256_max_ps(zero_vec, _mm256_min_ps(y_vec, max_h_vec));
            w_vec = _mm256_min_ps(w_vec, _mm256_sub_ps(max_w_vec, x_vec));
            h_vec = _mm256_min_ps(h_vec, _mm256_sub_ps(max_h_vec, y_vec));
            alignas(32) float xr[8], yr[8], wr[8], hr[8];
            _mm256_store_ps(xr, x_vec);
            _mm256_store_ps(yr, y_vec);
            _mm256_store_ps(wr, w_vec);
            _mm256_store_ps(hr, h_vec);
            for (int j = 0; j < 8; ++j) {
                objects[i+j].x = xr[j]; objects[i+j].y = yr[j];
                objects[i+j].width = wr[j]; objects[i+j].height = hr[j];
            }
        }
        for (; i < objects.size(); ++i) {
            auto& obj = objects[i];
            float x = (obj.x - letterbox_info.pad_w) / letterbox_info.scale;
            float y = (obj.y - letterbox_info.pad_h) / letterbox_info.scale;
            float w = obj.width / letterbox_info.scale;
            float h = obj.height / letterbox_info.scale;
            x = std::max(0.0f, std::min(x, static_cast<float>(m_original_width)));
            y = std::max(0.0f, std::min(y, static_cast<float>(m_original_height)));
            w = std::min(w, static_cast<float>(m_original_width) - x);
            h = std::min(h, static_cast<float>(m_original_height) - y);
            obj.x = x; obj.y = y; obj.width = w; obj.height = h;
        }
        return;
    }

    if (SIMDSupport::sseSupported()) {
        // SSE路径
        const __m128 pad_w_vec = _mm_set1_ps(letterbox_info.pad_w);
        const __m128 pad_h_vec = _mm_set1_ps(letterbox_info.pad_h);
        const __m128 scale_vec = _mm_set1_ps(letterbox_info.scale);
        const __m128 zero_vec = _mm_setzero_ps();
        const __m128 max_w_vec = _mm_set1_ps(static_cast<float>(m_original_width));
        const __m128 max_h_vec = _mm_set1_ps(static_cast<float>(m_original_height));

        size_t i = 0;
        const size_t sse_count = objects.size() & ~3;

        for (; i < sse_count; i += 4) {
            __m128 x_vec = _mm_setr_ps(objects[i + 0].x, objects[i + 1].x, objects[i + 2].x, objects[i + 3].x);
            __m128 y_vec = _mm_setr_ps(objects[i + 0].y, objects[i + 1].y, objects[i + 2].y, objects[i + 3].y);
            __m128 w_vec = _mm_setr_ps(objects[i + 0].width, objects[i + 1].width, objects[i + 2].width, objects[i + 3].width);
            __m128 h_vec = _mm_setr_ps(objects[i + 0].height, objects[i + 1].height, objects[i + 2].height, objects[i + 3].height);

            // 减去padding
            x_vec = _mm_sub_ps(x_vec, pad_w_vec);
            y_vec = _mm_sub_ps(y_vec, pad_h_vec);

            // 除以缩放因子
            x_vec = _mm_div_ps(x_vec, scale_vec);
            y_vec = _mm_div_ps(y_vec, scale_vec);
            w_vec = _mm_div_ps(w_vec, scale_vec);
            h_vec = _mm_div_ps(h_vec, scale_vec);

            // 裁剪x, y到边界
            x_vec = _mm_max_ps(zero_vec, _mm_min_ps(x_vec, max_w_vec));
            y_vec = _mm_max_ps(zero_vec, _mm_min_ps(y_vec, max_h_vec));

            // 限制宽高
            __m128 remaining_w = _mm_sub_ps(max_w_vec, x_vec);
            __m128 remaining_h = _mm_sub_ps(max_h_vec, y_vec);
            w_vec = _mm_min_ps(w_vec, remaining_w);
            h_vec = _mm_min_ps(h_vec, remaining_h);

            alignas(16) float x_result[4], y_result[4], w_result[4], h_result[4];
            _mm_store_ps(x_result, x_vec);
            _mm_store_ps(y_result, y_vec);
            _mm_store_ps(w_result, w_vec);
            _mm_store_ps(h_result, h_vec);

            for (int j = 0; j < 4; ++j) {
                objects[i + j].x = x_result[j];
                objects[i + j].y = y_result[j];
                objects[i + j].width = w_result[j];
                objects[i + j].height = h_result[j];
            }
        }

        // 处理剩余对象
        for (; i < objects.size(); ++i) {
            auto& obj = objects[i];

            // 坐标变换
            float x = (obj.x - letterbox_info.pad_w) / letterbox_info.scale;
            float y = (obj.y - letterbox_info.pad_h) / letterbox_info.scale;
            float w = obj.width / letterbox_info.scale;
            float h = obj.height / letterbox_info.scale;

            // 边界裁剪
            x = std::max(0.0f, std::min(x, static_cast<float>(m_original_width)));
            y = std::max(0.0f, std::min(y, static_cast<float>(m_original_height)));
            w = std::min(w, static_cast<float>(m_original_width) - x);
            h = std::min(h, static_cast<float>(m_original_height) - y);

            obj.x = x;
            obj.y = y;
            obj.width = w;
            obj.height = h;
        }
    }
    else {
        // 标量路径
        for (size_t i = 0; i < objects.size(); ++i) {
            auto& obj = objects[i];

            // 坐标变换
            float x = (obj.x - letterbox_info.pad_w) / letterbox_info.scale;
            float y = (obj.y - letterbox_info.pad_h) / letterbox_info.scale;
            float w = obj.width / letterbox_info.scale;
            float h = obj.height / letterbox_info.scale;

            // 边界裁剪
            x = std::max(0.0f, std::min(x, static_cast<float>(m_original_width)));
            y = std::max(0.0f, std::min(y, static_cast<float>(m_original_height)));
            w = std::min(w, static_cast<float>(m_original_width) - x);
            h = std::min(h, static_cast<float>(m_original_height) - y);

            obj.x = x;
            obj.y = y;
            obj.width = w;
            obj.height = h;
        }
    }
}

// NMS函数 - SSE/Scalar 两路径
std::vector<DMLObject> YoloBaseDetectorDML::NMSBoxes(std::vector<DMLObject>& objects, float threshold) {
    if (objects.empty()) return {};

    // 按置信度降序排序
    std::stable_sort(objects.begin(), objects.end(),
        [](const DMLObject& a, const DMLObject& b) {
            return a.prob > b.prob;
        });

    std::vector<DMLObject> result;
    result.reserve(objects.size());
    std::vector<bool> suppressed(objects.size(), false);

    if (SIMDSupport::avx2Supported()) {
        // ===== AVX2 path: 8 boxes per iteration =====
        const __m256 threshold_vec = _mm256_set1_ps(threshold);
        for (size_t i = 0; i < objects.size(); ++i) {
            if (suppressed[i]) continue;
            result.emplace_back(objects[i]);
            const float x1_i = objects[i].x;
            const float y1_i = objects[i].y;
            const float x2_i = objects[i].x + objects[i].width;
            const float y2_i = objects[i].y + objects[i].height;
            const float area_i = objects[i].width * objects[i].height;
            const int label_i = objects[i].label;
            if (area_i <= 0.0f) continue;
            const __m256 x1iv = _mm256_set1_ps(x1_i);
            const __m256 y1iv = _mm256_set1_ps(y1_i);
            const __m256 x2iv = _mm256_set1_ps(x2_i);
            const __m256 y2iv = _mm256_set1_ps(y2_i);
            const __m256 areaiv = _mm256_set1_ps(area_i);
            size_t j = i + 1;
            for (; j + 7 < objects.size(); j += 8) {
                bool any = false;
                for (int k = 0; k < 8; ++k) {
                    if (!suppressed[j+k] && objects[j+k].label == label_i) { any = true; break; }
                }
                if (!any) continue;
                __m256 x1jv = _mm256_setr_ps(objects[j+0].x, objects[j+1].x, objects[j+2].x, objects[j+3].x,
                                             objects[j+4].x, objects[j+5].x, objects[j+6].x, objects[j+7].x);
                __m256 y1jv = _mm256_setr_ps(objects[j+0].y, objects[j+1].y, objects[j+2].y, objects[j+3].y,
                                             objects[j+4].y, objects[j+5].y, objects[j+6].y, objects[j+7].y);
                __m256 x2jv = _mm256_setr_ps(objects[j+0].x+objects[j+0].width, objects[j+1].x+objects[j+1].width,
                                             objects[j+2].x+objects[j+2].width, objects[j+3].x+objects[j+3].width,
                                             objects[j+4].x+objects[j+4].width, objects[j+5].x+objects[j+5].width,
                                             objects[j+6].x+objects[j+6].width, objects[j+7].x+objects[j+7].width);
                __m256 y2jv = _mm256_setr_ps(objects[j+0].y+objects[j+0].height, objects[j+1].y+objects[j+1].height,
                                             objects[j+2].y+objects[j+2].height, objects[j+3].y+objects[j+3].height,
                                             objects[j+4].y+objects[j+4].height, objects[j+5].y+objects[j+5].height,
                                             objects[j+6].y+objects[j+6].height, objects[j+7].y+objects[j+7].height);
                __m256 areajv = _mm256_setr_ps(objects[j+0].width*objects[j+0].height, objects[j+1].width*objects[j+1].height,
                                               objects[j+2].width*objects[j+2].height, objects[j+3].width*objects[j+3].height,
                                               objects[j+4].width*objects[j+4].height, objects[j+5].width*objects[j+5].height,
                                               objects[j+6].width*objects[j+6].height, objects[j+7].width*objects[j+7].height);
                __m256 ix1 = _mm256_max_ps(x1iv, x1jv);
                __m256 iy1 = _mm256_max_ps(y1iv, y1jv);
                __m256 ix2 = _mm256_min_ps(x2iv, x2jv);
                __m256 iy2 = _mm256_min_ps(y2iv, y2jv);
                __m256 zero = _mm256_setzero_ps();
                __m256 iw = _mm256_max_ps(zero, _mm256_sub_ps(ix2, ix1));
                __m256 ih = _mm256_max_ps(zero, _mm256_sub_ps(iy2, iy1));
                __m256 inter = _mm256_mul_ps(iw, ih);
                __m256 uni = _mm256_sub_ps(_mm256_add_ps(areaiv, areajv), inter);
                __m256 valid = _mm256_cmp_ps(uni, zero, _CMP_GT_OQ);
                __m256 iou = _mm256_div_ps(inter, uni);
                iou = _mm256_blendv_ps(zero, iou, valid);
                __m256 sup = _mm256_cmp_ps(iou, threshold_vec, _CMP_GE_OQ);
                int mask = _mm256_movemask_ps(sup);
                if (mask == 0) continue;
                for (int k = 0; k < 8; ++k) {
                    if ((mask & (1<<k)) && !suppressed[j+k] && objects[j+k].label == label_i) {
                        suppressed[j+k] = true;
                    }
                }
            }
            // tail scalar
            for (; j < objects.size(); ++j) {
                if (suppressed[j] || objects[i].label != objects[j].label) continue;
                float x1_j = objects[j].x, y1_j = objects[j].y;
                float x2_j = objects[j].x + objects[j].width, y2_j = objects[j].y + objects[j].height;
                float area_j = objects[j].width * objects[j].height;
                float ix1 = std::max(x1_i, x1_j), iy1 = std::max(y1_i, y1_j);
                float ix2 = std::min(x2_i, x2_j), iy2 = std::min(y2_i, y2_j);
                float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
                float uni = area_i + area_j - inter;
                float iou = (uni > 0) ? (inter / uni) : 0;
                if (iou >= threshold) suppressed[j] = true;
            }
        }
        return result;
    }

    if (SIMDSupport::sseSupported()) {
        // SSE路径
        const __m128 threshold_vec = _mm_set1_ps(threshold);

        for (size_t i = 0; i < objects.size(); ++i) {
            if (suppressed[i]) continue;

            result.emplace_back(objects[i]);

            const float x1_i = objects[i].x;
            const float y1_i = objects[i].y;
            const float x2_i = objects[i].x + objects[i].width;
            const float y2_i = objects[i].y + objects[i].height;
            const float area_i = objects[i].width * objects[i].height;
            const int label_i = objects[i].label;

            if (area_i <= 0.0f) continue;

            const __m128 x1_i_vec = _mm_set1_ps(x1_i);
            const __m128 y1_i_vec = _mm_set1_ps(y1_i);
            const __m128 x2_i_vec = _mm_set1_ps(x2_i);
            const __m128 y2_i_vec = _mm_set1_ps(y2_i);
            const __m128 area_i_vec = _mm_set1_ps(area_i);

            size_t j = i + 1;

            // SSE处理 - 一次处理4个框
            for (; j + 3 < objects.size(); j += 4) {
                bool has_valid_candidate = false;
                for (int k = 0; k < 4; ++k) {
                    if (!suppressed[j + k] && objects[j + k].label == label_i) {
                        has_valid_candidate = true;
                        break;
                    }
                }

                if (!has_valid_candidate) continue;

                // 预取下一批数据
                if (j + 8 < objects.size()) {
                    _mm_prefetch((const char*)&objects[j + 8], _MM_HINT_T0);
                }

                __m128 x1_j_vec = _mm_setr_ps(
                    objects[j + 0].x, objects[j + 1].x, objects[j + 2].x, objects[j + 3].x
                );
                __m128 y1_j_vec = _mm_setr_ps(
                    objects[j + 0].y, objects[j + 1].y, objects[j + 2].y, objects[j + 3].y
                );
                __m128 x2_j_vec = _mm_setr_ps(
                    objects[j + 0].x + objects[j + 0].width, objects[j + 1].x + objects[j + 1].width,
                    objects[j + 2].x + objects[j + 2].width, objects[j + 3].x + objects[j + 3].width
                );
                __m128 y2_j_vec = _mm_setr_ps(
                    objects[j + 0].y + objects[j + 0].height, objects[j + 1].y + objects[j + 1].height,
                    objects[j + 2].y + objects[j + 2].height, objects[j + 3].y + objects[j + 3].height
                );
                __m128 area_j_vec = _mm_setr_ps(
                    objects[j + 0].width * objects[j + 0].height, objects[j + 1].width * objects[j + 1].height,
                    objects[j + 2].width * objects[j + 2].height, objects[j + 3].width * objects[j + 3].height
                );

                // 计算交集
                __m128 inter_x1 = _mm_max_ps(x1_i_vec, x1_j_vec);
                __m128 inter_y1 = _mm_max_ps(y1_i_vec, y1_j_vec);
                __m128 inter_x2 = _mm_min_ps(x2_i_vec, x2_j_vec);
                __m128 inter_y2 = _mm_min_ps(y2_i_vec, y2_j_vec);

                __m128 inter_w = _mm_max_ps(_mm_setzero_ps(), _mm_sub_ps(inter_x2, inter_x1));
                __m128 inter_h = _mm_max_ps(_mm_setzero_ps(), _mm_sub_ps(inter_y2, inter_y1));
                __m128 inter_area = _mm_mul_ps(inter_w, inter_h);

                // 计算并集
                __m128 union_area = _mm_sub_ps(_mm_add_ps(area_i_vec, area_j_vec), inter_area);

                // 计算IOU
                __m128 zero_vec = _mm_setzero_ps();
                __m128 valid_union_mask = _mm_cmpgt_ps(union_area, zero_vec);
                __m128 iou_vec = _mm_div_ps(inter_area, union_area);
                iou_vec = _mm_blendv_ps(zero_vec, iou_vec, valid_union_mask);

                // 检查是否超过阈值
                __m128 suppress_mask = _mm_cmpge_ps(iou_vec, threshold_vec);

                alignas(16) float suppress_results[4];
                _mm_store_ps(suppress_results, suppress_mask);

                // 更新抑制状态
                for (int k = 0; k < 4; ++k) {
                    size_t idx = j + k;
                    if (!suppressed[idx] && objects[idx].label == label_i) {
                        if (suppress_results[k] != 0.0f) {
                            suppressed[idx] = true;
                        }
                    }
                }
            }

            // 处理剩余的框 - 标量
            for (; j < objects.size(); ++j) {
                if (suppressed[j] || objects[i].label != objects[j].label) continue;

                float x1_j = objects[j].x;
                float y1_j = objects[j].y;
                float x2_j = objects[j].x + objects[j].width;
                float y2_j = objects[j].y + objects[j].height;
                float area_j = objects[j].width * objects[j].height;

                float inter_x1 = std::max(x1_i, x1_j);
                float inter_y1 = std::max(y1_i, y1_j);
                float inter_x2 = std::min(x2_i, x2_j);
                float inter_y2 = std::min(y2_i, y2_j);

                float inter_area = std::max(0.0f, inter_x2 - inter_x1) *
                    std::max(0.0f, inter_y2 - inter_y1);
                float union_area = area_i + area_j - inter_area;
                float iou = (union_area > 0) ? (inter_area / union_area) : 0;

                if (iou >= threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    else {
        // 标量路径
        for (size_t i = 0; i < objects.size(); ++i) {
            if (suppressed[i]) continue;

            result.emplace_back(objects[i]);

            const float x1_i = objects[i].x;
            const float y1_i = objects[i].y;
            const float x2_i = objects[i].x + objects[i].width;
            const float y2_i = objects[i].y + objects[i].height;
            const float area_i = objects[i].width * objects[i].height;
            const int label_i = objects[i].label;

            if (area_i <= 0.0f) continue;

            for (size_t j = i + 1; j < objects.size(); ++j) {
                if (suppressed[j] || objects[i].label != objects[j].label) continue;

                float x1_j = objects[j].x;
                float y1_j = objects[j].y;
                float x2_j = objects[j].x + objects[j].width;
                float y2_j = objects[j].y + objects[j].height;
                float area_j = objects[j].width * objects[j].height;

                float inter_x1 = std::max(x1_i, x1_j);
                float inter_y1 = std::max(y1_i, y1_j);
                float inter_x2 = std::min(x2_i, x2_j);
                float inter_y2 = std::min(y2_i, y2_j);

                float inter_area = std::max(0.0f, inter_x2 - inter_x1) *
                    std::max(0.0f, inter_y2 - inter_y1);
                float union_area = area_i + area_j - inter_area;
                float iou = (union_area > 0) ? (inter_area / union_area) : 0;

                if (iou >= threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return result;
}


// 主检测函数 - 返回结构化数据
std::vector<DetectionObject> YoloBaseDetectorDML::Detect(const unsigned char* imageData, int width, int height, float conf, float nms) {
    m_detectionResults.clear();
    m_rawDetections.clear();

    // LetterBox预处理
    LetterBoxInfo letterbox_info;
    if (!LetterBoxPreProcess(imageData, width, height, letterbox_info)) {
        return m_detectionResults;
    }

    if (!m_input_tensor) {
        return m_detectionResults;
    }

    // 运行推理
    m_ort->Run(m_session, NULL, m_input_name, &m_input_tensor, 1, m_output_name, 1, &m_output_tensor);

    // 获取输出数据
    void* outputData;
    m_ort->GetTensorMutableData(m_output_tensor, &outputData);
    m_output_data = static_cast<float*>(outputData);

    // 生成检测提案
    GenerateProposals(m_output_data, m_rawDetections, conf);

    // NMS处理
    m_rawDetections = NMSBoxes(m_rawDetections, nms);

    // 坐标还原
    ScaleBoxes(m_rawDetections, letterbox_info);

    if (m_output_tensor) {
        m_ort->ReleaseValue(m_output_tensor);
        m_output_tensor = nullptr;
    }

    // 直接构造最终结果
    for (const auto& obj : m_rawDetections) {
        DetectionObject det;
        det.bbox.x = obj.x;
        det.bbox.y = obj.y;
        det.bbox.width = obj.width;
        det.bbox.height = obj.height;
        det.label = obj.label;
        det.prob = obj.prob;
        m_detectionResults.push_back(det);
    }

    return m_detectionResults;
}

void YoloBaseDetectorDML::Detect(const unsigned char* imageData, int width, int height, float conf, float nms, std::vector<DetectionObject>& output) {
    output.clear();
    m_rawDetections.clear();

    LetterBoxInfo letterbox_info;
    if (!LetterBoxPreProcess(imageData, width, height, letterbox_info)) {
        return;
    }

    if (!m_input_tensor) {
        return;
    }
    m_ort->Run(m_session, NULL, m_input_name, &m_input_tensor, 1, m_output_name, 1, &m_output_tensor);

    void* outputData;
    m_ort->GetTensorMutableData(m_output_tensor, &outputData);
    m_output_data = static_cast<float*>(outputData);

    GenerateProposals(m_output_data, m_rawDetections, conf);
    m_rawDetections = NMSBoxes(m_rawDetections, nms);
    ScaleBoxes(m_rawDetections, letterbox_info);

    if (m_output_tensor) {
        m_ort->ReleaseValue(m_output_tensor);
        m_output_tensor = nullptr;
    }

    for (const auto& obj : m_rawDetections) {
        DetectionObject det;
        det.bbox.x = obj.x;
        det.bbox.y = obj.y;
        det.bbox.width = obj.width;
        det.bbox.height = obj.height;
        det.label = obj.label;
        det.prob = obj.prob;
        output.push_back(det);
    }
}

std::vector<DetectionObject> YoloBaseDetectorDML::DetectBMP(const unsigned char* bmpData, size_t bmpSize, float conf, float nms) {
    std::vector<unsigned char> pixelData;
    int width = 0, height = 0;

    if (isBMPFile(bmpData, bmpSize)) {
        if (!extractBMPPixelData(bmpData, bmpSize, pixelData, width, height)) {
            return std::vector<DetectionObject>();
        }
        return Detect(pixelData.data(), width, height, conf, nms);
    }

    return std::vector<DetectionObject>();
}

std::vector<DetectionObject> YoloBaseDetectorDML::DetectBGR(const unsigned char* imageData, int width, int height, float conf, float nms) {
    return Detect(imageData, width, height, conf, nms);
}

void YoloBaseDetectorDML::DetectBGR(const unsigned char* imageData, int width, int height, float conf, float nms, std::vector<DetectionObject>& output) {
    Detect(imageData, width, height, conf, nms, output);
}


void YoloBaseDetectorDML::GenerateGridsAndStride() {
    m_grid_strides.clear();
    int stride = 8;
    for (size_t i = 0; i < 3; i++) {
        int num_grid_w = m_input_dims[2] / stride;
        int num_grid_h = m_input_dims[3] / stride;

        for (int g1 = 0; g1 < num_grid_w; g1++) {
            for (int g0 = 0; g0 < num_grid_h; g0++) {
                GridAndStride gs{};
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                m_grid_strides.push_back(gs);
            }
        }
        stride *= 2;
    }
}


void YoloBaseDetectorDML::ReleaseResources() {
    if (m_input_tensor) {
        m_ort->ReleaseValue(m_input_tensor);
        m_input_tensor = nullptr;
    }
    if (m_output_tensor) {
        m_ort->ReleaseValue(m_output_tensor);
        m_output_tensor = nullptr;
    }
    if (m_session) {
        m_ort->ReleaseSession(m_session);
        m_session = nullptr;
    }
    if (m_session_options) {
        m_ort->ReleaseSessionOptions(m_session_options);
        m_session_options = nullptr;
    }
    if (m_memory_info) {
        m_ort->ReleaseMemoryInfo(m_memory_info);
        m_memory_info = nullptr;
    }
    if (m_env) {
        m_ort->ReleaseEnv(m_env);
        m_env = nullptr;
    }

    if (m_blob) {
        delete[] m_blob;
        m_blob = nullptr;
    }
    if (m_resized_data) {
        delete[] m_resized_data;
        m_resized_data = nullptr;
    }
}

void YoloBaseDetectorDML::ResetState() {
    m_detectionResults.clear();
    m_rawDetections.clear();
}
