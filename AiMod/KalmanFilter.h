#pragma once
#define NOMINMAX
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include "DetectionTypes.h"


namespace detail {

    inline void meas_from_object(const DetectionObject& o, float z[5]) {
        const float cx = o.bbox.x + o.bbox.width * 0.5f;
        const float cy = o.bbox.y + o.bbox.height * 0.5f;
        z[0] = static_cast<float>(o.label);
        z[1] = cx;
        z[2] = cy;
        z[3] = o.bbox.width;
        z[4] = o.bbox.height;
    }

    inline void meas_to_bbox(const float z[5], float& x0, float& y0, float& x1, float& y1) {
        const float cx = z[1], cy = z[2], w = z[3], h = z[4];
        x0 = cx - w * 0.5f;
        y0 = cy - h * 0.5f;
        x1 = cx + w * 0.5f;
        y1 = cy + h * 0.5f;
    }

    inline float iou_from_meas(const float a[5], const float b[5]) {
        float ax0, ay0, ax1, ay1;
        float bx0, by0, bx1, by1;
        meas_to_bbox(a, ax0, ay0, ax1, ay1);
        meas_to_bbox(b, bx0, by0, bx1, by1);

        const float x_min = std::max(ax0, bx0);
        const float x_max = std::min(ax1, bx1);
        const float y_min = std::max(ay0, by0);
        const float y_max = std::min(ay1, by1);

        const float inter_w = std::max(x_max - x_min + 1.0f, 0.0f);
        const float inter_h = std::max(y_max - y_min + 1.0f, 0.0f);
        const float inter = inter_w * inter_h;
        if (inter <= 0.0f) return 0.0f;

        const float area_a = (ax1 - ax0) * (ay1 - ay0);
        const float area_b = (bx1 - bx0) * (by1 - by0);
        const float uni = area_a + area_b - inter;
        if (uni <= 0.0f) return 0.0f;
        return inter / uni;
    }

    inline bool invert_5x5(const float S[5][5], float S_inv[5][5]) {
        double m[5][10];
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) m[i][j] = static_cast<double>(S[i][j]);
            for (int j = 5; j < 10; ++j) m[i][j] = (i == (j - 5)) ? 1.0 : 0.0;
        }
        for (int col = 0; col < 5; ++col) {
            int piv = col;
            double best = std::abs(m[piv][col]);
            for (int r = col + 1; r < 5; ++r) {
                double v = std::abs(m[r][col]);
                if (v > best) { best = v; piv = r; }
            }
            if (best < 1e-12) return false;
            if (piv != col) {
                for (int c = 0; c < 10; ++c) std::swap(m[piv][c], m[col][c]);
            }
            double div = m[col][col];
            for (int c = 0; c < 10; ++c) m[col][c] /= div;
            for (int r = 0; r < 5; ++r) if (r != col) {
                double f = m[r][col];
                for (int c = 0; c < 10; ++c) m[r][c] -= f * m[col][c];
            }
        }
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j)
                S_inv[i][j] = static_cast<float>(m[i][j + 5]);
        return true;
    }

    static constexpr int HUNGARIAN_MAX_DIM = 32;

    inline std::vector<int> hungarian_min(const std::vector<std::vector<float>>& cost) {
        const int n = static_cast<int>(cost.size());
        const int m = n ? static_cast<int>(cost[0].size()) : 0;
        const int dim = std::max(n, m);
        const double INF = 1e18;

        if (dim > HUNGARIAN_MAX_DIM) return std::vector<int>(n, -1);

        double a[HUNGARIAN_MAX_DIM + 1][HUNGARIAN_MAX_DIM + 1] = {};
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) a[i][j] = cost[i - 1][j - 1];
            for (int j = m + 1; j <= dim; ++j) a[i][j] = 1.0;
        }

        double u[HUNGARIAN_MAX_DIM + 1] = {}, v[HUNGARIAN_MAX_DIM + 1] = {};
        int p[HUNGARIAN_MAX_DIM + 1] = {}, way[HUNGARIAN_MAX_DIM + 1] = {};
        for (int i = 1; i <= dim; ++i) {
            p[0] = i;
            int j0 = 0;
            double minv[HUNGARIAN_MAX_DIM + 1];
            char used[HUNGARIAN_MAX_DIM + 1] = {};
            for (int j = 0; j <= dim; ++j) minv[j] = INF;
            do {
                used[j0] = true;
                int i0 = p[j0], j1 = 0;
                double delta = INF;
                for (int j = 1; j <= dim; ++j) if (!used[j]) {
                    double cur = a[i0][j] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
                for (int j = 0; j <= dim; ++j) {
                    if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                    else { minv[j] -= delta; }
                }
                j0 = j1;
            } while (p[j0] != 0);
            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0 != 0);
        }
        std::vector<int> assign(n, -1);
        for (int j = 1; j <= m; ++j) {
            if (p[j] >= 1 && p[j] <= n) assign[p[j] - 1] = j - 1;
        }
        return assign;
    }
}

class KalmanSimple {
public:
    KalmanSimple(int terminate_set, int generate_threshold,
        float vx_noise, float vy_noise, float w_noise, float h_noise, float r_std)
        : terminate_count_(terminate_set),
        terminate_init_(terminate_set),
        generate_threshold_(generate_threshold),
        hit_streak_(1),
        id_(next_id_++),
        has_z_(false),
        last_prob_(0.0f),
        vx_noise_(vx_noise),
        vy_noise_(vy_noise),
        w_noise_(w_noise),
        h_noise_(h_noise),
        r_var_(r_std * r_std) {
        for (int i = 0; i < 7; ++i) {
            X_post_[i] = 0.0f;
            X_pri_[i] = 0.0f;
            for (int j = 0; j < 7; ++j) {
                P_post_[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }

    static void reset_next_id(int v = 0) { next_id_ = v; }
    void set_prob(float p) { last_prob_ = p; }
    float prob() const { return last_prob_; }
    bool is_confirmed() const { return hit_streak_ >= generate_threshold_; }

    void init_from_meas(const float z[5]) {
        for (int i = 0; i < 5; ++i) X_post_[i] = z[i];
        X_post_[5] = 0.0f; X_post_[6] = 0.0f;
    }

    void predict(float dt = 1.f) {
        float A[7][7] = { 0 };
        for (int i = 0; i < 7; ++i) A[i][i] = 1.f;
        A[1][5] = dt;
        A[2][6] = dt;

        for (int i = 0; i < 7; ++i) {
            float s = 0.f;
            for (int j = 0; j < 7; ++j) s += A[i][j] * X_post_[j];
            X_pri_[i] = s;
        }
        float AP[7][7];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j) {
                float s = 0.f;
                for (int k = 0; k < 7; ++k) s += A[i][k] * P_post_[k][j];
                AP[i][j] = s;
            }
        const float q_diag[7] = {
            0.f,         // 类别不变
            0.f,         // 中心点 X 过程噪声
            0.f,         // 中心点 Y 过程噪声
            w_noise_,    // 宽度过程噪声
            h_noise_,    // 高度过程噪声
            vx_noise_,   // X 速度过程噪声
            vy_noise_    // Y 速度过程噪声
        };

        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j) {
                float s = 0.f;
                for (int k = 0; k < 7; ++k) s += AP[i][k] * A[j][k];
                P_pri_[i][j] = s + ((i == j) ? q_diag[i] : 0.f);
            }
    }

    bool update(const float z[5]) {
        hit_streak_++;

        float y[5];
        for (int i = 0; i < 5; ++i) y[i] = z[i] - X_pri_[i];

        float S[5][5];
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j)
                S[i][j] = P_pri_[i][j] + ((i == j) ? r_var_ : 0.0f);

        float S_inv[5][5];
        if (!detail::invert_5x5(S, S_inv)) return update_unmatched();

        float K[7][5];
        for (int r = 0; r < 7; ++r) {
            for (int c = 0; c < 5; ++c) {
                double acc = 0.0;
                for (int k = 0; k < 5; ++k) acc += static_cast<double>(P_pri_[r][k]) * S_inv[k][c];
                K[r][c] = static_cast<float>(acc);
            }
        }
        for (int r = 0; r < 7; ++r) {
            double delta = 0.0;
            for (int c = 0; c < 5; ++c) delta += static_cast<double>(K[r][c]) * y[c];
            X_post_[r] = X_pri_[r] + static_cast<float>(delta);
        }
        float HP[5][7];
        for (int r = 0; r < 5; ++r)
            for (int c = 0; c < 7; ++c) HP[r][c] = P_pri_[r][c];
        float KHP[7][7];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j) {
                double acc = 0.0;
                for (int k = 0; k < 5; ++k) acc += static_cast<double>(K[i][k]) * HP[k][j];
                KHP[i][j] = static_cast<float>(acc);
            }
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j) P_post_[i][j] = P_pri_[i][j] - KHP[i][j];

        // 输出卡尔曼滤波后的平滑估计 X_post_
        for (int i = 0; i < 5; ++i) last_z6_[i] = X_post_[i];
        last_z6_[5] = static_cast<float>(id_);
        has_z_ = true;
        terminate_count_ = terminate_init_;
        return true;
    }

    bool update_unmatched() {
        if (terminate_count_ == 1) return false;
        --terminate_count_;
        for (int i = 0; i < 7; ++i) X_post_[i] = X_pri_[i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j) P_post_[i][j] = P_pri_[i][j];
        // 未匹配时用预测值作为估计
        for (int k = 0; k < 5; ++k) last_z6_[k] = X_post_[k];
        last_z6_[5] = static_cast<float>(id_);
        has_z_ = true;
        return true;
    }

    const float* state_prior_5() const { return X_pri_; }
    bool has_z() const { return has_z_; }
    const float* last_z6() const { return last_z6_; }
    int id() const { return id_; }

private:
    int terminate_count_;
    int terminate_init_;
    int generate_threshold_;
    int hit_streak_;
    int id_;
    bool has_z_;
    float X_post_[7], X_pri_[7];
    float P_post_[7][7], P_pri_[7][7];
    float last_z6_[6];
    float last_prob_;
    float vx_noise_;
    float vy_noise_;
    float w_noise_;
    float h_noise_;
    float r_var_;
    inline static int next_id_ = 0;
};

class KalmanP {
public:
    void init(int GENERATE = 2, int TERMINATE = 5,
        float vx_noise = 1.0f, float vy_noise = 1.0f,
        float w_noise = 0.01f, float h_noise = 0.01f,
        float r_std = 1.0f)
    {
        generate_set_ = GENERATE;
        terminate_set_ = TERMINATE;
        r_std_ = r_std;
        vx_noise_ = vx_noise;
        vy_noise_ = vy_noise;
        w_noise_ = w_noise;
        h_noise_ = h_noise;
    }

    void reset(bool reset_id = true) {
        tracks_.clear();
        to_remove_.clear();
        if (reset_id) KalmanSimple::reset_next_id(0);
    }

    void predict(const std::vector<DetectionObject>& dets, std::vector<DetectionObject>& outputs) {
        for (auto& t : tracks_) t->predict();

        meas_list_.clear();
        meas_prob_.clear();
        for (const auto& d : dets) {
            float z[5]; detail::meas_from_object(d, z);
            meas_list_.push_back({ z[0], z[1], z[2], z[3], z[4] });
            meas_prob_.push_back(d.prob);
        }

        state_list5_.clear();
        for (const auto& t : tracks_) {
            const float* s = t->state_prior_5();
            state_list5_.push_back({ s[0], s[1], s[2], s[3], s[4] });
        }

        cost_.resize(state_list5_.size());
        for (auto& row : cost_) {
            row.assign(meas_list_.size(), 1.0f);
        }

        for (size_t i = 0; i < state_list5_.size(); ++i) {
            for (size_t j = 0; j < meas_list_.size(); ++j) {
                float iou = detail::iou_from_meas(state_list5_[i].data(), meas_list_[j].data());
                float cost_iou = 1.0f - iou;
                float dx = state_list5_[i][1] - meas_list_[j][1];
                float dy = state_list5_[i][2] - meas_list_[j][2];
                float w_avg = (state_list5_[i][3] + meas_list_[j][3]) * 0.5f;
                float h_avg = (state_list5_[i][4] + meas_list_[j][4]) * 0.5f;
                float dist_norm_x = std::abs(dx) / (w_avg * 3.0f);
                float dist_norm_y = std::abs(dy) / (h_avg * 3.0f);
                float cost_dist = std::min((dist_norm_x + dist_norm_y) * 0.5f, 1.0f);
                float area_pred = state_list5_[i][3] * state_list5_[i][4];
                float area_meas = meas_list_[j][3] * meas_list_[j][4];
                float cost_shape = std::abs(area_pred - area_meas) / std::max(area_pred, area_meas);
                cost_[i][j] = 0.1f * cost_iou + 0.7f * cost_dist + 0.2f * cost_shape;
            }
        }

        assign_ = detail::hungarian_min(cost_);
        mea_used_.assign(meas_list_.size(), false);
        for (size_t i = 0; i < tracks_.size(); ++i) {
            int j = (i < assign_.size() ? assign_[i] : -1);
            bool matched = (j >= 0 && j < static_cast<int>(meas_list_.size()));
            if (matched) {
                if (cost_[i][j] < 1.0f) {
                    float z[5]; for (int k = 0; k < 5; ++k) z[k] = meas_list_[j][k];
                    tracks_[i]->set_prob(meas_prob_[j]);
                    if (!tracks_[i]->update(z)) {
                        to_remove_.push_back(static_cast<int>(i));
                    }
                    mea_used_[j] = true;
                    continue;
                }
            }
            if (!tracks_[i]->update_unmatched()) {
                to_remove_.push_back(static_cast<int>(i));
            }
        }

        if (!to_remove_.empty()) {
            kept_.clear();
            kept_.reserve(tracks_.size());
            for (size_t i = 0; i < tracks_.size(); ++i) {
                if (std::find(to_remove_.begin(), to_remove_.end(), static_cast<int>(i)) == to_remove_.end())
                    kept_.emplace_back(std::move(tracks_[i]));
            }
            tracks_.swap(kept_);
            to_remove_.clear();
        }

        for (size_t j = 0; j < mea_used_.size(); ++j) {
            if (!mea_used_[j]) {
                auto t = std::make_unique<KalmanSimple>(terminate_set_, generate_set_,
                    vx_noise_, vy_noise_, w_noise_, h_noise_, r_std_);
                float z[5]; for (int k = 0; k < 5; ++k) z[k] = meas_list_[j][k];
                t->init_from_meas(z);
                t->set_prob(meas_prob_[j]);
                tracks_.emplace_back(std::move(t));
            }
        }

        outputs.clear();
        for (const auto& t : tracks_) {
            if (!t->has_z() || !t->is_confirmed()) continue;

            const float* z6 = t->last_z6();
            float x0, y0, x1, y1;
            detail::meas_to_bbox(z6, x0, y0, x1, y1);

            DetectionObject o;
            o.label = static_cast<int>(z6[0]);
            o.bbox.x = x0;
            o.bbox.y = y0;
            o.bbox.width = x1 - x0;
            o.bbox.height = y1 - y0;
            o.prob = t->prob();
            o.track_id = static_cast<int>(z6[5]);
            outputs.push_back(o);
        }
    }
private:
    int terminate_set_;
    int generate_set_;
    float vx_noise_;
    float vy_noise_;
    float w_noise_;
    float h_noise_;
    float r_std_;
    std::vector<std::unique_ptr<KalmanSimple>> tracks_;
    std::vector<int> to_remove_;
    // 复用缓冲区，避免每帧堆分配
    std::vector<std::array<float, 5>> meas_list_;
    std::vector<float> meas_prob_;
    std::vector<std::array<float, 5>> state_list5_;
    std::vector<std::vector<float>> cost_;
    std::vector<int> assign_;
    std::vector<char> mea_used_;
    std::vector<std::unique_ptr<KalmanSimple>> kept_;
};
