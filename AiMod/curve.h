#ifndef CURVE_H
#define CURVE_H

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cmath>

// 热路径最大维度 = 64（网络 4→64→32→2）
static constexpr int NN_MAX_DIM = 64;

struct Vec2d {
    double x = 0.0, y = 0.0;
    Vec2d() = default;
    Vec2d(double x, double y) : x(x), y(y) {}
    double  operator[](int i) const { return i == 0 ? x : y; }
    double& operator[](int i) { return i == 0 ? x : y; }
    Vec2d operator+(const Vec2d& o) const { return { x + o.x, y + o.y }; }
    Vec2d operator-(const Vec2d& o) const { return { x - o.x, y - o.y }; }
    Vec2d operator*(double s) const { return { x * s, y * s }; }
    Vec2d operator/(double s) const { return { x / s, y / s }; }
    double norm() const { return std::sqrt(x * x + y * y); }
    static Vec2d Zero() { return { 0.0, 0.0 }; }
};

// 权重存储仍用 vector（只在初始化时分配一次）
using MatXd = std::vector<std::vector<double>>;

class DenseLayer {
public:
    DenseLayer() = default;
    void init(int input_size, int output_size, bool use_relu);
    // 零堆分配前向：从 in 读 input_size 个，写 output_size 个到 out
    void forward(const double* in, double* out) const;
    void load_weights(const std::vector<std::vector<double>>& weights_data,
        const std::vector<double>& biases_data);
    int getOutputSize() const { return m_output_size; }

private:
    int m_input_size = 0;
    int m_output_size = 0;
    bool m_use_relu = false;
    MatXd m_weights;                    // [input_size][output_size]
    std::vector<double> m_biases;       // [output_size]
};

class NeuralNetwork {
public:
    NeuralNetwork() = default;
    // 零堆分配前向：in[4] → out[2]，内部使用栈缓冲区
    void forward(const double* in, double* out);
    void load_embedded_weights();

private:
    DenseLayer m_layers[3];   // 固定3层 Dense，无 Dropout（推理不应 dropout）
    int m_num_layers = 0;
    std::array<double, NN_MAX_DIM> m_buf_a{};  // ping-pong 缓冲区
    std::array<double, NN_MAX_DIM> m_buf_b{};
};


class MMousePredictor {
public:
    MMousePredictor() = default;

    void init(int width = 800, int height = 600, int target_radius = 8, double mouse_step_size = 4.0, int points = 50);

    std::vector<std::pair<double, double>> moveTo(double target_x, double target_y);
    std::vector<std::pair<double, double>> moveTo(double from_x, double from_y, double target_x, double target_y);

    std::vector<std::pair<double, double>> moveToAbsolute(double target_x, double target_y);
    std::vector<std::pair<double, double>> moveToAbsolute(double from_x, double from_y, double target_x, double target_y);

private:
    Vec2d predict_next_point(const Vec2d& curr_pos, const Vec2d& target_pos);
    std::vector<Vec2d> draw_predicted_path(const Vec2d& start_pos, const Vec2d& target_pos);
    std::vector<std::pair<double, double>> convertToRelativePathCombined(const std::vector<Vec2d>& absolute_path, double clampedTarget_x, double clampedTarget_y);

    NeuralNetwork m_model;   // 值语义，无 raw pointer
    int targetPoints = 50;
    int width = 800;
    int height = 600;
    int target_radius = 8;
    double mouse_step_size = 4.0;

    // 复用缓冲区，避免每次moveTo堆分配
    std::vector<Vec2d> m_pathBuf;
    std::vector<std::pair<double, double>> m_pairBuf;
    std::vector<std::pair<double, double>> m_tweenBuf;
    std::vector<std::pair<double, double>> m_relativeBuf;
};


double easeOutSine(double x);
double easeInOutQuad(double x);
double linear(double x);

std::vector<std::pair<double, double>> tweenPoints(
    const std::vector<std::pair<double, double>>& points,
    double (*tween)(double),
    int targetPoints
);

#endif // CURVE_H
