#include "curve.h"
#include "model_weights.h"
#include <cmath>
#include <iostream>
#include <numeric>

// ==================== 插值函数实现 ====================

double easeOutSine(double x) {
    return std::sin((x * 3.14159265358979323846) / 2.0);
}

double easeInOutQuad(double x) {
    return x < 0.5 ? 2.0 * x * x : 1.0 - std::pow(-2.0 * x + 2.0, 2.0) / 2.0;
}

double linear(double x) {
    return x;
}

std::vector<std::pair<double, double>> tweenPoints(
    const std::vector<std::pair<double, double>>& points,
    double (*tween)(double),
    int targetPoints
) {
    if (points.empty() || targetPoints <= 0) return {};
    if (points.size() == 1) return std::vector<std::pair<double, double>>(targetPoints, points[0]);

    std::vector<std::pair<double, double>> res;
    res.reserve(targetPoints);

    for (int i = 0; i < targetPoints; i++) {
        double t = static_cast<double>(i) / (targetPoints - 1);
        double tweened_t = tween(t);
        int index = static_cast<int>(tweened_t * (points.size() - 1));
        index = std::max(0, std::min(index, static_cast<int>(points.size() - 1)));
        res.push_back(points[index]);
    }
    return res;
}

// ==================== DenseLayer ====================

void DenseLayer::init(int input_size, int output_size, bool use_relu) {
    m_input_size = input_size;
    m_output_size = output_size;
    m_use_relu = use_relu;
    m_weights.assign(input_size, std::vector<double>(output_size, 0.0));
    m_biases.assign(output_size, 0.0);
}

void DenseLayer::forward(const double* in, double* out) const {
    for (int j = 0; j < m_output_size; ++j) {
        double sum = m_biases[j];
        for (int i = 0; i < m_input_size; ++i) {
            sum += m_weights[i][j] * in[i];
        }
        out[j] = m_use_relu ? (sum > 0.0 ? sum : 0.0) : std::tanh(sum);
    }
}

void DenseLayer::load_weights(const std::vector<std::vector<double>>& weights_data,
    const std::vector<double>& biases_data) {
    m_weights = weights_data;
    m_biases = biases_data;
}

// ==================== NeuralNetwork ====================

void NeuralNetwork::forward(const double* in, double* out) {
    // ping-pong: in → buf_a → buf_b → ... → out
    const double* src = in;
    for (int i = 0; i < m_num_layers; ++i) {
        double* dst = (i == m_num_layers - 1) ? out
            : ((i % 2 == 0) ? m_buf_a.data() : m_buf_b.data());
        m_layers[i].forward(src, dst);
        src = dst;
    }
}

void NeuralNetwork::load_embedded_weights() {
    MousePredictor::ModelWeights model_weights;
    m_num_layers = 0;

    for (size_t i = 0; i < model_weights.dense_layers.size() && m_num_layers < 3; ++i) {
        const auto& ld = model_weights.dense_layers[i];
        bool use_relu = (ld.activation == "relu");
        m_layers[m_num_layers].init(ld.input_size, ld.output_size, use_relu);
        m_layers[m_num_layers].load_weights(ld.weights, ld.biases);
        m_num_layers++;
    }
}

// ==================== 鼠标预测器实现 ====================

void MMousePredictor::init(
    int width,
    int height,
    int target_radius,
    double mouse_step_size,
    int points
) {
    this->width = width;
    this->height = height;
    this->target_radius = target_radius;
    this->mouse_step_size = mouse_step_size;
    targetPoints = points;

    m_model.load_embedded_weights();
}

std::vector<std::pair<double, double>> MMousePredictor::moveTo(double target_x, double target_y) {
    return moveTo(0.0, 0.0, target_x, target_y);
}

std::vector<std::pair<double, double>> MMousePredictor::moveTo(
    double from_x, double from_y,
    double target_x, double target_y
) {
    Vec2d start_pos(from_x, from_y);
    Vec2d target_pos(target_x, target_y);

    // draw_predicted_path 写入 m_pathBuf
    m_pathBuf.clear();
    Vec2d current_pos = start_pos;
    m_pathBuf.push_back(current_pos);
    for (int i = 0; i < 100; ++i) {
        Vec2d next_pos = predict_next_point(current_pos, target_pos);
        m_pathBuf.push_back(next_pos);
        if ((next_pos - target_pos).norm() < target_radius) break;
        current_pos = next_pos;
    }

    double clampedTarget_x = target_x - from_x;
    double clampedTarget_y = target_y - from_y;

    // convertToRelativePathCombined inline
    if (m_pathBuf.empty()) return {};

    m_pairBuf.clear();
    for (const auto& p : m_pathBuf) {
        m_pairBuf.emplace_back(p[0], p[1]);
    }

    m_tweenBuf = tweenPoints(m_pairBuf, easeOutSine, targetPoints);

    m_relativeBuf.clear();
    std::pair<double, double> origin = { 0.0, 0.0 };
    std::pair<double, double> extraNumbers = { 0.0, 0.0 };
    std::pair<double, double> totalOffset = { 0.0, 0.0 };

    bool xReached = false;
    bool yReached = false;

    for (const auto& point : m_tweenBuf) {
        std::pair<double, double> currentPoint = { point.first, point.second };
        std::pair<double, double> offset = {
            currentPoint.first - origin.first,
            currentPoint.second - origin.second
        };

        extraNumbers.first += offset.first;
        extraNumbers.second += offset.second;

        double outputX = 0.0;
        double outputY = 0.0;
        bool hasOutput = false;

        if (!xReached && std::abs(extraNumbers.first) >= 1.0) {
            double roundedValue = std::round(extraNumbers.first);
            if (std::abs(totalOffset.first + roundedValue) <= std::abs(clampedTarget_x)) {
                outputX = roundedValue;
                totalOffset.first += roundedValue;
                extraNumbers.first -= roundedValue;
                hasOutput = true;
            } else {
                double remaining = clampedTarget_x - totalOffset.first;
                if (std::abs(remaining) > 1e-6) {
                    outputX = remaining;
                    totalOffset.first = clampedTarget_x;
                    hasOutput = true;
                }
                extraNumbers.first = 0.0;
                xReached = true;
            }
        }

        if (!yReached && std::abs(extraNumbers.second) >= 1.0) {
            double roundedValue = std::round(extraNumbers.second);
            if (std::abs(totalOffset.second + roundedValue) <= std::abs(clampedTarget_y)) {
                outputY = roundedValue;
                totalOffset.second += roundedValue;
                extraNumbers.second -= roundedValue;
                hasOutput = true;
            } else {
                double remaining = clampedTarget_y - totalOffset.second;
                if (std::abs(remaining) > 1e-6) {
                    outputY = remaining;
                    totalOffset.second = clampedTarget_y;
                    hasOutput = true;
                }
                extraNumbers.second = 0.0;
                yReached = true;
            }
        }

        if (hasOutput) {
            m_relativeBuf.push_back({ outputX, outputY });
        }

        origin = currentPoint;

        if (xReached && yReached) break;
        if (std::abs(totalOffset.first - clampedTarget_x) < 1e-6 &&
            std::abs(totalOffset.second - clampedTarget_y) < 1e-6) break;
    }

    double xError = 0.0;
    double yError = 0.0;
    if (std::abs(totalOffset.first - clampedTarget_x) > 1e-6) {
        xError = clampedTarget_x - totalOffset.first;
    }
    if (std::abs(totalOffset.second - clampedTarget_y) > 1e-6) {
        yError = clampedTarget_y - totalOffset.second;
    }
    if (std::abs(xError) > 1e-6 || std::abs(yError) > 1e-6) {
        m_relativeBuf.push_back({ xError, yError });
    }

    return m_relativeBuf;
}

// ==================== 绝对路径输出 ====================

std::vector<std::pair<double, double>> MMousePredictor::moveToAbsolute(double target_x, double target_y) {
    return moveToAbsolute(0.0, 0.0, target_x, target_y);
}

std::vector<std::pair<double, double>> MMousePredictor::moveToAbsolute(
    double from_x, double from_y,
    double target_x, double target_y
) {
    Vec2d start_pos(from_x, from_y);
    Vec2d target_pos(target_x, target_y);

    std::vector<Vec2d> path_vec = draw_predicted_path(start_pos, target_pos);

    std::vector<std::pair<double, double>> path;
    path.reserve(path_vec.size());
    for (const auto& p : path_vec) {
        path.emplace_back(p[0], p[1]);
    }

    if (path.empty()) {
        return { {target_x, target_y} };
    }

    if (targetPoints > (int)path.size()) {
        return tweenPoints(path, easeOutSine, targetPoints);
    }

    return path;
}

// ==================== 合并式相对路径转换 (X,Y) ====================

std::vector<std::pair<double, double>> MMousePredictor::convertToRelativePathCombined(
    const std::vector<Vec2d>& absolute_path,
    double clampedTarget_x,
    double clampedTarget_y
) {
    if (absolute_path.empty()) {
        return {};
    }

    std::vector<std::pair<double, double>> path;
    path.reserve(absolute_path.size());
    for (const auto& p : absolute_path) {
        path.emplace_back(p[0], p[1]);
    }

    std::vector<std::pair<double, double>> tweenedCoords;
    tweenedCoords = tweenPoints(path, easeOutSine, targetPoints);

    std::vector<std::pair<double, double>> relativePath;
    std::pair<double, double> origin = { 0.0, 0.0 };
    std::pair<double, double> extraNumbers = { 0.0, 0.0 };
    std::pair<double, double> totalOffset = { 0.0, 0.0 };

    bool xReached = false;
    bool yReached = false;

    for (const auto& point : tweenedCoords) {
        std::pair<double, double> currentPoint = { point.first, point.second };
        std::pair<double, double> offset = {
            currentPoint.first - origin.first,
            currentPoint.second - origin.second
        };

        extraNumbers.first += offset.first;
        extraNumbers.second += offset.second;

        double outputX = 0.0;
        double outputY = 0.0;
        bool hasOutput = false;

        // 处理 X 方向
        if (!xReached && std::abs(extraNumbers.first) >= 1.0) {
            double roundedValue = std::round(extraNumbers.first);

            if (std::abs(totalOffset.first + roundedValue) <= std::abs(clampedTarget_x)) {
                outputX = roundedValue;
                totalOffset.first += roundedValue;
                extraNumbers.first -= roundedValue;
                hasOutput = true;
            }
            else {
                double remaining = clampedTarget_x - totalOffset.first;
                if (std::abs(remaining) > 1e-6) {
                    outputX = remaining;
                    totalOffset.first = clampedTarget_x;
                    hasOutput = true;
                }
                extraNumbers.first = 0.0;
                xReached = true;
            }
        }

        // 处理 Y 方向
        if (!yReached && std::abs(extraNumbers.second) >= 1.0) {
            double roundedValue = std::round(extraNumbers.second);

            if (std::abs(totalOffset.second + roundedValue) <= std::abs(clampedTarget_y)) {
                outputY = roundedValue;
                totalOffset.second += roundedValue;
                extraNumbers.second -= roundedValue;
                hasOutput = true;
            }
            else {
                double remaining = clampedTarget_y - totalOffset.second;
                if (std::abs(remaining) > 1e-6) {
                    outputY = remaining;
                    totalOffset.second = clampedTarget_y;
                    hasOutput = true;
                }
                extraNumbers.second = 0.0;
                yReached = true;
            }
        }

        // 合并X和Y到一个点
        if (hasOutput) {
            relativePath.push_back({ outputX, outputY });
        }

        origin = currentPoint;

        if (xReached && yReached) {
            break;
        }
        if (std::abs(totalOffset.first - clampedTarget_x) < 1e-6 &&
            std::abs(totalOffset.second - clampedTarget_y) < 1e-6) {
            break;
        }
    }

    // 处理剩余误差（合并到一个点）
    double xError = 0.0;
    double yError = 0.0;

    if (std::abs(totalOffset.first - clampedTarget_x) > 1e-6) {
        xError = clampedTarget_x - totalOffset.first;
    }

    if (std::abs(totalOffset.second - clampedTarget_y) > 1e-6) {
        yError = clampedTarget_y - totalOffset.second;
    }

    if (std::abs(xError) > 1e-6 || std::abs(yError) > 1e-6) {
        relativePath.push_back({ xError, yError });
    }

    return relativePath;
}

// ==================== 辅助函数 ====================

Vec2d MMousePredictor::predict_next_point(
    const Vec2d& curr_pos,
    const Vec2d& target_pos
) {
    Vec2d distance_to_target = target_pos - curr_pos;
    double distance_magnitude = distance_to_target.norm();

    Vec2d normalized_direction = Vec2d::Zero();
    if (distance_magnitude > 1e-9) {
        normalized_direction = distance_to_target / distance_magnitude;
    }

    // 栈上特征向量，零堆分配
    double features[4] = {
        distance_to_target[0] / width,
        distance_to_target[1] / height,
        normalized_direction[0],
        normalized_direction[1]
    };

    double prediction[2];
    m_model.forward(features, prediction);

    Vec2d pred_vec(prediction[0], prediction[1]);
    Vec2d next_pos = curr_pos + pred_vec * mouse_step_size;
    return next_pos;
}

std::vector<Vec2d> MMousePredictor::draw_predicted_path(
    const Vec2d& start_pos,
    const Vec2d& target_pos
) {
    std::vector<Vec2d> path;
    Vec2d current_pos = start_pos;
    path.push_back(current_pos);

    for (int i = 0; i < 100; ++i) {
        Vec2d next_pos = predict_next_point(current_pos, target_pos);
        path.push_back(next_pos);

        if ((next_pos - target_pos).norm() < target_radius) {
            break;
        }

        current_pos = next_pos;
    }

    return path;
}
