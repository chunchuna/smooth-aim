#pragma once
#include <cmath>
#include <algorithm>

// Kp + Ki aim controller with built-in distance decay
// - Kp: proportional gain (controls aim speed)
// - Ki: integral gain (tracks moving targets, compensates steady-state error)
// - Distance decay: output = Kp * error * decay(|error|)
//   where decay approaches 0 as error approaches 0, preventing overshoot/jitter
//
// The decay function: speed = Kp * error * (1 - exp(-|error| / decayRadius))
//   - Far from target: factor ~1, full speed
//   - Near target: factor ~0, natural deceleration
//   - decayRadius controls the transition zone size

class AimController {
public:
    double kp = 0.7;          // proportional gain
    double ki = 0.0008;       // integral gain
    double decayRadius = 15.0; // distance at which decay is ~63% (pixels)
    double maxIntegral = 80.0; // anti-windup clamp for integral
    double maxOutput = 150.0;  // clamp final output

    void reset() {
        m_integral = 0.0;
        m_prevError = 0.0;
        m_firstFrame = true;
    }

    double update(double error) {
        // Integral accumulation with anti-windup
        m_integral += error;
        m_integral = std::clamp(m_integral, -maxIntegral, maxIntegral);

        // Distance-based decay: smoothly decelerate near target
        double absErr = std::abs(error);
        double decayFactor = 1.0 - std::exp(-absErr / std::max(decayRadius, 0.1));

        // P term with decay (auto-decelerate near target)
        double pOut = kp * error * decayFactor;

        // I term (tracks moving targets)
        double iOut = ki * m_integral;

        double output = pOut + iOut;

        // Clamp output
        output = std::clamp(output, -maxOutput, maxOutput);

        m_prevError = error;
        m_firstFrame = false;
        return output;
    }

private:
    double m_integral = 0.0;
    double m_prevError = 0.0;
    bool m_firstFrame = true;
};
