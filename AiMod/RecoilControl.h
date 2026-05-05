#pragma once
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <cmath>
#include <random>
#include <windows.h>

// Per-bullet recoil delta: dx, dy (mouse pixels), delay_ms (ms to next bullet)
struct RecoilBullet {
    int dx;
    int dy;
    int delay_ms;
};

struct WeaponPattern {
    std::string name;
    int magSize;
    std::vector<RecoilBullet> pattern;
};

// All weapon recoil patterns (CS2 + Valorant)
inline std::vector<WeaponPattern>& GetWeaponPatterns() {
    static std::vector<WeaponPattern> patterns;
    if (!patterns.empty()) return patterns;

    // CS2 Weapons
    patterns.push_back({"AK-47", 30, {
        {0,0,99},{-4,7,99},{4,19,99},{-3,29,99},{-1,31,99},
        {13,31,99},{8,28,99},{13,21,99},{-17,12,99},{-42,-3,99},
        {-21,2,99},{12,11,99},{-15,7,99},{-26,-8,99},{-3,4,99},
        {40,1,99},{19,7,99},{14,10,99},{27,0,99},{33,-10,99},
        {-21,-2,99},{7,3,99},{-7,9,99},{-8,4,99},{19,-3,99},
        {5,6,99},{-20,-1,99},{-33,-4,99},{-45,-21,99},{-14,1,80}
    }});

    patterns.push_back({"M4A4", 30, {
        {0,0,88},{2,7,88},{0,9,87},{-6,16,87},{7,21,87},
        {-9,23,87},{-5,27,87},{16,15,88},{11,13,88},{22,5,88},
        {-4,11,88},{-18,6,88},{-30,-4,88},{-24,0,88},{-25,-6,88},
        {0,4,87},{8,4,87},{-11,1,87},{-13,-2,87},{2,2,88},
        {33,-1,88},{10,6,88},{27,3,88},{10,2,88},{11,0,88},
        {-12,0,87},{6,5,87},{4,5,87},{3,1,87},{4,-1,87}
    }});

    patterns.push_back({"M4A1-S", 20, {
        {0,0,88},{1,6,88},{0,4,88},{-4,14,88},{4,18,88},
        {-6,21,88},{-4,24,88},{14,14,88},{8,12,88},{18,5,88},
        {-4,10,88},{-14,5,88},{-25,-3,88},{-19,0,88},{-22,-3,88},
        {1,3,88},{8,3,88},{-9,1,88},{-13,-2,88},{3,2,88}
    }});

    patterns.push_back({"Galil AR", 35, {
        {0,0,90},{4,4,90},{-2,5,90},{6,10,90},{12,15,90},
        {-1,21,90},{2,24,90},{6,16,90},{11,10,90},{-4,14,90},
        {-22,8,90},{-30,-3,90},{-29,-13,90},{-9,8,90},{-12,2,90},
        {-7,1,50},{0,1,90},{4,7,90},{25,7,90},{14,4,90},
        {25,-3,90},{31,-9,90},{6,3,90},{-12,3,90},{13,-1,90},
        {10,-1,90},{16,-4,90},{-9,5,90},{-32,-5,90},{-24,-3,90},
        {-15,5,90},{6,8,90},{-14,-3,90},{-24,-14,90},{-13,-1,90}
    }});

    patterns.push_back({"FAMAS", 25, {
        {0,0,88},{-4,5,88},{1,4,88},{-6,10,88},{-1,17,88},
        {0,20,88},{14,18,88},{16,12,88},{-6,12,88},{-20,8,88},
        {-16,5,88},{-13,2,88},{4,5,87},{23,4,88},{12,6,88},
        {20,-3,88},{5,0,88},{15,0,88},{3,5,80},{-4,3,88},
        {-25,-1,80},{-3,2,84},{11,0,80},{15,-7,88},{15,-10,88}
    }});

    patterns.push_back({"UMP-45", 25, {
        {0,0,90},{-1,6,90},{-4,8,90},{-2,18,90},{-4,23,90},
        {-9,23,90},{-3,26,90},{11,17,90},{-4,12,90},{9,13,90},
        {18,8,90},{15,5,90},{-1,3,90},{5,6,90},{0,6,90},
        {9,-3,90},{5,-1,90},{-12,4,90},{-19,1,85},{-1,-2,90},
        {15,-5,90},{17,-2,85},{-6,3,90},{-20,-2,90},{-3,-1,90}
    }});

    patterns.push_back({"SG 553", 30, {
        {0,0,89},{-4,9,89},{-13,15,89},{-9,25,89},{-6,29,88},
        {-8,31,88},{-7,36,80},{-20,14,80},{14,17,89},{-8,12,88},
        {-15,8,89},{-5,5,89},{6,5,88},{-8,6,89},{2,11,88},
        {-14,-6,89},{-20,-17,89},{-18,-9,88},{-8,-2,89},{41,3,88},
        {56,-5,89},{43,-1,88},{18,9,89},{14,9,88},{6,7,89},
        {21,-3,95},{29,-4,89},{-6,8,89},{-15,5,89},{-38,-5,89}
    }});

    patterns.push_back({"AUG", 30, {
        {0,0,89},{5,6,89},{0,13,89},{-5,22,89},{-7,26,88},
        {5,29,88},{9,30,80},{14,21,80},{6,15,89},{14,13,88},
        {-16,11,89},{-5,6,89},{13,0,88},{1,6,89},{-22,5,88},
        {-38,-11,89},{-31,-13,89},{-3,6,88},{-5,5,89},{-9,0,88},
        {24,1,89},{32,3,88},{15,6,89},{-5,1,88},{0,0,89},
        {0,0,88},{0,0,89},{0,0,88},{0,0,89},{0,0,88}
    }});

    // Valorant Weapons
    patterns.push_back({"Vandal", 25, {
        {0,0,102},{0,3,102},{0,3,102},{0,3,102},{0,3,102},
        {1,4,102},{-1,4,102},{1,3,102},{-1,4,102},{1,3,102},
        {3,4,101},{3,4,101},{-2,4,101},{-2,5,101},{3,4,101},
        {4,5,100},{4,5,100},{-3,5,100},{-3,4,100},{4,5,100},
        {5,5,99},{5,5,99},{-4,5,99},{-4,5,99},{5,5,99}
    }});

    patterns.push_back({"Phantom", 30, {
        {0,0,90},{0,2,90},{0,3,90},{0,3,90},{1,3,90},{-1,3,90},
        {1,4,90},{-1,4,90},{2,3,90},{-1,3,90},{1,4,90},{-2,3,90},
        {3,3,90},{2,3,90},{-2,3,90},{-3,3,90},{2,3,90},{3,2,90},
        {-3,3,90},{-2,3,90},{3,3,90},{2,3,90},{-3,2,90},{-2,3,90},
        {3,2,90},{2,2,90},{-2,2,90},{-3,2,90},{2,2,90},{1,2,90}
    }});

    patterns.push_back({"Spectre", 30, {
        {0,0,75},{0,2,75},{0,2,75},{1,2,75},{-1,3,75},
        {1,3,75},{-1,3,75},{2,2,75},{-1,2,75},{1,3,75},
        {-2,2,75},{2,2,75},{-1,2,75},{2,2,75},{-2,2,75},
        {1,2,75},{-1,2,75},{2,1,75},{-2,2,75},{1,1,75},
        {-1,2,75},{2,1,75},{-2,1,75},{1,1,75},{-1,1,75},
        {1,1,75},{-1,1,75},{1,1,75},{0,1,75},{0,1,75}
    }});

    // Off (disabled)
    patterns.push_back({"Off", 0, {}});

    return patterns;
}

// Get weapon names list for GUI combo box
inline std::vector<std::string> GetWeaponNames() {
    auto& pats = GetWeaponPatterns();
    std::vector<std::string> names;
    for (auto& w : pats) names.push_back(w.name);
    return names;
}

// Recoil control engine
class RecoilController {
public:
    // Config (set from GUI)
    bool enabled = false;
    int weaponIndex = 0;        // index into GetWeaponPatterns()
    float strength = 1.0f;      // multiplier for recoil compensation
    int smoothSteps = 4;        // lerp smoothing (1=instant, higher=smoother)
    int holdDelayMs = 100;      // ms to hold before activating recoil
    int timeOffsetMs = 0;       // timing adjustment (negative=earlier)
    int triggerKey = VK_LBUTTON; // key that triggers spray (default: left mouse)

    void Reset() {
        m_spraying = false;
        m_bulletIdx = -1;
        m_targetX = 0; m_targetY = 0;
        m_currentX = 0; m_currentY = 0;
        m_accumX = 0; m_accumY = 0;
        m_sprayStart = {};
        m_keyPressTime = {};
    }

    // Call every frame from detection loop. Returns (dx, dy) mouse move to apply.
    std::pair<int, int> Update() {
        if (!enabled) { Reset(); return {0, 0}; }

        auto& patterns = GetWeaponPatterns();
        if (weaponIndex < 0 || weaponIndex >= (int)patterns.size()) return {0, 0};
        auto& weapon = patterns[weaponIndex];
        if (weapon.magSize == 0 || weapon.pattern.empty()) return {0, 0};

        bool keyDown = (GetAsyncKeyState(triggerKey) & 0x8000) != 0;
        auto now = std::chrono::high_resolution_clock::now();

        if (keyDown && !m_wasKeyDown) {
            // Key just pressed
            m_keyPressTime = now;
            m_spraying = false;
            m_bulletIdx = -1;
            m_targetX = 0; m_targetY = 0;
            m_currentX = 0; m_currentY = 0;
            m_accumX = 0; m_accumY = 0;
        }

        if (!keyDown && m_wasKeyDown) {
            // Key released
            m_spraying = false;
            m_wasKeyDown = false;
            return {0, 0};
        }

        m_wasKeyDown = keyDown;
        if (!keyDown) return {0, 0};

        // Check hold threshold
        if (!m_spraying) {
            double heldMs = std::chrono::duration<double, std::milli>(now - m_keyPressTime).count();
            if (heldMs >= holdDelayMs) {
                m_spraying = true;
                m_sprayStart = now;
                m_bulletIdx = -1;
            } else {
                return {0, 0};
            }
        }

        // Calculate bullet index from elapsed time
        double elapsedMs = std::chrono::duration<double, std::milli>(now - m_sprayStart).count() - timeOffsetMs;
        if (elapsedMs < 0) elapsedMs = 0;

        double cumMs = 0;
        int newBulletIdx = 0;
        for (int i = 0; i < weapon.magSize && i < (int)weapon.pattern.size(); i++) {
            if (cumMs + weapon.pattern[i].delay_ms > elapsedMs) break;
            cumMs += weapon.pattern[i].delay_ms;
            newBulletIdx = i + 1;
        }
        newBulletIdx = (std::min)(newBulletIdx, weapon.magSize - 1);

        // Accumulate new bullet deltas
        while (m_bulletIdx < newBulletIdx) {
            m_bulletIdx++;
            if (m_bulletIdx < (int)weapon.pattern.size()) {
                m_targetX += weapon.pattern[m_bulletIdx].dx * strength;
                m_targetY += weapon.pattern[m_bulletIdx].dy * strength;
            }
        }

        // Lerp smoothing
        float alpha = 1.0f / (std::max)(smoothSteps, 1);
        // Add slight randomization to avoid robotic patterns
        static std::mt19937 rng(42);
        std::uniform_real_distribution<float> jitter(0.85f, 1.15f);
        alpha *= jitter(rng);
        alpha = (std::min)(alpha, 1.0f);

        m_currentX += (m_targetX - m_currentX) * alpha;
        m_currentY += (m_targetY - m_currentY) * alpha;

        int moveX = (int)std::round(m_currentX - m_accumX);
        int moveY = (int)std::round(m_currentY - m_accumY);
        m_accumX += moveX;
        m_accumY += moveY;

        return {moveX, moveY};
    }

private:
    bool m_wasKeyDown = false;
    bool m_spraying = false;
    int m_bulletIdx = -1;
    float m_targetX = 0, m_targetY = 0;
    float m_currentX = 0, m_currentY = 0;
    float m_accumX = 0, m_accumY = 0;
    std::chrono::high_resolution_clock::time_point m_sprayStart;
    std::chrono::high_resolution_clock::time_point m_keyPressTime;
};
