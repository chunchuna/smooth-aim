#pragma once

class P_PID {
public:
    P_PID();
    ~P_PID();

    P_PID(const P_PID&) = delete;
    P_PID& operator=(const P_PID&) = delete;

    bool   init(double kp, double kd, double predict, double rate, double smooth);
    double update(double error);
    void   reset();
    void   updateParams(double kp, double kd, double predict, double rate, double smooth);

private:
    class Impl;
    Impl* m_impl;
};
