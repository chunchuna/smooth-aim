#ifndef SCREENCAPTURE_H
#define SCREENCAPTURE_H

#include "Obs.h"
#include "GDIScreenCapture.h"
#include "DXScreenCapture.h"
#include "WGCScreenCapture.h"
#include <vector>
#include <cstdint>
#include <memory>
#include <string>
#include <Windows.h>

class ScreenCapture {
public:
    ScreenCapture();
    ~ScreenCapture();

    bool Init(int width, int height, int mode);
    bool SetWindow(HWND hwnd);
    bool SetRegion(int x, int y, int width, int height);
    bool SetObsNetwork(const std::string& ipAddress = "0.0.0.0", int port = 7788);
    void Reset();
    std::vector<uint8_t> Capture();
    bool Capture(uint8_t* buffer, size_t bufferSize);

    int GetWidth() const { return width; }
    int GetHeight() const { return height; }
    int GetChannels() const { return channels; }

private:
    void Release();

    std::unique_ptr<GDIScreenCapture> gdiCapture;
    std::unique_ptr<DXScreenCapture> dxCapture;
    std::unique_ptr<WGCScreenCapture> wgcCapture;
    std::unique_ptr<Obs> obsCapture;

    int width;
    int height;
    int channels;
    float captureTime;
    bool initialized;
    int method; // 0=GDI, 1=DXGI, 2=WGC, 3=OBS

    // OBSÕ¯¬Á≈‰÷√
    std::string obsIpAddress;
    int obsPort;
};

#endif // SCREENCAPTURE_H
