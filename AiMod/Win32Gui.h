#pragma once
#include <Windows.h>
#include <CommCtrl.h>
#pragma comment(lib, "comctl32.lib")
#include <string>
#include <vector>
#include <map>
#include <set>
#include <functional>

// Win32 native GUI control panel (replaces OpenCV trackbar GUI)

enum GuiCtrlID {
    ID_COMBO_MODEL = 1001,
    ID_COMBO_YOLOTYPE,
    ID_COMBO_AIMKEY,
    ID_COMBO_MOVEMODE,
    ID_CHECK_AIM,
    ID_CHECK_PREVIEW,
    ID_BTN_LOAD_MODEL,
    ID_BTN_SAVE,
    ID_BTN_LOAD_CONFIG,
    ID_SLIDER_SMOOTH = 1100,
    ID_SLIDER_FOV,
    ID_SLIDER_HEADOFF,
    ID_SLIDER_KPX,
    ID_SLIDER_KDX,
    ID_SLIDER_PREDX,
    ID_SLIDER_RATEX,
    ID_SLIDER_KPY,
    ID_SLIDER_KDY,
    ID_SLIDER_PREDY,
    ID_SLIDER_RATEY,
    ID_SLIDER_CONF,
    ID_SLIDER_NMS,
    // value display labels (offset by 100 from slider IDs)
    ID_LABEL_SMOOTH = 1200,
    ID_LABEL_FOV,
    ID_LABEL_HEADOFF,
    ID_LABEL_KPX,
    ID_LABEL_KDX,
    ID_LABEL_PREDX,
    ID_LABEL_RATEX,
    ID_LABEL_KPY,
    ID_LABEL_KDY,
    ID_LABEL_PREDY,
    ID_LABEL_RATEY,
    ID_LABEL_CONF,
    ID_LABEL_NMS,
    // status labels
    ID_LABEL_STATUS = 1300,
    ID_LABEL_FPS,
    ID_LABEL_DEVICE,
    // recoil controls
    ID_COMBO_RECOIL_WEAPON = 1400,
    ID_SLIDER_RECOIL_STRENGTH,
    ID_SLIDER_RECOIL_SMOOTH,
    ID_SLIDER_RECOIL_HOLDMS,
    ID_SLIDER_RECOIL_TIMEOFF,
    ID_CHECK_RECOIL_ENABLED,
    ID_CHECK_RECOIL_AIMONLY,
    ID_CHECK_SCAN_TRANSFER,
    ID_COMBO_RECOIL_KEY,
    ID_LABEL_RECOIL_STRENGTH,
    ID_LABEL_RECOIL_SMOOTH,
    ID_LABEL_RECOIL_HOLDMS,
    ID_LABEL_RECOIL_TIMEOFF,
    // class filter checkboxes start at this ID
    ID_CHECK_CLASS_BASE = 1500,
    ID_BTN_CLASS_ALL = 1600,
    ID_BTN_CLASS_NONE,
    // profile controls
    ID_COMBO_PROFILE = 1700,
    ID_BTN_PROFILE_SAVE,
    ID_BTN_PROFILE_LOAD,
    ID_BTN_PROFILE_DELETE,
    ID_BTN_PROFILE_NEW,
    ID_EDIT_PROFILE_NAME,
};

class Win32GuiPanel {
public:
    std::function<void()> onLoadModel;
    std::function<void()> onSaveConfig;
    std::function<void()> onLoadConfig;
    std::function<void()> onQuit;
    std::function<void(const std::string&)> onProfileSave;   // save current config as named profile
    std::function<void(const std::string&)> onProfileLoad;   // load a named profile
    std::function<void(const std::string&)> onProfileDelete; // delete a named profile

    void RefreshProfileList(const std::vector<std::string>& names, const std::string& current);

    bool Create(HINSTANCE hInst);
    void Destroy();
    bool IsRunning() const { return m_running; }
    HWND GetHwnd() const { return m_hwnd; }

    void SetStatusText(const std::string& text);
    void SetFpsText(const std::string& text);
    void SetDeviceText(const std::string& text);
    void PopulateModelCombo(const std::vector<std::string>& names, int sel);
    void SyncControlsFromValues();
    void MessageLoop();

    // Class filter: rebuild checkboxes when model changes
    void RebuildClassFilter(const std::map<int, std::string>& classNames);
    std::set<int> GetEnabledClassIds() const;

    // GUI value bindings (read by main loop to sync config)
    int valSmooth = 10, valFov = 300, valHeadOff = 20;
    int valKpX = 8, valKdX = 2, valPredX = 15, valRateX = 60;
    int valKpY = 8, valKdY = 2, valPredY = 15, valRateY = 60;
    int valConf = 50, valNms = 50;
    int valModelIdx = 0, valYoloType = 0, valAimKey = 0, valMoveMode = 0;
    int valAimEnabled = 1, valPreview = 1;

    // Recoil config values
    int valRecoilEnabled = 0;
    int valRecoilWeapon = 0;    // index into weapon list (last = Off)
    int valRecoilStrength = 10; // *0.1 = 1.0
    int valRecoilSmooth = 4;
    int valRecoilHoldMs = 100;
    int valRecoilTimeOff = 0;
    int valRecoilKey = 1;       // index into AimKeyOptions (1=Left Mouse)
    int valRecoilAimOnly = 0;   // 1=recoil only when aim key held
    int valScanTransfer = 0;    // 1=spray transfer (X-only aim during spray)

    // Profile
    std::string currentProfileName;

private:
    HWND m_hwnd = nullptr;
    HINSTANCE m_hInst = nullptr;
    bool m_running = true;
    HFONT m_font = nullptr;

    struct SliderInfo {
        int id;
        int labelId;
        int* valuePtr;
        float scale;
        const char* unit;
        const char* label;
        HWND hSlider;
        HWND hLabel;
        HWND hValueLabel;
    };
    std::vector<SliderInfo> m_sliders;

    // Class filter state
    struct ClassCheckInfo {
        int classId;
        HWND hCheck;
    };
    std::vector<ClassCheckInfo> m_classChecks;

    int m_controlsEndY = 0; // track Y position after main controls
    int m_totalContentH = 0; // total height of all controls
    int m_scrollY = 0;       // current scroll offset

    void CreateControls();
    void UpdateSliderLabel(SliderInfo& si);
    void OnHScroll(HWND hCtrl);
    void OnCommand(WPARAM wParam);
    void OnVScroll(WPARAM wParam);
    void OnMouseWheel(short delta);
    void UpdateScrollBar();

    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
};
