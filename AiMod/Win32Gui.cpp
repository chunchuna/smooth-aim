#include "Win32Gui.h"
#include <sstream>
#include <iomanip>

static const wchar_t* CLASS_NAME = L"AiModPanel";

LRESULT CALLBACK Win32GuiPanel::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    Win32GuiPanel* self = nullptr;
    if (msg == WM_NCCREATE) {
        auto cs = reinterpret_cast<CREATESTRUCT*>(lParam);
        self = reinterpret_cast<Win32GuiPanel*>(cs->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
    } else {
        self = reinterpret_cast<Win32GuiPanel*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    }

    if (!self) return DefWindowProc(hwnd, msg, wParam, lParam);

    switch (msg) {
    case WM_HSCROLL:
        self->OnHScroll((HWND)lParam);
        return 0;
    case WM_VSCROLL:
        self->OnVScroll(wParam);
        return 0;
    case WM_MOUSEWHEEL:
        self->OnMouseWheel(GET_WHEEL_DELTA_WPARAM(wParam));
        return 0;
    case WM_SIZE:
        self->UpdateScrollBar();
        return 0;
    case WM_COMMAND:
        self->OnCommand(wParam);
        return 0;
    case WM_CLOSE:
        self->m_running = false;
        if (self->onQuit) self->onQuit();
        return 0;
    case WM_DESTROY:
        return 0;
    default:
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }
}

bool Win32GuiPanel::Create(HINSTANCE hInst) {
    m_hInst = hInst;

    INITCOMMONCONTROLSEX icc = { sizeof(icc), ICC_BAR_CLASSES | ICC_STANDARD_CLASSES };
    InitCommonControlsEx(&icc);

    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = CLASS_NAME;
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    RegisterClassExW(&wc);

    m_hwnd = CreateWindowExW(
        0, CLASS_NAME, L"AiMod Control Panel",
        (WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX) | WS_VSCROLL,
        CW_USEDEFAULT, CW_USEDEFAULT, 500, 800,
        nullptr, nullptr, hInst, this
    );

    if (!m_hwnd) return false;

    m_font = CreateFontW(14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");

    CreateControls();
    ShowWindow(m_hwnd, SW_SHOW);
    UpdateWindow(m_hwnd);
    return true;
}

void Win32GuiPanel::Destroy() {
    if (m_hwnd) { DestroyWindow(m_hwnd); m_hwnd = nullptr; }
    if (m_font) { DeleteObject(m_font); m_font = nullptr; }
    UnregisterClassW(CLASS_NAME, m_hInst);
}

static HWND MakeLabel(HWND parent, HFONT font, const char* text, int x, int y, int w, int h, int id = 0) {
    HWND hw = CreateWindowA("STATIC", text, WS_CHILD | WS_VISIBLE | SS_LEFT,
        x, y, w, h, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
    SendMessage(hw, WM_SETFONT, (WPARAM)font, TRUE);
    return hw;
}

static HWND MakeSlider(HWND parent, int x, int y, int w, int h, int id, int minVal, int maxVal, int curVal) {
    HWND hs = CreateWindowA(TRACKBAR_CLASSA, "", WS_CHILD | WS_VISIBLE | TBS_HORZ | TBS_AUTOTICKS,
        x, y, w, h, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
    SendMessage(hs, TBM_SETRANGE, TRUE, MAKELPARAM(minVal, maxVal));
    SendMessage(hs, TBM_SETPOS, TRUE, curVal);
    return hs;
}

static HWND MakeCombo(HWND parent, HFONT font, int x, int y, int w, int h, int id,
                       const char** items, int count, int sel) {
    HWND hc = CreateWindowA("COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
        x, y, w, h, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
    SendMessage(hc, WM_SETFONT, (WPARAM)font, TRUE);
    for (int i = 0; i < count; i++) {
        SendMessageA(hc, CB_ADDSTRING, 0, (LPARAM)items[i]);
    }
    SendMessage(hc, CB_SETCURSEL, sel, 0);
    return hc;
}

static HWND MakeCheck(HWND parent, HFONT font, const char* text, int x, int y, int w, int h, int id, bool checked) {
    HWND hc = CreateWindowA("BUTTON", text, WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
        x, y, w, h, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
    SendMessage(hc, WM_SETFONT, (WPARAM)font, TRUE);
    SendMessage(hc, BM_SETCHECK, checked ? BST_CHECKED : BST_UNCHECKED, 0);
    return hc;
}

static HWND MakeBtn(HWND parent, HFONT font, const char* text, int x, int y, int w, int h, int id) {
    HWND hb = CreateWindowA("BUTTON", text, WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
        x, y, w, h, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
    SendMessage(hb, WM_SETFONT, (WPARAM)font, TRUE);
    return hb;
}

void Win32GuiPanel::CreateControls() {
    int x = 10, y = 10;
    int labelW = 90, sliderW = 230, valW = 80;
    int rowH = 28;
    int comboH = 200;

    // Model selection
    MakeLabel(m_hwnd, m_font, "Model:", x, y + 3, 50, 20);
    CreateWindowA("COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
        x + 55, y, 280, comboH, m_hwnd, (HMENU)(INT_PTR)ID_COMBO_MODEL, nullptr, nullptr);
    HWND hModelCombo = GetDlgItem(m_hwnd, ID_COMBO_MODEL);
    SendMessage(hModelCombo, WM_SETFONT, (WPARAM)m_font, TRUE);
    MakeBtn(m_hwnd, m_font, "Load Model", x + 340, y, 100, 24, ID_BTN_LOAD_MODEL);
    y += rowH + 4;

    // YOLO Type
    static const char* yoloTypes[] = { "YOLOX","YOLOv5","YOLOv7","YOLOv8","YOLOv10","YOLOv11","YOLOv12","YOLOv26" };
    MakeLabel(m_hwnd, m_font, "YOLO Type:", x, y + 3, 70, 20);
    MakeCombo(m_hwnd, m_font, x + 75, y, 150, comboH, ID_COMBO_YOLOTYPE, yoloTypes, 8, valYoloType);
    y += rowH + 8;

    // Aim settings
    MakeLabel(m_hwnd, m_font, "--- Aim Settings ---", x, y, 200, 18);
    y += 20;
    MakeCheck(m_hwnd, m_font, "Aim Enabled", x, y, 120, 20, ID_CHECK_AIM, valAimEnabled != 0);
    MakeCheck(m_hwnd, m_font, "Preview", x + 140, y, 100, 20, ID_CHECK_PREVIEW, valPreview != 0);
    y += rowH;

    static const char* aimKeys[] = { "Right Mouse","Left Mouse","X1 (Side)","X2 (Side)","Shift","Alt","CapsLock" };
    MakeLabel(m_hwnd, m_font, "Aim Key:", x, y + 3, 60, 20);
    MakeCombo(m_hwnd, m_font, x + 65, y, 150, comboH, ID_COMBO_AIMKEY, aimKeys, 7, valAimKey);
    y += rowH;

    static const char* moveModes[] = { "Curve", "Direct" };
    MakeLabel(m_hwnd, m_font, "Move:", x, y + 3, 45, 20);
    MakeCombo(m_hwnd, m_font, x + 50, y, 120, comboH, ID_COMBO_MOVEMODE, moveModes, 2, valMoveMode);
    y += rowH + 8;

    // Slider helper
    int nextAutoLabelId = 2000;
    auto addSlider = [&](const char* label, int id, int minV, int maxV, int* valPtr, float scale, const char* unit) {
        int labelId = nextAutoLabelId++;
        HWND hLbl = MakeLabel(m_hwnd, m_font, label, x, y + 5, labelW, 18);
        HWND hSlider = MakeSlider(m_hwnd, x + labelW, y, sliderW, 25, id, minV, maxV, *valPtr);
        HWND hVal = MakeLabel(m_hwnd, m_font, "", x + labelW + sliderW + 5, y + 5, valW, 18, labelId);

        SliderInfo si;
        si.id = id; si.labelId = labelId; si.valuePtr = valPtr;
        si.scale = scale; si.unit = unit; si.label = label;
        si.hSlider = hSlider; si.hLabel = hLbl; si.hValueLabel = hVal;
        m_sliders.push_back(si);
        UpdateSliderLabel(m_sliders.back());
        y += rowH;
    };

    addSlider("Smooth:",    ID_SLIDER_SMOOTH,  1, 100, &valSmooth,  0.1f, "");
    addSlider("FOV:",       ID_SLIDER_FOV,     0, 500, &valFov,     1.0f, "px");
    addSlider("HeadOff:",   ID_SLIDER_HEADOFF, -50, 50, &valHeadOff, 1.0f, "px");
    y += 8;

    MakeLabel(m_hwnd, m_font, "--- PID X ---", x, y, 200, 18);
    y += 20;
    addSlider("Kp X:",      ID_SLIDER_KPX,   0, 1000, &valKpX,   0.1f, "");
    addSlider("Kd X:",      ID_SLIDER_KDX,   0, 1000, &valKdX,   0.1f, "");
    addSlider("Pred X:",    ID_SLIDER_PREDX,  0, 500, &valPredX, 0.01f, "");
    addSlider("Rate X:",    ID_SLIDER_RATEX,  0, 100, &valRateX, 0.01f, "");
    y += 8;

    MakeLabel(m_hwnd, m_font, "--- PID Y ---", x, y, 200, 18);
    y += 20;
    addSlider("Kp Y:",      ID_SLIDER_KPY,   0, 1000, &valKpY,   0.1f, "");
    addSlider("Kd Y:",      ID_SLIDER_KDY,   0, 1000, &valKdY,   0.1f, "");
    addSlider("Pred Y:",    ID_SLIDER_PREDY,  0, 500, &valPredY, 0.01f, "");
    addSlider("Rate Y:",    ID_SLIDER_RATEY,  0, 100, &valRateY, 0.01f, "");
    y += 8;

    MakeLabel(m_hwnd, m_font, "--- Detection ---", x, y, 200, 18);
    y += 20;
    addSlider("Conf:",      ID_SLIDER_CONF,   0, 100, &valConf,  0.01f, "");
    addSlider("NMS:",       ID_SLIDER_NMS,    0, 100, &valNms,   0.01f, "");
    y += 12;

    // === Recoil Section ===
    MakeLabel(m_hwnd, m_font, "--- Recoil Control ---", x, y, 200, 18);
    y += 20;
    MakeCheck(m_hwnd, m_font, "Recoil Enabled", x, y, 130, 20, ID_CHECK_RECOIL_ENABLED, valRecoilEnabled != 0);
    MakeCheck(m_hwnd, m_font, "Aim Only", x + 140, y, 90, 20, ID_CHECK_RECOIL_AIMONLY, valRecoilAimOnly != 0);
    y += rowH;

    MakeLabel(m_hwnd, m_font, "Recoil Key:", x, y + 3, 70, 20);
    CreateWindowA("COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
        x + 75, y, 185, comboH, m_hwnd, (HMENU)(INT_PTR)ID_COMBO_RECOIL_KEY, nullptr, nullptr);
    {
        HWND hRcKeyCombo = GetDlgItem(m_hwnd, ID_COMBO_RECOIL_KEY);
        SendMessage(hRcKeyCombo, WM_SETFONT, (WPARAM)m_font, TRUE);
    }
    y += rowH;

    MakeLabel(m_hwnd, m_font, "Weapon:", x, y + 3, 55, 20);
    // Weapon combo populated externally
    CreateWindowA("COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
        x + 60, y, 200, comboH, m_hwnd, (HMENU)(INT_PTR)ID_COMBO_RECOIL_WEAPON, nullptr, nullptr);
    HWND hRcCombo = GetDlgItem(m_hwnd, ID_COMBO_RECOIL_WEAPON);
    SendMessage(hRcCombo, WM_SETFONT, (WPARAM)m_font, TRUE);
    y += rowH;

    addSlider("Strength:", ID_SLIDER_RECOIL_STRENGTH, 1, 30, &valRecoilStrength, 0.1f, "");
    addSlider("Smooth:",   ID_SLIDER_RECOIL_SMOOTH,   1, 16, &valRecoilSmooth,   1.0f, "");
    addSlider("Hold ms:",  ID_SLIDER_RECOIL_HOLDMS,   0, 500, &valRecoilHoldMs,  1.0f, "ms");
    addSlider("TimeOff:",  ID_SLIDER_RECOIL_TIMEOFF, -100, 100, &valRecoilTimeOff, 1.0f, "ms");
    {
        HWND hChk = CreateWindowA("BUTTON", "Scan Transfer (X-only on spray)",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
            x, y, 260, 20, m_hwnd, (HMENU)(INT_PTR)ID_CHECK_SCAN_TRANSFER, nullptr, nullptr);
        SendMessage(hChk, WM_SETFONT, (WPARAM)m_font, TRUE);
        SendMessage(hChk, BM_SETCHECK, valScanTransfer ? BST_CHECKED : BST_UNCHECKED, 0);
        y += rowH;
    }
    y += 12;

    // === Class Filter (placeholder, rebuilt on model load) ===
    MakeLabel(m_hwnd, m_font, "--- Class Filter ---", x, y, 200, 18);
    y += 20;
    MakeBtn(m_hwnd, m_font, "All", x, y, 50, 22, ID_BTN_CLASS_ALL);
    MakeBtn(m_hwnd, m_font, "None", x + 55, y, 50, 22, ID_BTN_CLASS_NONE);
    y += 26;
    m_controlsEndY = y; // class checkboxes are added dynamically below this
    y += 12;

    // === Profile Section ===
    MakeLabel(m_hwnd, m_font, "--- Profile ---", x, y, 200, 18);
    y += 20;
    MakeLabel(m_hwnd, m_font, "Profile:", x, y + 3, 50, 20);
    CreateWindowA("COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
        x + 55, y, 200, comboH, m_hwnd, (HMENU)(INT_PTR)ID_COMBO_PROFILE, nullptr, nullptr);
    {
        HWND hProfCombo = GetDlgItem(m_hwnd, ID_COMBO_PROFILE);
        SendMessage(hProfCombo, WM_SETFONT, (WPARAM)m_font, TRUE);
    }
    y += rowH;
    MakeBtn(m_hwnd, m_font, "Load", x, y, 55, 24, ID_BTN_PROFILE_LOAD);
    MakeBtn(m_hwnd, m_font, "Save", x + 60, y, 55, 24, ID_BTN_PROFILE_SAVE);
    MakeBtn(m_hwnd, m_font, "Delete", x + 120, y, 55, 24, ID_BTN_PROFILE_DELETE);
    y += 28;
    MakeLabel(m_hwnd, m_font, "New:", x, y + 3, 30, 20);
    CreateWindowA("EDIT", "", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
        x + 35, y, 140, 22, m_hwnd, (HMENU)(INT_PTR)ID_EDIT_PROFILE_NAME, nullptr, nullptr);
    {
        HWND hEdit = GetDlgItem(m_hwnd, ID_EDIT_PROFILE_NAME);
        SendMessage(hEdit, WM_SETFONT, (WPARAM)m_font, TRUE);
    }
    MakeBtn(m_hwnd, m_font, "Create", x + 180, y, 55, 22, ID_BTN_PROFILE_NEW);
    y += 28;

    // === Buttons ===
    MakeBtn(m_hwnd, m_font, "Save Config", x, y, 100, 28, ID_BTN_SAVE);
    MakeBtn(m_hwnd, m_font, "Load Config", x + 110, y, 100, 28, ID_BTN_LOAD_CONFIG);
    y += 36;

    // === Status labels ===
    MakeLabel(m_hwnd, m_font, "Status: Ready", x, y, 460, 18, ID_LABEL_STATUS);
    y += 20;
    MakeLabel(m_hwnd, m_font, "", x, y, 460, 18, ID_LABEL_FPS);
    y += 20;
    MakeLabel(m_hwnd, m_font, "Device: Unknown", x, y, 460, 18, ID_LABEL_DEVICE);
    y += 24;

    m_totalContentH = y;
    UpdateScrollBar();
}

void Win32GuiPanel::UpdateSliderLabel(SliderInfo& si) {
    float displayVal = (*si.valuePtr) * si.scale;
    char buf[64];
    if (si.scale >= 1.0f)
        snprintf(buf, sizeof(buf), "%d %s", (int)displayVal, si.unit);
    else
        snprintf(buf, sizeof(buf), "%.2f %s", displayVal, si.unit);
    SetWindowTextA(si.hValueLabel, buf);
}

void Win32GuiPanel::OnHScroll(HWND hCtrl) {
    for (auto& si : m_sliders) {
        if (si.hSlider == hCtrl) {
            *si.valuePtr = (int)SendMessage(hCtrl, TBM_GETPOS, 0, 0);
            UpdateSliderLabel(si);
            return;
        }
    }
}

void Win32GuiPanel::OnCommand(WPARAM wParam) {
    int id = LOWORD(wParam);
    int notif = HIWORD(wParam);

    switch (id) {
    case ID_BTN_LOAD_MODEL:
        if (onLoadModel) onLoadModel();
        break;
    case ID_BTN_SAVE:
        if (onSaveConfig) onSaveConfig();
        break;
    case ID_BTN_LOAD_CONFIG:
        if (onLoadConfig) onLoadConfig();
        break;
    case ID_COMBO_MODEL:
        if (notif == CBN_SELCHANGE) {
            valModelIdx = (int)SendMessage(GetDlgItem(m_hwnd, ID_COMBO_MODEL), CB_GETCURSEL, 0, 0);
        }
        break;
    case ID_COMBO_YOLOTYPE:
        if (notif == CBN_SELCHANGE) {
            valYoloType = (int)SendMessage(GetDlgItem(m_hwnd, ID_COMBO_YOLOTYPE), CB_GETCURSEL, 0, 0);
        }
        break;
    case ID_COMBO_AIMKEY:
        if (notif == CBN_SELCHANGE) {
            valAimKey = (int)SendMessage(GetDlgItem(m_hwnd, ID_COMBO_AIMKEY), CB_GETCURSEL, 0, 0);
        }
        break;
    case ID_COMBO_MOVEMODE:
        if (notif == CBN_SELCHANGE) {
            valMoveMode = (int)SendMessage(GetDlgItem(m_hwnd, ID_COMBO_MOVEMODE), CB_GETCURSEL, 0, 0);
        }
        break;
    case ID_CHECK_AIM:
        valAimEnabled = (SendMessage(GetDlgItem(m_hwnd, ID_CHECK_AIM), BM_GETCHECK, 0, 0) == BST_CHECKED) ? 1 : 0;
        break;
    case ID_CHECK_PREVIEW:
        valPreview = (SendMessage(GetDlgItem(m_hwnd, ID_CHECK_PREVIEW), BM_GETCHECK, 0, 0) == BST_CHECKED) ? 1 : 0;
        break;
    case ID_CHECK_RECOIL_ENABLED:
        valRecoilEnabled = (SendMessage(GetDlgItem(m_hwnd, ID_CHECK_RECOIL_ENABLED), BM_GETCHECK, 0, 0) == BST_CHECKED) ? 1 : 0;
        break;
    case ID_CHECK_RECOIL_AIMONLY:
        valRecoilAimOnly = (SendMessage(GetDlgItem(m_hwnd, ID_CHECK_RECOIL_AIMONLY), BM_GETCHECK, 0, 0) == BST_CHECKED) ? 1 : 0;
        break;
    case ID_CHECK_SCAN_TRANSFER:
        valScanTransfer = (SendMessage(GetDlgItem(m_hwnd, ID_CHECK_SCAN_TRANSFER), BM_GETCHECK, 0, 0) == BST_CHECKED) ? 1 : 0;
        break;
    case ID_COMBO_RECOIL_KEY:
        if (notif == CBN_SELCHANGE) {
            valRecoilKey = (int)SendMessage(GetDlgItem(m_hwnd, ID_COMBO_RECOIL_KEY), CB_GETCURSEL, 0, 0);
        }
        break;
    case ID_COMBO_RECOIL_WEAPON:
        if (notif == CBN_SELCHANGE) {
            valRecoilWeapon = (int)SendMessage(GetDlgItem(m_hwnd, ID_COMBO_RECOIL_WEAPON), CB_GETCURSEL, 0, 0);
        }
        break;
    case ID_BTN_CLASS_ALL:
        for (auto& cc : m_classChecks)
            SendMessage(cc.hCheck, BM_SETCHECK, BST_CHECKED, 0);
        break;
    case ID_BTN_CLASS_NONE:
        for (auto& cc : m_classChecks)
            SendMessage(cc.hCheck, BM_SETCHECK, BST_UNCHECKED, 0);
        break;
    case ID_BTN_PROFILE_LOAD:
    {
        HWND hCombo = GetDlgItem(m_hwnd, ID_COMBO_PROFILE);
        int sel = (int)SendMessage(hCombo, CB_GETCURSEL, 0, 0);
        if (sel >= 0) {
            char buf[256] = {};
            SendMessageA(hCombo, CB_GETLBTEXT, sel, (LPARAM)buf);
            if (onProfileLoad) onProfileLoad(std::string(buf));
        }
        break;
    }
    case ID_BTN_PROFILE_SAVE:
    {
        HWND hCombo = GetDlgItem(m_hwnd, ID_COMBO_PROFILE);
        int sel = (int)SendMessage(hCombo, CB_GETCURSEL, 0, 0);
        if (sel >= 0) {
            char buf[256] = {};
            SendMessageA(hCombo, CB_GETLBTEXT, sel, (LPARAM)buf);
            if (onProfileSave) onProfileSave(std::string(buf));
        }
        break;
    }
    case ID_BTN_PROFILE_DELETE:
    {
        HWND hCombo = GetDlgItem(m_hwnd, ID_COMBO_PROFILE);
        int sel = (int)SendMessage(hCombo, CB_GETCURSEL, 0, 0);
        if (sel >= 0) {
            char buf[256] = {};
            SendMessageA(hCombo, CB_GETLBTEXT, sel, (LPARAM)buf);
            if (onProfileDelete) onProfileDelete(std::string(buf));
        }
        break;
    }
    case ID_BTN_PROFILE_NEW:
    {
        char buf[256] = {};
        GetWindowTextA(GetDlgItem(m_hwnd, ID_EDIT_PROFILE_NAME), buf, sizeof(buf));
        std::string name(buf);
        if (!name.empty() && onProfileSave) {
            onProfileSave(name);
            SetWindowTextA(GetDlgItem(m_hwnd, ID_EDIT_PROFILE_NAME), "");
        }
        break;
    }
    }
}

void Win32GuiPanel::SetStatusText(const std::string& text) {
    if (m_hwnd) SetWindowTextA(GetDlgItem(m_hwnd, ID_LABEL_STATUS), text.c_str());
}

void Win32GuiPanel::SetFpsText(const std::string& text) {
    if (m_hwnd) SetWindowTextA(GetDlgItem(m_hwnd, ID_LABEL_FPS), text.c_str());
}

void Win32GuiPanel::SetDeviceText(const std::string& text) {
    if (m_hwnd) SetWindowTextA(GetDlgItem(m_hwnd, ID_LABEL_DEVICE), text.c_str());
}

void Win32GuiPanel::RebuildClassFilter(const std::map<int, std::string>& classNames) {
    // Destroy old checkboxes
    for (auto& cc : m_classChecks) {
        if (cc.hCheck) DestroyWindow(cc.hCheck);
    }
    m_classChecks.clear();

    if (!m_hwnd || classNames.empty()) return;

    int x = 10;
    int y = m_controlsEndY;
    int col = 0;
    for (auto& kv : classNames) {
        char label[128];
        snprintf(label, sizeof(label), "%d: %s", kv.first, kv.second.c_str());
        int cx = x + col * 150;
        HWND hc = CreateWindowA("BUTTON", label, WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
            cx, y, 145, 18, m_hwnd, (HMENU)(INT_PTR)(ID_CHECK_CLASS_BASE + kv.first), nullptr, nullptr);
        SendMessage(hc, WM_SETFONT, (WPARAM)m_font, TRUE);
        SendMessage(hc, BM_SETCHECK, BST_CHECKED, 0); // default: all enabled
        m_classChecks.push_back({kv.first, hc});
        col++;
        if (col >= 3) { col = 0; y += 20; }
    }
}

std::set<int> Win32GuiPanel::GetEnabledClassIds() const {
    std::set<int> enabled;
    for (auto& cc : m_classChecks) {
        if (SendMessage(cc.hCheck, BM_GETCHECK, 0, 0) == BST_CHECKED) {
            enabled.insert(cc.classId);
        }
    }
    return enabled;
}

void Win32GuiPanel::UpdateScrollBar() {
    if (!m_hwnd) return;
    RECT rc;
    GetClientRect(m_hwnd, &rc);
    int clientH = rc.bottom - rc.top;

    SCROLLINFO si = {};
    si.cbSize = sizeof(si);
    si.fMask = SIF_RANGE | SIF_PAGE | SIF_POS;
    si.nMin = 0;
    si.nMax = m_totalContentH;
    si.nPage = clientH;
    si.nPos = m_scrollY;
    SetScrollInfo(m_hwnd, SB_VERT, &si, TRUE);
}

void Win32GuiPanel::OnVScroll(WPARAM wParam) {
    SCROLLINFO si = {};
    si.cbSize = sizeof(si);
    si.fMask = SIF_ALL;
    GetScrollInfo(m_hwnd, SB_VERT, &si);

    int oldPos = si.nPos;
    switch (LOWORD(wParam)) {
    case SB_LINEUP:        si.nPos -= 20; break;
    case SB_LINEDOWN:      si.nPos += 20; break;
    case SB_PAGEUP:        si.nPos -= si.nPage; break;
    case SB_PAGEDOWN:      si.nPos += si.nPage; break;
    case SB_THUMBTRACK:    si.nPos = si.nTrackPos; break;
    }

    si.nPos = max(si.nPos, 0);
    int maxScroll = max((int)(si.nMax - (int)si.nPage), 0);
    si.nPos = min(si.nPos, maxScroll);

    if (si.nPos != oldPos) {
        int delta = oldPos - si.nPos;
        m_scrollY = si.nPos;
        ScrollWindow(m_hwnd, 0, delta, nullptr, nullptr);
        SetScrollPos(m_hwnd, SB_VERT, m_scrollY, TRUE);
        UpdateWindow(m_hwnd);
    }
}

void Win32GuiPanel::OnMouseWheel(short delta) {
    int scroll = -delta / 2; // 120 units per notch, scroll ~60px per notch
    SCROLLINFO si = {};
    si.cbSize = sizeof(si);
    si.fMask = SIF_ALL;
    GetScrollInfo(m_hwnd, SB_VERT, &si);

    int oldPos = si.nPos;
    si.nPos += scroll;
    si.nPos = max(si.nPos, 0);
    int maxScroll = max((int)(si.nMax - (int)si.nPage), 0);
    si.nPos = min(si.nPos, maxScroll);

    if (si.nPos != oldPos) {
        int delta2 = oldPos - si.nPos;
        m_scrollY = si.nPos;
        ScrollWindow(m_hwnd, 0, delta2, nullptr, nullptr);
        SetScrollPos(m_hwnd, SB_VERT, m_scrollY, TRUE);
        UpdateWindow(m_hwnd);
    }
}

void Win32GuiPanel::SyncControlsFromValues() {
    if (!m_hwnd) return;

    for (auto& si : m_sliders) {
        SendMessage(si.hSlider, TBM_SETPOS, TRUE, *si.valuePtr);
        UpdateSliderLabel(si);
    }

    SendMessage(GetDlgItem(m_hwnd, ID_COMBO_MODEL), CB_SETCURSEL, valModelIdx, 0);
    SendMessage(GetDlgItem(m_hwnd, ID_COMBO_YOLOTYPE), CB_SETCURSEL, valYoloType, 0);
    SendMessage(GetDlgItem(m_hwnd, ID_COMBO_AIMKEY), CB_SETCURSEL, valAimKey, 0);
    SendMessage(GetDlgItem(m_hwnd, ID_COMBO_MOVEMODE), CB_SETCURSEL, valMoveMode, 0);
    SendMessage(GetDlgItem(m_hwnd, ID_COMBO_RECOIL_WEAPON), CB_SETCURSEL, valRecoilWeapon, 0);

    SendMessage(GetDlgItem(m_hwnd, ID_CHECK_AIM), BM_SETCHECK,
        valAimEnabled ? BST_CHECKED : BST_UNCHECKED, 0);
    SendMessage(GetDlgItem(m_hwnd, ID_CHECK_PREVIEW), BM_SETCHECK,
        valPreview ? BST_CHECKED : BST_UNCHECKED, 0);
    SendMessage(GetDlgItem(m_hwnd, ID_CHECK_RECOIL_ENABLED), BM_SETCHECK,
        valRecoilEnabled ? BST_CHECKED : BST_UNCHECKED, 0);
    SendMessage(GetDlgItem(m_hwnd, ID_CHECK_RECOIL_AIMONLY), BM_SETCHECK,
        valRecoilAimOnly ? BST_CHECKED : BST_UNCHECKED, 0);
    SendMessage(GetDlgItem(m_hwnd, ID_CHECK_SCAN_TRANSFER), BM_SETCHECK,
        valScanTransfer ? BST_CHECKED : BST_UNCHECKED, 0);
    SendMessage(GetDlgItem(m_hwnd, ID_COMBO_RECOIL_KEY), CB_SETCURSEL, valRecoilKey, 0);
}

void Win32GuiPanel::RefreshProfileList(const std::vector<std::string>& names, const std::string& current) {
    if (!m_hwnd) return;
    HWND hCombo = GetDlgItem(m_hwnd, ID_COMBO_PROFILE);
    if (!hCombo) return;
    SendMessage(hCombo, CB_RESETCONTENT, 0, 0);
    int selIdx = -1;
    for (int i = 0; i < (int)names.size(); i++) {
        SendMessageA(hCombo, CB_ADDSTRING, 0, (LPARAM)names[i].c_str());
        if (names[i] == current) selIdx = i;
    }
    if (selIdx >= 0) SendMessage(hCombo, CB_SETCURSEL, selIdx, 0);
    currentProfileName = current;
}

void Win32GuiPanel::MessageLoop() {
    MSG msg;
    while (m_running) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                m_running = false;
                if (onQuit) onQuit();
                return;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        Sleep(50);
    }
}

void Win32GuiPanel::PopulateModelCombo(const std::vector<std::string>& names, int sel) {
    HWND hCombo = GetDlgItem(m_hwnd, ID_COMBO_MODEL);
    if (!hCombo) return;
    SendMessage(hCombo, CB_RESETCONTENT, 0, 0);
    for (const auto& name : names) {
        SendMessageA(hCombo, CB_ADDSTRING, 0, (LPARAM)name.c_str());
    }
    if (sel >= 0 && sel < (int)names.size()) {
        SendMessage(hCombo, CB_SETCURSEL, sel, 0);
    }
    valModelIdx = sel;
}
