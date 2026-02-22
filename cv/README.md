# CV Limb Health Monitor

Screen-capture-based detection of a game’s **limb health HUD** (thin green/yellow/red outline). Tracks head, chest, torso, left/right arm, and left/right leg, and can drive **Arduino LEDs** per limb (green = off, yellow = blink, red = on).

---

## Capabilities

- **Thin HUD outline detection** — Finds the colored wireframe outline (green/yellow/red) via HSV and per-limb template masks; does not rely on filled body regions.
- **Per-limb state** — Each limb gets a state: `green`, `yellow`, `red`, `missing`, `occluded`, or `unknown`, with temporal smoothing and red latch.
- **Screen capture (ROI)** — Captures a configurable rectangle of the screen; optional **interactive ROI mode** to position/size the box with the keyboard.
- **Limb templates** — Templates are generated once from a reference image (`player_reference.png`) and stored in `limb_templates.pkl`; the detector samples only the outline within each limb region.
- **Arduino output** — Sends limb index + state over serial (9600 baud). One pin per limb: **green = off**, **yellow = blink**, **red = on**; other states = off. Only sends when a limb’s state changes.
- **Debug images** — Optional first-frame saves: `debug_captured_area.png`, `debug_mask.png`, `debug_thin_outline.png`, `debug_contours.png`.
- **Dashboard (optional)** — `cvopen.py` can log frame-level CSV; `dashboard_survival.py` provides a Streamlit dashboard (timeline, transitions, KPIs). See [DASHBOARD_QUICKSTART.md](DASHBOARD_QUICKSTART.md).

---

## Requirements

- Python 3.x
- Dependencies in `requirements.txt`: `numpy`, `opencv-python`, `mss`, `streamlit`, `pandas`, `plotly`
- **Arduino (optional):** `pyserial` and the `sketch.ino` in the repo root (4 pins: left arm, right arm, left leg, right leg)

---

## Install

```bash
cd cv
pip install -r requirements.txt
```

If you use Arduino:

```bash
pip install pyserial
```

---

## Setup (first-time)

### 1. Set the capture region (ROI)

Edit `new_opencv.py` and set:

- `ROI_X`, `ROI_Y`, `ROI_WIDTH`, `ROI_HEIGHT` to the screen rectangle that contains the player HUD.

Or run with **interactive ROI mode** to position the box with the keyboard:

- In `new_opencv.py` set `ADJUST_ROI_MODE = True`.
- Run `python new_opencv.py`, move/resize with arrow keys and W/S/A/D/+/- until the box fits the model, then press **Space** to save. Values are printed and can be written to `roi_settings.txt`.

### 2. Generate limb templates

Templates must match the ROI size used at runtime.

1. With the game visible and the HUD in **green** (healthy), capture the **same ROI** (crop a screenshot to that rectangle) and save it as **`player_reference.png`** in the `cv` folder.
2. In `new_find_model.py`, set `TARGET_WIDTH` and `TARGET_HEIGHT` to your `ROI_WIDTH` and `ROI_HEIGHT` from step 1.
3. Run:

   ```bash
   python new_find_model.py
   ```

   This creates `limb_templates.pkl` and `template_*_mask.png` / `template_limbs_overlay.png`. The main script loads templates from `limb_templates.pkl`.

### 3. Arduino (optional)

- Upload `sketch.ino` (in the repo root) to the board.
- In `new_opencv.py`, set the serial port (e.g. `arduino_serial = serial.Serial(port='COM3', ...)` on Windows or `'/dev/cu.usbmodem101'` on macOS).
- Pins in the sketch: **8** = left arm, **9** = right arm, **10** = left leg, **11** = right leg.

---

## Run the monitor

```bash
cd cv
python new_opencv.py
```

- Prints per-limb state every loop (green/yellow/red/missing/occluded/unknown).
- If Arduino is connected, it sends updates only when a limb’s state changes.
- With `SAVE_DEBUG_IMAGE = True`, the first frame writes the debug images into `cv/`.

Stop with **Ctrl+C**.

---

## Main options (in code)

| File / variable        | Effect |
|------------------------|--------|
| `ADJUST_ROI_MODE`      | `True` = start in interactive ROI adjust (arrow keys, Space to save). |
| `ROI_X`, `ROI_Y`, `ROI_WIDTH`, `ROI_HEIGHT` | Capture rectangle on screen. |
| `SAVE_DEBUG_IMAGE`     | Save debug PNGs on first frame. |
| `DEBUG_MODE`           | Extra per-limb debug prints when state is not green/yellow/red. |

---

## Files overview

| File | Role |
|------|------|
| `new_opencv.py` | Main detector: screen capture, outline + template detection, Arduino serial, ROI adjust. |
| `new_find_model.py` | One-off template generator from `player_reference.png` → `limb_templates.pkl`. |
| `dashboard_survival.py` | Streamlit dashboard over `cv/logs/limb_run_*.csv` (see DASHBOARD_QUICKSTART.md). |
| `requirements.txt` | Python dependencies. |
| `sketch.ino` (repo root) | Arduino sketch: one pin per limb, green=off / yellow=blink / red=on. |

---

## Troubleshooting

- **“Failed to load templates”** — Run `python new_find_model.py` and ensure `player_reference.png` exists and `TARGET_WIDTH`/`TARGET_HEIGHT` match your ROI.
- **All limbs “unknown”** — Check `debug_captured_area.png` and `debug_thin_outline.png`; ensure the ROI contains the HUD and the outline is visible. Adjust ROI or re-capture `player_reference.png` and regenerate templates.
- **Arduino not reacting** — Confirm port and baud (9600), and that the sketch is uploaded and the Python script is using the same port.
