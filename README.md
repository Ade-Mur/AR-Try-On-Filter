# AR Try-On Filter

A real-time AR try-on filter application using MediaPipe Face Mesh and OpenCV.

## Setup

This project uses **Python 3.12** because MediaPipe doesn't support Python 3.13 yet.

### Install Dependencies

```bash
py -3.12 -m pip install -r requirements.txt
```

### Usage

#### ⭐ BEST: Complete AR Overlay (Glasses + Hat):
```bash
py -3.12 complete_ar_overlay.py
```

#### Overlay real glasses image (glasses.png):
```bash
py -3.12 real_glasses_overlay.py
```

#### Run with realistic sunglasses (with tinted lenses and effects):
```bash
py -3.12 realistic_sunglasses.py
```

#### Run the basic AR glasses try-on:
```bash
py -3.12 ar_tryon.py
```

#### Run with multiple wireframe glasses styles:
```bash
py -3.12 advanced_glasses.py
```

#### Or use the example script:
```bash
py -3.12 example_usage.py
```

## Sunglasses Styles

The realistic sunglasses version includes:
1. **Classic** - Classic oval frames with dark tint (Press '1')
2. **Aviator** - Teardrop aviator style with blue tint (Press '2')
3. **Wayfarer** - Square wayfarer style with medium tint (Press '3')
4. **Round** - Round John Lennon-style frames (Press '4')

**Features:**
- Realistic lens tinting and blur effects
- Lens highlights and reflections
- Dark frame rendering
- Automatic eye tracking
- Smooth rendering

## Correct Import Pattern

```python
import cv2
import mediapipe as mp

# Access solutions like this:
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

# Create a face mesh instance:
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

## Project Structure

- `ar_tryon.py` - Main application with face mesh detection
- `example_usage.py` - Example implementation
- `requirements.txt` - Python dependencies
- `pyrightconfig.json` - Linter configuration
- `.vscode/settings.json` - VS Code/Cursor settings

## Features

- **Complete AR System** - Glasses AND Hat support! Add hat.png for full overlay ⭐
- **Real glasses image overlay** - Use actual glasses.png image on your face
- **Hat overlay** - Add hat.png for head accessories
- Real-time face detection and landmark tracking
- AR glasses overlay using MediaPipe face mesh
- Multiple glasses styles with interactive switching
- Webcam integration with mirror mode
- Automatic positioning based on face landmarks
- Alpha transparency blending for realistic appearance

## Controls

### Complete AR Overlay (complete_ar_overlay.py) ⭐ BEST
- Press `q` to quit the application
- Automatically overlays glasses.png AND hat.png (if hat image exists)
- Both items track your head movement in real-time

### Real Glasses Image Overlay (real_glasses_overlay.py)
- Press `q` to quit the application
- Automatically tracks your face and overlays glasses.png

### Realistic Sunglasses Mode (realistic_sunglasses.py)
- Press `1` for Classic sunglasses
- Press `2` for Aviator sunglasses  
- Press `3` for Wayfarer sunglasses
- Press `4` for Round sunglasses
- Press `q` to quit the application

### Basic Mode (ar_tryon.py)
- Press `q` to quit the application

### Advanced Wireframe Mode (advanced_glasses.py)
- Press `1` for Rounded glasses
- Press `2` for Square glasses
- Press `3` for Aviator glasses
- Press `q` to quit the application

## Note on Linting

If you see linting errors about `face_mesh` attribute access, this is a known issue with the type checker. The code works correctly at runtime. You can safely ignore these warnings or add `# type: ignore` comments if needed.

