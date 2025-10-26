"""
Complete AR Overlay - Glasses and Hats
Supports both glasses.png and hat.png images
"""
import cv2
import mediapipe as mp
import numpy as np


def initialize_face_mesh():
    """Initialize MediaPipe face mesh solution"""
    mp_face_mesh = mp.solutions.face_mesh
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return face_mesh


def get_eye_landmarks(face_landmarks, image_shape):
    """Extract eye landmarks from face mesh"""
    h, w = image_shape[:2]
    
    left_eye = []
    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173]
    for idx in left_eye_indices:
        if idx < len(face_landmarks.landmark):
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            left_eye.append((x, y))
    
    right_eye = []
    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 398]
    for idx in right_eye_indices:
        if idx < len(face_landmarks.landmark):
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            right_eye.append((x, y))
    
    if left_eye and right_eye:
        left_center = np.mean(left_eye, axis=0).astype(int)
        right_center = np.mean(right_eye, axis=0).astype(int)
        eye_distance = np.linalg.norm(right_center - left_center)
        eye_center = ((left_center + right_center) / 2).astype(int)
        angle = np.arctan2(right_center[1] - left_center[1], 
                          right_center[0] - left_center[0]) * 180 / np.pi
        return eye_center, eye_distance, angle
    
    return None, None, None


def get_forehead_landmarks(face_landmarks, image_shape):
    """Extract forehead landmarks for hat positioning"""
    h, w = image_shape[:2]
    
    # Forehead points (top of face mesh)
    forehead_indices = [10, 151, 9, 10, 151, 337]
    forehead_points = []
    
    for idx in forehead_indices:
        if idx < len(face_landmarks.landmark):
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            forehead_points.append((x, y))
    
    if forehead_points:
        # Get the topmost point
        forehead_top = min(forehead_points, key=lambda p: p[1])
        
        # Get left and right temple points for width calculation
        left_temple_idx = 234  # Left temple
        right_temple_idx = 454  # Right temple
        
        if left_temple_idx < len(face_landmarks.landmark) and right_temple_idx < len(face_landmarks.landmark):
            left_temple = face_landmarks.landmark[left_temple_idx]
            right_temple = face_landmarks.landmark[right_temple_idx]
            
            left_temple_pos = (int(left_temple.x * w), int(left_temple.y * h))
            right_temple_pos = (int(right_temple.x * w), int(right_temple.y * h))
            
            forehead_center_x = (left_temple_pos[0] + right_temple_pos[0]) // 2
            forehead_center_y = forehead_top[1]
            
            # Calculate head width
            head_width = np.linalg.norm(np.array(right_temple_pos) - np.array(left_temple_pos))
            
            return (forehead_center_x, forehead_center_y), head_width
    
    return None, None


def overlay_image(frame, overlay_img, center, width, angle=0, vertical_offset=0):
    """Overlay an image on the frame at specified position and size"""
    if center is None or width is None:
        return frame
    
    h, w = frame.shape[:2]
    gh, gw = overlay_img.shape[:2]
    
    # Calculate scale based on width
    scale = width / gw
    
    # Resize overlay
    new_width = int(gw * scale)
    new_height = int(gh * scale)
    resized = cv2.resize(overlay_img, (new_width, new_height), 
                        interpolation=cv2.INTER_LINEAR)
    
    # Rotate if needed
    if abs(angle) > 1:
        M = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), angle, 1)
        resized = cv2.warpAffine(resized, M, (new_width, new_height))
    
    # Calculate position
    x_offset = int(center[0] - new_width // 2)
    y_offset = int(center[1] - new_height // 2) + vertical_offset
    
    # Create region of interest
    y1 = max(0, y_offset)
    y2 = min(h, y_offset + new_height)
    x1 = max(0, x_offset)
    x2 = min(w, x_offset + new_width)
    
    if y2 <= y1 or x2 <= x1:
        return frame
    
    # Calculate overlay region
    gx1 = max(0, -x_offset)
    gx2 = min(new_width, new_width + w - x2)
    gy1 = max(0, -y_offset)
    gy2 = min(new_height, new_height + h - y2)
    
    # Handle alpha channel
    if len(resized.shape) == 3 and resized.shape[2] == 4:
        # Has alpha channel
        overlay_roi = resized[gy1:gy2, gx1:gx2]
        alpha = overlay_roi[:, :, 3:4] / 255.0
        overlay_rgb = overlay_roi[:, :, :3]
        frame_roi = frame[y1:y2, x1:x2]
        
        blended = (alpha * overlay_rgb + (1 - alpha) * frame_roi).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended
    elif len(resized.shape) == 3:
        # No alpha channel - simple overlay
        overlay_roi = resized[gy1:gy2, gx1:gx2]
        frame[y1:y2, x1:x2] = cv2.addWeighted(
            frame[y1:y2, x1:x2], 0.5, 
            overlay_roi, 0.5, 0
        )
    
    return frame


def main():
    """Main function with glasses and hat overlay"""
    # Load images
    glasses_img = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)
    hat_img = cv2.imread('hat.png', cv2.IMREAD_UNCHANGED)
    
    if glasses_img is None:
        print("Warning: Could not load glasses.png")
        print("Continuing without glasses overlay...")
    
    if hat_img is None:
        print("Warning: Could not load hat.png")
        print("Continuing without hat overlay...")
    
    print("=" * 60)
    print("Complete AR Overlay - Glasses + Hat")
    print("=" * 60)
    if glasses_img is not None:
        print(f"✓ Loaded glasses ({glasses_img.shape[1]}x{glasses_img.shape[0]})")
    if hat_img is not None:
        print(f"✓ Loaded hat ({hat_img.shape[1]}x{hat_img.shape[0]})")
    print()
    print("Controls:")
    print("- Press 'q' to quit")
    print("=" * 60)
    
    face_mesh = initialize_face_mesh()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get eye position for glasses
                if glasses_img is not None:
                    eye_center, eye_distance, angle = get_eye_landmarks(face_landmarks, frame.shape)
                    if eye_center is not None and eye_distance is not None:
                        frame = overlay_image(frame, glasses_img, eye_center, 
                                               eye_distance * 2.5, angle, 
                                               -int(eye_distance * 0.25))
                
                # Get forehead position for hat
                if hat_img is not None:
                    hat_center, head_width = get_forehead_landmarks(face_landmarks, frame.shape)
                    if hat_center is not None and head_width is not None:
                        # Position hat above forehead
                        # Calculate vertical offset to place hat above head
                        eye_center, eye_distance, _ = get_eye_landmarks(face_landmarks, frame.shape)
                        if eye_center is not None:
                            vertical_offset = -int(eye_distance * 1.2)  # Above the head
                        else:
                            vertical_offset = -100
                        
                        frame = overlay_image(frame, hat_img, hat_center, 
                                               head_width * 1.3, 0, vertical_offset)
        
        # Display info
        info_text = ""
        if glasses_img is not None:
            info_text += "Glasses "
        if hat_img is not None:
            info_text += "Hat "
        
        cv2.putText(frame, f"AR Overlay: {info_text}| Press 'q' to quit",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Complete AR Overlay - Glasses + Hat', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Thank you for using Complete AR Overlay!")


if __name__ == "__main__":
    main()

