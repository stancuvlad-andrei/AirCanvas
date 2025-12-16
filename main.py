import cv2
import numpy as np
import random
import math
import os
from collections import deque

# Attempt to import TensorFlow
TF_AVAILABLE = False
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    print("[WARNING] TensorFlow not installed. AI features disabled.")

# ------------------------------ CONFIGURATION ----------------------------------
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAP_WIDTH = 1280
CAP_HEIGHT = 720

# Colors
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (0, 255, 255)   # Yellow
]

UI_COLORS = {
    'MAGIC': (100, 100, 255),
    'SPRAY': (150, 0, 150),
    'PREDICT': (0, 255, 255),
    'ERASE': (0, 0, 0),
    'TEXT': (255, 255, 255),
    'HEADER_BG': (50, 50, 50)
}

# Drawing & Detection Settings
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 100
CIRCULARITY_THRESHOLD = 0.65
WRITING_MIN_AREA = 200     # Sensitivity for pen detection
READING_MIN_AREA = 1500    # Sensitivity for AI reading

# HSV Color Ranges (Blue Marker)
HSV_LOWER = np.array([90, 70, 70])
HSV_UPPER = np.array([130, 255, 255])
KERNEL = np.ones((5, 5), np.uint8)

# ------------------------------ AI ENGINE ----------------------------------
class DigitRecognizer:
    def __init__(self, model_path='my_super_model.keras'):
        self.model = None
        self.trained = False
        self.label_map = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
            20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
            30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
            36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
        }
        
        if TF_AVAILABLE and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                self.trained = True
                print(f"[INFO] Model loaded successfully: {model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
        else:
            print(f"[WARNING] Model file '{model_path}' not found.")

    def preprocess(self, roi):
        # 1. Pad to Square
        h, w = roi.shape
        if h > w:
            pad = (h - w) // 2
            roi = cv2.copyMakeBorder(roi, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        elif w > h:
            pad = (w - h) // 2
            roi = cv2.copyMakeBorder(roi, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

        # 2. Resize to EMNIST standard
        roi = cv2.resize(roi, (28, 28))
        
        # 3. Rotate & Flip
        roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
        roi = cv2.flip(roi, 1)
        
        # 4. Normalize
        roi = roi.astype('float32') / 255.0
        roi = roi.reshape(1, 28, 28, 1)
        return roi

    def predict(self, img_canvas):
        if not self.trained: return "Model Error"

        gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        
        # Low threshold to capture dark colors
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > READING_MIN_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                valid_contours.append((x, cnt))
        
        if not valid_contours: return ""
        valid_contours.sort(key=lambda item: item[0])
        
        full_text = ""
        for x, cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            pad = 25
            y1, y2 = max(0, y-pad), min(WINDOW_HEIGHT, y+h+pad)
            x1, x2 = max(0, x-pad), min(WINDOW_WIDTH, x+w+pad)
            
            roi = thresh[y1:y2, x1:x2]
            if roi.size == 0: continue

            try:
                input_data = self.preprocess(roi)
                prediction = self.model.predict(input_data, verbose=0)
                char = self.label_map.get(np.argmax(prediction), '?')
                full_text += char
            except:
                continue
        
        # Save to file
        if full_text:
            try:
                with open("rezultate.txt", "a") as f:
                    f.write(f"Detectat: {full_text}\n")
            except Exception as e:
                print(f"[ERROR] Could not save to file: {e}")
                
        return full_text

# ------------------------------ HELPER FUNCTIONS ----------------------------------
def get_rainbow_color(hue):
    c = cv2.cvtColor(np.uint8([[[hue % 180, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return (int(c[0]), int(c[1]), int(c[2]))

def draw_spray(img, x, y, color):
    for _ in range(50):
        off_x, off_y = random.randint(-20, 20), random.randint(-20, 20)
        if off_x**2 + off_y**2 <= 400:
            cv2.circle(img, (x + off_x, y + off_y), 1, color, -1)

def detect_and_draw_shape(img_canvas, is_rainbow):
    gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return
    
    cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    mean_val = cv2.mean(img_canvas, mask=mask)
    detected_color = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))
    
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    center = (x + w//2, y + h//2)
    
    # Clear old shape
    cv2.rectangle(img_canvas, (x-25, y-25), (x+w+25, y+h+25), (0,0,0), -1)
    
    corners = len(approx)
    if corners == 3:
        p1, p2, p3 = (x + w//2, y), (x, y + h), (x + w, y + h)
        if is_rainbow:
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            cv2.line(img_canvas, p1, p2, colors[0], 10)
            cv2.line(img_canvas, p2, p3, colors[1], 10)
            cv2.line(img_canvas, p3, p1, colors[2], 10)
        else:
            cv2.polylines(img_canvas, [np.array([p1, p2, p3], np.int32)], True, detected_color, 10)
            
    elif 4 <= corners <= 6:
        if is_rainbow:
            cv2.rectangle(img_canvas, (x, y), (x+w, y+h), (0, 255, 255), 10)
        else:
            cv2.rectangle(img_canvas, (x, y), (x+w, y+h), detected_color, 10)
            
    else: # Circle
        radius = max(w, h) // 2
        if is_rainbow:
            cv2.circle(img_canvas, center, radius, (255, 0, 255), 10)
        else:
            cv2.circle(img_canvas, center, radius, detected_color, 10)

# ------------------------------ INITIALIZATION ----------------------------------
cap = cv2.VideoCapture(0)
cap.set(3, CAP_WIDTH)
cap.set(4, CAP_HEIGHT)

recognizer = DigitRecognizer()
points_buffer = deque(maxlen=7)
canvas = None

# State Variables
curr_color = COLORS[0]
spray_mode = False
rainbow_mode = False
shape_lock = False
hue_counter = 0
last_btn_idx = -1
xp, yp = 0, 0
last_prediction = ""

print("System Ready. Screens: [Teacher Control] + [Projector View]")

# ------------------------------ MAIN LOOP ----------------------------------
while True:
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    if canvas is None: canvas = np.zeros_like(frame)
    h, w = frame.shape[:2]
    header_h = max(80, int(h * 0.15))

    # Rainbow Logic
    hue_counter = (hue_counter + 1) % 180
    rainbow_color = get_rainbow_color(hue_counter)
    draw_color = rainbow_color if rainbow_mode else curr_color

    # Detection Logic
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.erode(mask, KERNEL, iterations=1)
    mask = cv2.dilate(mask, KERNEL, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    obj_detected, pen_down = False, False
    curr_x, curr_y = 0, 0

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        
        if area > WRITING_MIN_AREA:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            curr_x, curr_y = int(x), int(y)
            obj_detected = True
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: perimeter = 1.0
            circularity = (4 * math.pi * area) / (perimeter ** 2)
            pen_down = circularity > CIRCULARITY_THRESHOLD

    # Smoothing & Cursor
    final_x, final_y = 0, 0
    if obj_detected:
        points_buffer.append((curr_x, curr_y))
        final_x = sum(p[0] for p in points_buffer) // len(points_buffer)
        final_y = sum(p[1] for p in points_buffer) // len(points_buffer)
        
        if pen_down:
            cv2.circle(frame, (final_x, final_y), 10, draw_color, -1)
            cv2.circle(frame, (final_x, final_y), 12, (255, 255, 255), 2)
        else:
            cv2.circle(frame, (final_x, final_y), 15, draw_color, 2)
            cv2.putText(frame, "^", (final_x-5, final_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
    else:
        points_buffer.clear()

    # Interaction
    if obj_detected:
        if final_y < header_h: # Header Region
            xp, yp = 0, 0
            points_buffer.clear()
            
            btn_w = w // 10
            btn_idx = int(final_x // btn_w)
            
            if btn_idx != last_btn_idx:
                shape_lock = False
                last_btn_idx = btn_idx
            
            # Button Logic
            if final_x < btn_w: # CLEAR
                canvas[:] = 0
                last_prediction = ""
            elif final_x < btn_w * 2: curr_color = COLORS[0]; rainbow_mode = False
            elif final_x < btn_w * 3: curr_color = COLORS[1]; rainbow_mode = False
            elif final_x < btn_w * 4: curr_color = COLORS[2]; rainbow_mode = False
            elif final_x < btn_w * 5: curr_color = COLORS[3]; rainbow_mode = False
            elif final_x < btn_w * 6: rainbow_mode = True # MAGIC
            elif final_x < btn_w * 7: # SPRAY
                if not shape_lock: 
                    spray_mode = not spray_mode
                    shape_lock = True
            elif final_x < btn_w * 8: # SHAPE
                if not shape_lock:
                    detect_and_draw_shape(canvas, rainbow_mode)
                    shape_lock = True
            elif final_x < btn_w * 9: # PREDICT
                if not shape_lock:
                    res = recognizer.predict(canvas)
                    last_prediction = f"AI: {res}"
                    print(f"Pred: {res}")
                    shape_lock = True
            else: # ERASE
                curr_color = (0, 0, 0)
                rainbow_mode = False
                spray_mode = False
        
        else: # Drawing Region
            shape_lock = False
            last_btn_idx = -1
            if xp == 0 and yp == 0: xp, yp = final_x, final_y
            
            if pen_down:
                if curr_color == (0, 0, 0) and not rainbow_mode:
                    cv2.line(canvas, (xp, yp), (final_x, final_y), curr_color, ERASER_THICKNESS)
                elif spray_mode:
                    draw_spray(canvas, final_x, final_y, draw_color)
                else:
                    cv2.line(canvas, (xp, yp), (final_x, final_y), draw_color, BRUSH_THICKNESS)
            xp, yp = final_x, final_y
    else:
        xp, yp = 0, 0
        shape_lock = False

    # ------------------------------ UI COMPOSITION ----------------------------------
    # 1. Teacher Control (Frame + Header)
    cv2.rectangle(frame, (0, 0), (w, header_h), UI_COLORS['HEADER_BG'], -1)
    btn_w = w // 10
    labels = ["CLEAR", "BLUE", "GREEN", "RED", "YELLO", "MAGIC", "SPRAY", "SHAPE", "PREDICT", "ERASE"]
    
    for i in range(10):
        x1, x2 = i * btn_w, (i + 1) * btn_w
        
        btn_color = (200, 200, 200)
        if 1 <= i <= 4: btn_color = COLORS[i-1]
        elif i == 5: btn_color = rainbow_color
        elif i == 6: btn_color = UI_COLORS['MAGIC']
        elif i == 7: btn_color = UI_COLORS['SPRAY']
        elif i == 8: btn_color = UI_COLORS['PREDICT']
        elif i == 9: btn_color = UI_COLORS['ERASE']
        
        cv2.rectangle(frame, (x1+5, 5), (x2-5, header_h-5), btn_color, -1)
        
        selected = False
        if 1 <= i <= 4 and curr_color == COLORS[i-1] and not rainbow_mode: selected = True
        elif i == 5 and rainbow_mode: selected = True
        elif i == 6 and spray_mode: selected = True
        elif i == 9 and curr_color == (0, 0, 0): selected = True
        
        if selected:
            cv2.rectangle(frame, (x1+5, 5), (x2-5, header_h-5), (255, 255, 255), 3)

        text_color = UI_COLORS['TEXT']
        if i in [0, 5, 8]: text_color = (0, 0, 0)
        font_scale = 0.35 if i == 8 else 0.4
        cv2.putText(frame, labels[i], (x1+5, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

    if last_prediction:
        cv2.rectangle(frame, (w//2 - 200, h - 70), (w//2 + 200, h - 20), (0, 0, 0), -1)
        cv2.putText(frame, last_prediction, (w//2 - 180, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, UI_COLORS['PREDICT'], 2)

    # 2. Projector View
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    frame_final = cv2.bitwise_or(cv2.bitwise_and(frame, img_inv), canvas)
    
    cv2.imshow("Teacher Control", frame_final)
    cv2.imshow("Projector View", canvas) # Clean view for students

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord('s'): cv2.imwrite("MyArt.jpg", canvas)

cap.release()
cv2.destroyAllWindows()