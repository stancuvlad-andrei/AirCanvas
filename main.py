import cv2
import numpy as np
import random
import math
import os
from collections import deque

# ----------------------------------- Configuration -----------------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
draw_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
curr_color = draw_colors[0] 
brush_thick = 15
eraser_thick = 100
spray_mode = False      
rainbow_mode = False    
shape_lock = False       
hue_counter = 0         
last_button_index = -1   
CIRCULARITY_THRESHOLD = 0.65 

# ----------------------------------- Color Detection -----------------------------------
my_color_lower = np.array([90, 70, 70])
my_color_upper = np.array([130, 255, 255])
kernel = np.ones((5, 5), np.uint8)

points_buffer = deque(maxlen=7)
xp, yp = 0, 0 
canvas = None 
last_prediction = "" 

# ----------------------------------- Digit Recognition -----------------------------------
class DigitRecognizer:
    def __init__(self):
        self.knn = None
        self.trained = False
        self.load_and_train()

    def load_and_train(self):
        if not os.path.exists('digits.png'):
            print("LIPSA digits.png - Descarca-l!")
            return

        img = cv2.imread('digits.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Split the image into 20x20 cells
        cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
        x = np.array(cells)
        
        train = x[:, :].reshape(-1, 400).astype(np.float32)
        k = np.arange(10)
        train_labels = np.repeat(k, 500)[:, np.newaxis]

        self.knn = cv2.ml.KNearest_create()
        self.knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
        self.trained = True
        print("KNN Antrenat (V14 - Robust Mode)!")

    def preprocess_simple(self, roi):
        h, w = roi.shape
        
        if h > w:
            pad_w = (h - w) // 2
            roi = cv2.copyMakeBorder(roi, 0, 0, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        elif w > h:
            pad_h = (w - h) // 2
            roi = cv2.copyMakeBorder(roi, pad_h, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=0)
            
        roi = cv2.resize(roi, (20, 20))
        return roi

    def predict(self, img_canvas):
        if not self.trained: return "Err"

        # 1. Binary Threshold and Find Contours
        gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0: return ""
        
        # 2. Filter Contours by Area
        valid_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 400:
                x, y, w, h = cv2.boundingRect(cnt)
                valid_contours.append((x, cnt))
        
        if not valid_contours: return ""
        
        # Sort contours from left to right
        valid_contours.sort(key=lambda item: item[0])
        
        full_number = ""
        
        for x, cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Padding
            padding = 20
            y1 = max(0, y-padding); y2 = min(720, y+h+padding)
            x1 = max(0, x-padding); x2 = min(1280, x+w+padding)
            
            roi = thresh[y1:y2, x1:x2]
            
            if roi.size == 0: continue

            try:
                processed_roi = self.preprocess_simple(roi)
                roi_float = processed_roi.reshape(-1, 400).astype(np.float32)
                ret, result, neighbours, dist = self.knn.findNearest(roi_float, k=5)
                digit = int(result[0][0])
                full_number += str(digit)
            except Exception:
                continue
                
        return full_number

brain = DigitRecognizer()

# ----------------------------------- Utils -----------------------------------
def draw_spray(img, x, y, color):
    for _ in range(50):
        off_x = random.randint(-20, 20)
        off_y = random.randint(-20, 20)
        if off_x**2 + off_y**2 <= 400:
            cv2.circle(img, (x + off_x, y + off_y), 1, color, -1)

def get_rainbow_color(hue_val):
    c = cv2.cvtColor(np.uint8([[[hue_val % 180, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return (int(c[0]), int(c[1]), int(c[2]))

# ----------------------------------- Shape Detection -----------------------------------
def detect_and_draw_shape(img_canvas, is_rainbow):
    gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return
    cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    mean_val = cv2.mean(img_canvas, mask=mask)
    detected_color = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    center_x = x + w//2
    center_y = y + h//2
    cv2.rectangle(img_canvas, (x-25, y-25), (x+w+25, y+h+25), (0,0,0), -1)
    colturi = len(approx)
    if colturi == 3:
        p1, p2, p3 = (x + w//2, y), (x, y + h), (x + w, y + h)
        if is_rainbow:
            cv2.line(img_canvas, p1, p2, (0, 0, 255), 10); cv2.line(img_canvas, p2, p3, (0, 255, 0), 10); cv2.line(img_canvas, p3, p1, (255, 0, 0), 10)
        else:
            pts = np.array([p1, p2, p3], np.int32); cv2.polylines(img_canvas, [pts], True, detected_color, 10)
    elif 4 <= colturi <= 6:
        if is_rainbow:
            cv2.line(img_canvas, (x, y), (x+w, y), (0, 0, 255), 10); cv2.line(img_canvas, (x+w, y), (x+w, y+h), (0, 255, 255), 10); cv2.line(img_canvas, (x+w, y+h), (x, y+h), (0, 255, 0), 10); cv2.line(img_canvas, (x, y+h), (x, y), (255, 0, 0), 10)
        else:
            cv2.rectangle(img_canvas, (x, y), (x + w, y + h), detected_color, 10)
    else:
        radius = max(w, h) // 2
        if is_rainbow:
            angle_step = 5
            for ang in range(0, 360, angle_step):
                start_ang = math.radians(ang); end_ang = math.radians(ang + angle_step); pt1_x = int(center_x + radius * math.cos(start_ang)); pt1_y = int(center_y + radius * math.sin(start_ang)); pt2_x = int(center_x + radius * math.cos(end_ang)); pt2_y = int(center_y + radius * math.sin(end_ang)); color_hsv = ang // 2; seg_color = get_rainbow_color(color_hsv); cv2.line(img_canvas, (pt1_x, pt1_y), (pt2_x, pt2_y), seg_color, 10)
        else:
            cv2.circle(img_canvas, (center_x, center_y), radius, detected_color, 10)

print("System Ready. Buttons: 10")

while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    if canvas is None: canvas = np.zeros((h, w, 3), dtype=np.uint8)
    header_h = max(80, int(h * 0.15))

    hue_counter = (hue_counter + 1) % 180
    rainbow_bgr = get_rainbow_color(hue_counter)
    draw_color_now = rainbow_bgr if rainbow_mode else curr_color

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, my_color_lower, my_color_upper)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    obj_detected = False
    pen_down = False 
    raw_x, raw_y = 0, 0

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area > 400:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            raw_x, raw_y = int(x), int(y)
            obj_detected = True
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: perimeter = 1.0 
            circularity = (4 * math.pi * area) / (perimeter * perimeter)
            if circularity > CIRCULARITY_THRESHOLD: pen_down = True
            else: pen_down = False

    final_x, final_y = 0, 0
    if obj_detected:
        points_buffer.append((raw_x, raw_y))
        avg_x = sum([p[0] for p in points_buffer]) // len(points_buffer)
        avg_y = sum([p[1] for p in points_buffer]) // len(points_buffer)
        final_x, final_y = avg_x, avg_y
        
        if pen_down:
            cv2.circle(frame, (final_x, final_y), 10, draw_color_now, -1)
            cv2.circle(frame, (final_x, final_y), 12, (255, 255, 255), 2)
        else:
            cv2.circle(frame, (final_x, final_y), 15, draw_color_now, 2)
            cv2.putText(frame, "^", (final_x-5, final_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color_now, 2)
    else:
        points_buffer.clear()

    # ----------------------------------- Logic button and drawing -----------------------------------
    if obj_detected:
        if final_y < header_h:
            xp, yp = 0, 0
            points_buffer.clear()
            
            btn_w = w // 10
            curr_btn_index = int(final_x // btn_w)
            
            if curr_btn_index != last_button_index:
                shape_lock = False
                last_button_index = curr_btn_index
            
            if final_x < btn_w: canvas[:,:,:] = 0; last_prediction = ""
            elif final_x < btn_w * 2: curr_color = draw_colors[0]; rainbow_mode = False 
            elif final_x < btn_w * 3: curr_color = draw_colors[1]; rainbow_mode = False
            elif final_x < btn_w * 4: curr_color = draw_colors[2]; rainbow_mode = False
            elif final_x < btn_w * 5: curr_color = draw_colors[3]; rainbow_mode = False
            elif final_x < btn_w * 6: rainbow_mode = True
            elif final_x < btn_w * 7:
                if not shape_lock: spray_mode = not spray_mode; shape_lock = True
            elif final_x < btn_w * 8:
                if not shape_lock: detect_and_draw_shape(canvas, rainbow_mode); shape_lock = True
            elif final_x < btn_w * 9:
                if not shape_lock:
                    result = brain.predict(canvas)
                    last_prediction = f"AI zice: {result}"
                    print(f"Detectie: {result}")
                    shape_lock = True
                    
            else: curr_color = (0, 0, 0); rainbow_mode = False; spray_mode = False
        
        else:
            shape_lock = False 
            last_button_index = -1
            if xp == 0 and yp == 0: xp, yp = final_x, final_y
            
            if pen_down:
                if curr_color == (0, 0, 0) and not rainbow_mode:
                    cv2.line(canvas, (xp, yp), (final_x, final_y), curr_color, eraser_thick)
                else:
                    if spray_mode: draw_spray(canvas, final_x, final_y, draw_color_now)
                    else: cv2.line(canvas, (xp, yp), (final_x, final_y), draw_color_now, brush_thick)
            xp, yp = final_x, final_y
    else:
        xp, yp = 0, 0
        shape_lock = False

    # ----------------------------------- GUI -----------------------------------
    cv2.rectangle(frame, (0, 0), (w, header_h), (50, 50, 50), -1)
    btn_w = w // 10
    labels = ["CLEAR", "BLUE", "GREEN", "RED", "YELLO", "MAGIC", "SPRAY", "SHAPE", "PREDICT", "ERASE"]
    
    for i in range(10):
        x1 = i * btn_w; x2 = (i + 1) * btn_w
        c_btn = (200, 200, 200)
        if i == 1: c_btn = draw_colors[0]
        if i == 2: c_btn = draw_colors[1]
        if i == 3: c_btn = draw_colors[2]
        if i == 4: c_btn = draw_colors[3]
        if i == 5: c_btn = rainbow_bgr     
        if i == 6: c_btn = (100, 100, 255) 
        if i == 7: c_btn = (150, 0, 150)   
        if i == 8: c_btn = (0, 255, 255) 
        if i == 9: c_btn = (0, 0, 0)       
        
        cv2.rectangle(frame, (x1+5, 5), (x2-5, header_h-5), c_btn, -1)
        selected = False
        if i == 1 and curr_color == draw_colors[0] and not rainbow_mode: selected = True
        if i == 2 and curr_color == draw_colors[1] and not rainbow_mode: selected = True
        if i == 3 and curr_color == draw_colors[2] and not rainbow_mode: selected = True
        if i == 4 and curr_color == draw_colors[3] and not rainbow_mode: selected = True
        if i == 5 and rainbow_mode: selected = True
        if i == 9 and curr_color == (0, 0, 0): selected = True
        if i == 6 and spray_mode: selected = True
        
        if selected: cv2.rectangle(frame, (x1+5, 5), (x2-5, header_h-5), (255, 255, 255), 4)

        tc = (255,255,255)
        if i == 0 or i == 5 or i == 8: tc = (0,0,0) 
        font_scale = 0.4
        if i == 8: font_scale = 0.35 
        cv2.putText(frame, labels[i], (x1+5, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, tc, 1)

    if last_prediction:
        cv2.rectangle(frame, (w//2 - 100, 650), (w//2 + 100, 700), (0,0,0), -1)
        cv2.putText(frame, last_prediction, (w//2 - 90, 690), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    frame_bg = cv2.bitwise_and(frame, img_inv)
    frame_final = cv2.bitwise_or(frame_bg, canvas)
    
    cv2.imshow("Teacher Control", frame_final)
    cv2.imshow("Projector", canvas)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord('s'): cv2.imwrite("Tabla_Mea.jpg", canvas)

cap.release()
cv2.destroyAllWindows()