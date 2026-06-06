import sys
import cv2
import time
import threading
import speech_recognition as sr
import pyttsx3
import easyocr 
import pytesseract # Fallback
import numpy as np
from datetime import datetime # Added for offline time

from ultralytics import YOLO
from deepface import DeepFace
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None
    print("PaddleOCR module not found. Please install it using 'pip install paddlepaddle paddleocr'")

from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout

# ================= MODELS =================
object_model = YOLO(r"D:\Vision Assistant\yolov5\yolov5\yolov8n.pt")
currency_model = YOLO(r"D:\Vision Assistant\yolov5\yolov5\currency_model.pt") # Load provided currency model
# Initialize PaddleOCR
ocr_model = None
if PaddleOCR:
    try:
        ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    except Exception as e:
        print(f"PaddleOCR failed to init: {e}")

# Initialize EasyOCR as backup (since it was already in the file)
if ocr_model is None:
    print("Using EasyOCR/Tesseract as fallback")
    try:
        reader_easy = easyocr.Reader(['en'])
    except:
        reader_easy = None


# ================= VOICE ENGINE =================
engine = pyttsx3.init()
engine.setProperty('rate', 120)
speech_lock = threading.Lock()
last_spoken = ""
last_time = 0

def speak(text):
    """Speak with lock to prevent overlapping"""
    global last_spoken, last_time
    if not text:
        return
    now = time.time()
    if text == last_spoken and now - last_time < 3:
        return
    if speech_lock.locked():
        return

    def run():
        global last_spoken, last_time
        with speech_lock:
            try:
                engine.say(text)
                engine.runAndWait()
                last_spoken = text
                last_time = time.time()
            except:
                pass
    threading.Thread(target=run, daemon=True).start()

# ================= PREPROCESSING_OCR =================
def preprocess_image(frame):
    """
    Apply preprocessing to improve OCR accuracy:
    - Grayscale
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Sharpening
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Sharpening
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Convert back to BGR for consistency if needed, but OCR usually takes gray or path
    # PaddleOCR accepts numpy array (BGR or Gray)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

# ================= CURRENCY & TEXT LOGIC =================
def check_currency_keywords(text_lower):
    """Check if text contains currency keywords with fuzzy matching"""
    currencies = []
    # Clean text but keep broad context
    clean_text = text_lower.replace("-", "").replace(".", "").replace(" ", "")
    
    # Context keywords that suggest this is Money
    context_words = ["rs", "rupees", "reserve", "bank", "india", "governor", "₹"]
    has_context = any(w in clean_text for w in context_words)
    
    # Logic Update: If context is weak but numbers are strong, we still detect.
    # We lowered the strictness so the system actually speaks.
    
    if "2000" in clean_text: currencies.append("₹2000 note")
    if "500" in clean_text: currencies.append("₹500 note")
    if "200" in clean_text: currencies.append("₹200 note")
    if "100" in clean_text: currencies.append("₹100 note")
    if "50" in clean_text: currencies.append("₹50 note")
    if "20" in clean_text: currencies.append("₹20 note")
    if "10" in clean_text: currencies.append("₹10 note")
    if "5" in clean_text: currencies.append("₹5 note")
    
    return list(set(currencies))

# ================= OCR WORKER THREAD =================
class OCRWorker(QThread):
    text_detected = pyqtSignal(str, list) # text, currency_list
    
    def __init__(self, use_paddle=True):
        super().__init__()
        self.running = True
        self.frame_to_process = None
        self.frame_lock = threading.Lock()
        self.active_reading = False
        self.use_paddle = use_paddle
        
        # Check if global ocr_model exists
        self.paddle_available = (ocr_model is not None)
    
    def set_reading_mode(self, enabled):
        self.active_reading = enabled

    def update_frame(self, frame):
        with self.frame_lock:
            self.frame_to_process = frame.copy()

    def run(self):
        while self.running:
            frame = None
            with self.frame_lock:
                if self.frame_to_process is not None:
                    frame = self.frame_to_process
                    self.frame_to_process = None # Clear after taking
            
            if frame is None:
                time.sleep(0.1)
                continue

            detected_text = []
            detected_currency = []
            
            try:
                # 1. Processing Logic
                full_text = ""
                
                if self.paddle_available and self.use_paddle:
                    # PaddleOCR
                    processed = preprocess_image(frame)
                    # Some PaddleOCR versions accept `cls` in .ocr(), others do not.
                    try:
                        ocr_out = ocr_model.ocr(processed, cls=True)
                    except TypeError:
                        ocr_out = ocr_model.ocr(processed)

                    if ocr_out and ocr_out[0]:
                        # SORTING LOGIC: Top-to-bottom, then Left-to-right
                        boxes = list(ocr_out[0]) if not isinstance(ocr_out[0], list) else ocr_out[0]
                        
                        # FIXED: Robust indexing to prevent "string index out of range"
                        # Paddle structure: [[ [x,y],[x,y],[x,y],[x,y] ], (text, score)]
                        def get_y(b):
                            try:
                                return b[0][0][1] # Get first point's Y coordinate
                            except (IndexError, TypeError):
                                return 0
                        
                        def get_x(b):
                            try:
                                return b[0][0][0] # Get first point's X coordinate
                            except (IndexError, TypeError):
                                return 0
                        
                        boxes.sort(key=get_y)
                        
                        # Group by lines (improved approach)
                        # Sort by Y first
                        boxes.sort(key=get_y)
                        
                        lines = []
                        current_line = []
                        last_y = -100
                        
                        for box in boxes:
                            y = get_y(box)
                            # Dynamic threshold based on box height could be better, but fixed 20px is okay for now
                            if y - last_y > 20: 
                                if current_line:
                                    # Sort previous line by X
                                    current_line.sort(key=get_x)
                                    lines.extend(current_line)
                                current_line = []
                                last_y = y # Update last_y to the new line's Y
                            else:
                                # Update last_y to average or keep min? 
                                # Better: keep last_y as the "line starter" y, don't update it for same line items
                                pass 
                                
                            current_line.append(box)
                            
                        if current_line:
                            current_line.sort(key=get_x)
                            lines.extend(current_line)

                        for line in lines:
                            if len(line) < 2 or not line[1]:
                                continue
                            text_data = line[1]
                            # Handle both tuple format (text, score) 
                            txt = str(text_data[0]) if isinstance(text_data, (tuple, list)) else str(text_data)
                            score = float(text_data[1]) if isinstance(text_data, (tuple, list)) and len(text_data) > 1 else 0.0
                            
                            # Logic Fix: Threshold lowered to 0.5 to catch more text
                            if txt and txt.strip() and score > 0.5:  
                                detected_text.append(txt)
                        full_text = " ".join(detected_text)
                
                elif 'reader_easy' in globals() and reader_easy:
                    # EasyOCR fallback
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if not self.active_reading:
                         gray = cv2.equalizeHist(gray)
                    
                    results = reader_easy.readtext(gray)
                    for (bbox, text, conf) in results:
                        if conf > 0.5:
                            detected_text.append(text)
                    full_text = " ".join(detected_text)
                
                # DEBUG: Print what OCR sees
                if full_text.strip():
                    print(f"OCR SEES: {full_text[:50]}...")

                # 2. Extract Currency
                if full_text:
                    detected_currency = check_currency_keywords(full_text.lower())
                
                # 3. Emit Results
                if self.active_reading:
                    self.text_detected.emit(full_text, detected_currency)
                elif detected_currency:
                    # If not reading, but found money, emit just money
                    self.text_detected.emit("", detected_currency)

            except Exception as e:
                print(f"OCR Worker Error: {e}")
            
            # Throttle to save CPU
            time.sleep(0.2)

    def stop(self):
        self.running = False
        self.wait()

# ================= GUI =================
class SightAI(QWidget):
    start_signal = pyqtSignal()
    stop_signal = pyqtSignal()
    reading_signal = pyqtSignal(bool)
    time_signal = pyqtSignal() # Added signal for Time
    voice_feedback_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PERCEPTAI")
        self.setGeometry(100, 100, 1200, 850)
        
        # Modern Dark Theme
        self.setStyleSheet("""
            QWidget {
                background-color: #0d1117;
                color: #c9d1d9;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel {
                color: #58a6ff;
            }
            QPushButton {
                background-color: #238636;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ea043;
            }
            QPushButton:disabled {
                background-color: #21262d;
                color: #8b949e;
            }
        """)

        # ====== GUI Widgets ======
        # Header
        self.title = QLabel("👁️ PERCEPTAI Vision")
        self.title.setFont(QFont("Segoe UI", 28, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("color: #58a6ff; margin-bottom: 10px;")

        # Video Area
        self.video = QLabel()
        self.video.setFixedSize(1100, 620)
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("border: 2px solid #30363d; border-radius: 12px; background-color: #010409;")

        self.diagnostics = QLabel(f"OCR: {'PaddleOCR' if ocr_model else 'EasyOCR'}")
        self.diagnostics.setStyleSheet("color: #ebac00; font-size: 12px;")
        
        self.info = QLabel("System Ready. Say 'Start' or 'Time'.")
        self.info.setAlignment(Qt.AlignCenter)
        self.info.setFont(QFont("Segoe UI", 16))
        self.info.setStyleSheet("color: #8b949e; margin-top: 10px;")
        
        self.voice_monitor = QLabel("🎤 Listening...")
        self.voice_monitor.setAlignment(Qt.AlignCenter)
        self.voice_monitor.setStyleSheet("color: #ffa657; font-style: italic;")

        self.counter = QLabel("Objects: 0 | Reading: OFF")
        self.counter.setAlignment(Qt.AlignCenter)
        self.counter.setStyleSheet("font-weight: bold; color: #f0f6fc;")

        # Controls
        self.start_btn = QPushButton("▶ START SYSTEM")
        self.stop_btn = QPushButton("⏹ STOP SYSTEM")
        self.night_btn = QPushButton("🌙 NIGHT VISION")
        self.read_btn = QPushButton("📖 START READING")

        self.stop_btn.setEnabled(False)
        self.read_btn.setCheckable(True)
        
        # Connect Buttons
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.night_btn.clicked.connect(self.toggle_night)
        self.read_btn.clicked.connect(self.toggle_reading_btn)

        # Layout
        btns_layout = QHBoxLayout()
        btns_layout.setSpacing(20)
        btns_layout.addWidget(self.start_btn)
        btns_layout.addWidget(self.stop_btn)
        btns_layout.addWidget(self.night_btn)
        btns_layout.addWidget(self.read_btn)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.title)
        main_layout.addWidget(self.diagnostics, alignment=Qt.AlignRight)
        main_layout.addWidget(self.video, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.info)
        main_layout.addWidget(self.voice_monitor)
        main_layout.addWidget(self.counter)
        main_layout.addLayout(btns_layout)

        # ====== State Variables ======
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.night_mode = False
        self.reading_mode = False
        self.running = False
        self.frame_count = 0
        self.current_emotion = "Neutral"
        self.emotion_lock = threading.Lock()
        
        # Smart Reading State
        self.last_read_block = ""
        
        # OCR Worker
        self.ocr_worker = OCRWorker()
        self.ocr_worker.text_detected.connect(self.handle_ocr_result)
        self.ocr_worker.start()

        # ====== Signals ======
        self.start_signal.connect(self.start_camera)
        self.stop_signal.connect(self.stop_camera)
        self.reading_signal.connect(self.set_reading_mode)
        self.time_signal.connect(self.announce_time) # Connect Time Signal
        self.voice_feedback_signal.connect(self.update_voice_feedback)

        # Start Voice Thread
        threading.Thread(target=self.voice_listener, daemon=True).start()
        # Start Emotion Thread
        threading.Thread(target=self.emotion_thread, daemon=True).start()

    # ================= LOGIC =================
    def handle_ocr_result(self, text, currency_list):
        # Money Announcement Logic
        if currency_list:
            for c in set(currency_list):
               speak(f"Money Detected: {c}")
               self.info.setText(f"💰 Found: {c}")

        # Text Reading Logic
        if self.reading_mode:
            if text and text.strip():
                import difflib
                # Check similarity with last spoken text
                seq = difflib.SequenceMatcher(None, self.last_read_block, text)
                similarity = seq.ratio()
                
                # Speak only if significantly different (< 0.8 similarity)
                if similarity < 0.8:
                    speak(text)
                    self.last_read_block = text
            else:
                # Do not clear immediately to prevent re-reading same text if it creates a blank frame flicker
                pass
    
    def announce_time(self):
        """Speaks the current time offline"""
        now = datetime.now()
        current_time = now.strftime("%I:%M %p") # Example: 07:30 PM
        speak(f"The time is {current_time}")
        self.info.setText(f"🕒 Time: {current_time}")

    def emotion_thread(self):
        """Analyze face emotion in background to prevent lag"""
        last_announced_emotion = "Neutral"
        
        while True:
            if not self.running:
                time.sleep(1)
                continue
                
            if hasattr(self, 'current_frame_rgb'):
                try:
                    # DeepFace analysis
                    small_frame = cv2.resize(self.current_frame_rgb, (0,0), fx=0.5, fy=0.5)
                    # Enforce detection = True avoids random guesses on walls/background
                    analysis = DeepFace.analyze(small_frame, actions=['emotion'], 
                                              enforce_detection=True, silent=True)
                    
                    if analysis and isinstance(analysis, list):
                        top_emotion = analysis[0]['dominant_emotion']
                    elif analysis and isinstance(analysis, dict):
                        top_emotion = analysis['dominant_emotion']
                    else:
                        top_emotion = "Unknown"

                    with self.emotion_lock:
                         if top_emotion != self.current_emotion:
                             self.current_emotion = top_emotion.capitalize()
                    
                    # Announcement Logic - Persistence check (avoid speaking every flicker)
                    if self.running and top_emotion != "neutral" and top_emotion != "Unknown":
                        if top_emotion != last_announced_emotion:
                            # Verify for one more cycle before speaking
                            time.sleep(0.5) 
                            speak(f"Person looks {top_emotion}")
                            last_announced_emotion = top_emotion
                    
                except ValueError: 
                    # DeepFace raises ValueError if no face detected when enforce_detection=True
                    with self.emotion_lock:
                        self.current_emotion = "No Face"
                except Exception as e:
                    pass
            
            time.sleep(2) # check emotion every 2 seconds

    def update_voice_feedback(self, text):
        self.voice_monitor.setText(f"🎤 {text}")

    def voice_listener(self):
        r = sr.Recognizer()
        mic = sr.Microphone()
        
        with mic as source:
            try:
                r.adjust_for_ambient_noise(source, duration=1)
            except: pass
            
        while True:
            try:
                with mic as source:
                    audio = r.listen(source, timeout=3, phrase_time_limit=5)
                
                self.voice_feedback_signal.emit("Processing...")
                text = r.recognize_google(audio).lower()
                self.voice_feedback_signal.emit(f"Heard: '{text}'")
                print(f"Voice Command: {text}")

                if "time" in text:
                    self.time_signal.emit() # Trigger offline time
                elif "start reading" in text:
                    self.reading_signal.emit(True)
                elif "stop reading" in text:
                    self.reading_signal.emit(False)
                elif "start" in text and "reading" not in text:
                    self.start_signal.emit()
                elif "stop" in text and "reading" not in text:
                    self.stop_signal.emit()
                elif "night" in text:
                    self.night_mode = not self.night_mode
                
            except sr.WaitTimeoutError:
                self.voice_feedback_signal.emit("Listening...")
            except sr.UnknownValueError:
                self.voice_feedback_signal.emit("Listening...")
            except Exception as e:
                self.voice_feedback_signal.emit("Listening...")
                time.sleep(1)

    def start_camera(self):
        if self.running: return
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not self.cap.isOpened():
                raise Exception("Could not open video device")
            
            self.running = True
            self.timer.start(30)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.info.setText("System Active - Scanning...")
            speak("System started")
        except Exception as e:
            self.info.setText(f"Camera Error: {e}")

    def stop_camera(self):
        if not self.running: return
        self.running = False
        self.timer.stop()
        if self.cap: self.cap.release()
        self.video.clear()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.info.setText("System Standby")
        speak("System stopped")

    def toggle_night(self):
        self.night_mode = not self.night_mode
        self.info.setText(f"Night Mode: {'ON' if self.night_mode else 'OFF'}")

    def toggle_reading_btn(self):
        is_on = self.read_btn.isChecked()
        self.set_reading_mode(is_on)

    def set_reading_mode(self, enabled):
        self.reading_mode = enabled
        self.read_btn.setChecked(enabled)
        self.ocr_worker.set_reading_mode(enabled)
        self.read_btn.setText(f"📖 STOP READING" if enabled else "📖 START READING")
        state_text = "ACTIVE" if enabled else "OFF"
        self.counter.setText(f"Reading Mode: {state_text}")
        if enabled:
            self.read_btn.setStyleSheet("background-color: #da3633; color: white;")
            speak("Reading mode activated")
        else:
            self.read_btn.setStyleSheet("background-color: #238636; color: white;")
            speak("Reading mode deactivated")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        self.frame_count += 1

        if self.night_mode:
            frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
            frame[:, :, 0] = cv2.add(frame[:, :, 0], 50)
            frame[:, :, 2] = cv2.subtract(frame[:, :, 2], 50)

        # Store for processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame_rgb = rgb.copy()
        self.ocr_worker.update_frame(frame)

        # Object Detection
        obj_count = 0
        
        def process_detections(model, color_base, conf_thresh=0.5):
            nonlocal obj_count, frame
            results = model(rgb, conf=conf_thresh, verbose=False)[0]
            detected = []
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls)]
                conf = float(box.conf)
                obj_count += 1
                
                h_px = (y2 - y1)
                dist_m = 500 / h_px if h_px > 0 else 0
                
                label_text = f"{label} {conf:.2f} | {dist_m:.1f}m"
                detected.append((label, conf, dist_m))

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_base, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_base, 2)
            
            return detected

        try:
            h_frame, w_frame, _ = frame.shape
            frame_area = h_frame * w_frame

            # Main Objects - Lowered threshold to 0.55 to catch more objects
            main_detected = process_detections(object_model, (0, 255, 255), conf_thresh=0.55) 
            
            # Currency - Keep high threshold 0.85
            currency_detected = []
            raw_currency = process_detections(currency_model, (0, 255, 0), conf_thresh=0.85)
            for c in raw_currency:
                currency_detected.append(c)

            # --- Persistence Logic ---
            if currency_detected and not self.reading_mode:
                # Require 3 consecutive frames of same currency
                c_label = currency_detected[0][0]
                c_dist = currency_detected[0][2]
                
                if not hasattr(self, 'currency_history'): self.currency_history = []
                self.currency_history.append(c_label)
                if len(self.currency_history) > 3: self.currency_history.pop(0)

                # Check if last 3 frames saw the same currency
                if len(self.currency_history) == 3 and all(x == c_label for x in self.currency_history):
                     speak(f"{c_label} found at {c_dist:.1f} meters")
                     self.currency_history = [] 
            else:
                self.currency_history = []

            # Announce objects only if no currency and not reading
            if not currency_detected and main_detected and not self.reading_mode:
                 # Group objects
                 current_objects = sorted(main_detected, key=lambda x: x[2]) # Sort by distance
                 
                 # SIMPLIFIED LOGIC: Just announce what we see, throttled by time
                 # Don't wait for 5 perfect frames of identical sets.
                 
                 if current_objects:
                     # Get unique labels with their closest distance
                     unique_objs = {}
                     for obj in current_objects:
                         lbl = obj[0]
                         dst = obj[2]
                         if lbl not in unique_objs:
                             unique_objs[lbl] = dst
                     
                     # Construct Announcement
                     # List ALL unique objects found
                     items_to_say = list(unique_objs.items())
                     
                     # Create the text
                     parts = []
                     for lbl, dst in items_to_say:
                         part = f"{lbl} at {dst:.1f}m"
                         # Emotion check for Person
                         if lbl.lower() == "person":
                             with self.emotion_lock:
                                 # Only say emotion if valid and NOT "No Face"
                                 if self.current_emotion not in ["Neutral", "Unknown", "No Face"]:
                                     part += f" looking {self.current_emotion}"
                         parts.append(part)
                     
                     scene_text = ", ".join(parts)
                     
                     # Speak only if significantly different from last spoken OR enough time passed
                     # Use global last_spoken/last_time from speak function? 
                     # Better: local throttle to avoid "Person, Person, Person"
                     
                     if not hasattr(self, 'last_scene_text'): self.last_scene_text = ""
                     if not hasattr(self, 'last_scene_time'): self.last_scene_time = 0
                     
                     now = time.time()
                     if scene_text != self.last_scene_text or (now - self.last_scene_time > 5):
                         speak(scene_text)
                         self.last_scene_text = scene_text
                         self.last_scene_time = now
            else:
                pass

                 
        except Exception as e:
            pass

        # Update UI Frame
        with self.emotion_lock:
             emo_text = self.current_emotion

        self.counter.setText(f"Objects: {obj_count} | Emotion: {emo_text}")
        
        if self.reading_mode:
             cv2.putText(frame, "READING MODE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        self.video.setPixmap(QPixmap.fromImage(qimg))
        
    def closeEvent(self, event):
        self.ocr_worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SightAI()
    window.show()
    sys.exit(app.exec_())