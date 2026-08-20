from tkinter import Canvas
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
import pytesseract
import customtkinter as ctk
import cv2, csv, os, re, threading, math, time, shutil, queue
import numpy as np

try:
    from gpiozero import Button, Servo
except ImportError:
    Button = None
    Servo = None

#np.int, np.float, np.bool = int, float, bool # monkey-patch

# ---------------- Raspberry Pi hardware configuration ----------------
# BCM GPIO numbering
SENSOR_1_GPIO = 17
SENSOR_2_GPIO = 27

# Physical distance between the two photoelectric sensors, in metres.
# Measure the real spacing and change this value.
SENSOR_SPACING_M = 0.50

# The verified interface on this Pi is LOW at idle and HIGH when detected.
SENSOR_ACTIVE_LOW = False

# Ignore repeated transitions from the same sensor for this many seconds.
SENSOR_DEBOUNCE_S = 0.05

# Reset the displayed current speed to zero after no new vehicle for this time.
SPEED_DISPLAY_TIMEOUT_S = 3.0

# Speedometer range. All displayed vehicle speeds use km/h.
SPEEDOMETER_MAX_KMH = 120.0

# Prevent the same OCR result from opening the gate repeatedly while the car
# is still in front of the camera.
PLATE_COOLDOWN_S = 30.0

# Camera indexes to try, in order. USB webcams are normally index 0; some
# Raspberry Pi setups expose a camera as index 1.
CAMERA_INDEXES = (0, 1)

# Barrier servo configuration (BCM GPIO12 / physical pin 32).
# The pulse range is intentionally wide enough for common small 4.8-6 V servos.
# OPEN/CLOSED values are calibrated from the previous project's approximate
# pulse positions, but expressed as gpiozero Servo values (-1 .. +1).
BARRIER_SERVO_ENABLED = True
BARRIER_SERVO_GPIO = 12
BARRIER_MIN_PULSE_S = 0.0010
BARRIER_MAX_PULSE_S = 0.0020
BARRIER_OPEN_VALUE = -0.23
BARRIER_CLOSED_VALUE = 0.65
BARRIER_MOVE_STEPS = 45
BARRIER_STEP_DELAY_S = 0.03
BARRIER_OPEN_HOLD_S = 10.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(BASE_DIR, "files")

class GateVisionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Gate Vision - Automatic Vehicle Speed Detection and License Plate Recognition System")
        self.window.geometry("1280x720")
        self.window.configure(fg_color='#F0F0F0')

        # Core state
        self.distance1     = 0.0
        self.distance2     = 0.0
        self.time1         = 0.0
        self.time2         = 0.0
        self.current_speed = 0.0
        self.maximum_speed = 0.0
        self.running       = True
        self.alert_open    = False

        # GUI work requested by worker threads is executed only by Tk's main
        # thread through this queue. This avoids intermittent Tkinter crashes.
        self._ui_queue = queue.Queue()

        # Duplicate-plate protection
        self._plate_history_lock = threading.Lock()
        self._plate_last_processed = {}

        # Two-sensor speed measurement state
        self._speed_lock = threading.Lock()
        self._sensor1_time = None
        self._sensor2_time = None
        self._last_speed_time = 0.0
        self.sensor1 = None
        self.sensor2 = None

        # Barrier lock — prevents simultaneous open/close commands (motor safety)
        self._barrier_lock = threading.Lock()
        self._close_early  = threading.Event()
        self.barrier_servo = None

        # Initialise hardware. Each subsystem fails gracefully so the GUI can
        # still be tested on a normal PC without Raspberry Pi GPIO hardware.
        self._setup_speed_sensors()
        self._setup_barrier()

        # Camera
        self.cap = self._open_camera()

        # AI model
        model_path = os.path.join(FILES_DIR, "license_plate_yolo11n.pt")
        try:
            self.yolo_model = YOLO(model_path)
            print(f"\n[AI] Loaded YOLO model: {model_path}")
        except Exception as e:
            self.yolo_model = None
            print(f"\n[AI] ERROR: Could not load YOLO model: {e}")

        # OCR uses the system Tesseract executable through pytesseract.
        self.tesseract_available = shutil.which("tesseract") is not None
        if not self.tesseract_available:
            print("\n[OCR] WARNING: Tesseract not found. Install with: sudo apt install tesseract-ocr -y")

        # Pre-load CSV database
        self.plate_db = self._load_plate_database(os.path.join(FILES_DIR, "registered_vehicles.csv"))

        # Thread-shared frame data (protected by lock)
        self._frame_lock = threading.Lock()
        self.raw_frame   = None
        self.plate_crop  = None

        # Thread synchronisation events
        self._ocr_trigger = threading.Event()
        self._ocr_busy    = threading.Event()

        # Build UI
        self.create_widgets()

        # Keyboard shortcut
        self.window.bind('<space>', self.on_key_press)

        # Start worker threads
        threading.Thread(target=self.camera_read_thread,     daemon=True).start()
        threading.Thread(target=self.yolo_processing_thread, daemon=True).start()
        threading.Thread(target=self.ocr_processing_thread,  daemon=True).start()
        threading.Thread(target=self._speed_sensor_loop,     daemon=True).start()
        threading.Thread(target=self.speed_thread,           daemon=True).start()

        # Start GUI loops
        self._process_ui_queue()
        self.update_gui_loop()
        self.animate_speedometer()

    # GENERAL HELPERS
    def _post_ui(self, callback):
        """Schedule a callable to run safely on Tk's main thread."""
        if self.running:
            self._ui_queue.put(callback)

    def _process_ui_queue(self):
        try:
            while True:
                callback = self._ui_queue.get_nowait()
                try:
                    callback()
                except Exception as e:
                    print(f"\n[UI Callback Error] {e}")
        except queue.Empty:
            pass

        if self.running:
            self.window.after(25, self._process_ui_queue)

    @staticmethod
    def _normalise_plate_key(text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def _plate_is_in_cooldown(self, plate_text: str) -> bool:
        key = self._normalise_plate_key(plate_text)
        now = time.monotonic()
        with self._plate_history_lock:
            last = self._plate_last_processed.get(key)
            if last is not None and now - last < PLATE_COOLDOWN_S:
                return True
            self._plate_last_processed[key] = now

            # Keep the small history bounded during long-running operation.
            cutoff = now - max(PLATE_COOLDOWN_S * 4, 120.0)
            stale = [k for k, t in self._plate_last_processed.items() if t < cutoff]
            for k in stale:
                self._plate_last_processed.pop(k, None)
        return False

    def _open_camera(self):
        for index in CAMERA_INDEXES:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok = False
            for _ in range(8):
                ret, frame = cap.read()
                if ret and frame is not None:
                    ok = True
                    break
                time.sleep(0.05)

            if ok:
                print(f"\n[CAMERA] Camera opened successfully at index {index}.")
                return cap
            cap.release()

        print("\n[CAMERA] WARNING: No working camera found at indexes 0 or 1.")
        return None

    # DATABASE
    @staticmethod
    def _load_plate_database(path: str) -> dict:
        db = {}
        try:
            with open(path, mode='r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    key = re.sub(r'[^A-Z0-9]', '', row['plate'].strip().upper())
                    db[key] = row
            print(f"\n[DB] Loaded {len(db)} registered plates from '{path}'")
        except FileNotFoundError:
            print(f"\n[DB] WARNING: '{path}' not found — running with empty database.")
        return db

    # THREAD 1 — CAMERA
    def camera_read_thread(self):
        target_interval = 1.0 / 30
        while self.running:
            t0 = time.monotonic()
            if self.cap is None:
                time.sleep(0.5)
                continue

            ret, frame = self.cap.read()
            if ret:
                with self._frame_lock:
                    self.raw_frame = frame
            elapsed = time.monotonic() - t0
            wait = target_interval - elapsed
            if wait > 0:
                time.sleep(wait)

    # THREAD 2 — YOLO DETECTION
    def yolo_processing_thread(self):
        while self.running:
            if self.yolo_model is None:
                time.sleep(0.5)
                continue

            # Run only when the gate is closed.
            if self._barrier_lock.locked():
                time.sleep(0.1)
                continue

            # Skip while OCR is still busy or already triggered
            if self._ocr_busy.is_set() or self._ocr_trigger.is_set():
                time.sleep(0.05)
                continue

            with self._frame_lock:
                frame = self.raw_frame.copy() if self.raw_frame is not None else None

            if frame is None:
                time.sleep(0.01)
                continue

            results = self.yolo_model(frame, imgsz=640, verbose=False)
            boxes = results[0].boxes

            if len(boxes) == 0:
                time.sleep(0.02)
                continue

            # pick the single highest-confidence detection
            confs    = boxes.conf.cpu().numpy()
            best_idx = int(confs.argmax())
            best_conf = float(confs[best_idx])

            if best_conf < 0.65:
                continue

            xyxy = boxes.xyxy.cpu().numpy()[best_idx]
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Signal OCR: store crop and set events
            self._ocr_busy.set() # block YOLO immediately
            with self._frame_lock:
                self.plate_crop = crop
            self._ocr_trigger.set() # wake OCR thread

    # THREAD 3 — OCR
    def ocr_processing_thread(self):
        while self.running:
            triggered = self._ocr_trigger.wait(timeout=0.2)
            if not triggered:
                continue

            try:
                with self._frame_lock:
                    crop = self.plate_crop.copy() if self.plate_crop is not None else None

                if crop is None:
                    continue

                # GUI updates requested by this worker are queued for the main thread.
                self._post_ui(lambda c=crop: self.update_plate_image_ui(c))

                # Tesseract OCR for Raspberry Pi
                raw_text = self._read_plate_tesseract(crop)
                plate_text = self._preprocess_plate(raw_text)

                if plate_text:
                    print(f"\n[OCR] Raw: {raw_text!r}  ->  Plate: {plate_text}")
                    """
                    if self._plate_is_in_cooldown(plate_text):
                        print(f"\n[OCR] Same plate '{plate_text}' ignored during {PLATE_COOLDOWN_S:.0f}s cooldown.")
                    else:
                        self._post_ui(lambda t=plate_text: self.update_info(t))
                    """
                    self._post_ui(lambda t=plate_text: self.update_info(t))    
                else:
                    print(f"\n[OCR] No valid plate text found. Raw result: {raw_text!r}")

            except Exception as e:
                print(f"\n[OCR Error] {e}")
            finally:
                with self._frame_lock:
                    self.plate_crop = None
                self._ocr_trigger.clear()
                self._ocr_busy.clear() # allow YOLO to resume


    # TESSERACT OCR
    def _read_plate_tesseract(self, crop: np.ndarray) -> str:
        """Read a detected license-plate crop using Tesseract OCR."""
        try:
            if crop is None or crop.size == 0:
                return ""

        # 1. Simple resize and grayscale (fast, no heavy filters)
            h, w = crop.shape[:2]
            enlarged = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

            # 2. Run Tesseract exactly ONCE (prevents hanging)
            config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            text = pytesseract.image_to_string(gray, config=config).strip()
            return re.sub(r"[^A-Z0-9\-]", "", text.upper())

        except Exception as e:
            print(f"\n[Tesseract Error] {e}")
            return ""

    # Strip spaces and non-alphanumeric chars, and change to uppercase
    FIXES1_TRANS = str.maketrans('40815762', 'ADBISTGZ')
    FIXES2_TRANS = str.maketrans('ADBISTGZ', '40815762')

    @staticmethod
    def _preprocess_plate(text: str) -> str:
        text = text.upper()
        
        # Changed [A-Z] to [A-Z0-9] at index 1 so the regex catches the OCR mistake
        match = re.search(r'[A-Z0-9]{2}-?[A-Z0-9]{4}', text)
        
        if match:
            plate = match.group(0)

            # Apply translations and return immediately
            return (
                plate[0].translate(FIXES2_TRANS) + 
                plate[1].translate(FIXES1_TRANS) + 
                plate[2:].translate(FIXES2_TRANS)
            )
                    
        # Fallback: safely return the original uppercase text if no plate is found
        # Alternatively, you could return an empty string "" depending on your needs.
        return text

    # GUI — VIDEO LOOP
    def update_gui_loop(self):
        with self._frame_lock:
            frame = self.raw_frame.copy() if self.raw_frame is not None else None

        if frame is not None:
            height = self.video_label.winfo_height()
            width  = int(height * (4 / 3))
            if width <= 0 or height <= 0:
                width, height = 40, 30

            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)
            ctk_image = ctk.CTkImage(light_image=pil_image, size=(width, height))
            self.video_label.configure(image=ctk_image, text="")

        self.window.after(37, self.update_gui_loop)

    def update_plate_image_ui(self, cv_img: np.ndarray):
        try:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            img_w, img_h = pil_img.size
            h = self.plate_label.winfo_height()
            w = int(h * (img_w / img_h)) if img_h > 0 else 300
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))
            self.plate_label.configure(image=ctk_img, text="")
            self.plate_label._image = ctk_img
        except Exception as e:
            print(f"\n[UI Error] {e}")

    # GUI — OWNER INFO / GATE LOGIC
    def update_info(self, plate_text: str):
        lookup_key = self._normalise_plate_key(plate_text)
        matched_row = self.plate_db.get(lookup_key)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if matched_row:
            info_str = (
                f"Vehicle Plate : {matched_row['plate']}\n"
                f"Status        : Registered\n"
                f"Owner         : {matched_row['name']}\n"
                f"Region        : {matched_row['region']}\n"
                f"Vehicle Type  : {matched_row['type']}\n"
                f"Speed         : {self.maximum_speed:.1f} km/h\n"
                f"Detected Time : {now}"
            )
            self.info_text.configure(text=info_str)

            print(f"\n[GATE OPENING FOR 10 SECONDS] {now}")
            self.set_alert_state(False)
            threading.Thread(target=self.open_barrier, daemon=True).start()

        else:
            info_str = (
                f"Vehicle Plate : {plate_text}\n"
                f"Status        : Unregistered\n"
                f"Speed         : {self.maximum_speed:.1f} km/h\n"
                f"Detected Time : {now}"
            )
            self.info_text.configure(text=info_str)
            print(f"\n[UNREGISTERED VEHICLE: {plate_text}] {now}")
            self.set_alert_state(True)

    # SPEED SENSOR SETUP
    def _setup_speed_sensors(self):
        """Set up independent edge callbacks for both speed sensors.

        GPIO17 and GPIO27 were verified to change LOW/HIGH on the target Pi.
        Using independent callbacks prevents Sensor 2 from being missed while
        the code is waiting for Sensor 1 to release.
        """
        if Button is None:
            print(
                "\n[SPEED] gpiozero is not installed. "
                "Install it with: sudo apt install python3-gpiozero -y"
            )
            return

        try:
            self.sensor1 = Button(
                SENSOR_1_GPIO,
                pull_up=False,
                bounce_time=SENSOR_DEBOUNCE_S,
            )
            self.sensor2 = Button(
                SENSOR_2_GPIO,
                pull_up=False,
                bounce_time=SENSOR_DEBOUNCE_S,
            )

            # Do not depend on gpiozero callback threads here.  On the target
            # Raspberry Pi, direct state reads were verified on both GPIO17 and
            # GPIO27, so a fast polling loop is more robust and cannot miss S2
            # because of a callback/backend issue.
            self._sensor1_prev = bool(self.sensor1.is_pressed)
            self._sensor2_prev = bool(self.sensor2.is_pressed)

            print(
                f"\n[SPEED] Polling sensors ready: GPIO{SENSOR_1_GPIO} and "
                f"GPIO{SENSOR_2_GPIO}, spacing={SENSOR_SPACING_M:.3f} m (S1 -> S2)"
            )
        except Exception as e:
            self.sensor1 = None
            self.sensor2 = None
            print(f"\n[SPEED] Sensor setup failed: {e}")

    def _speed_sensor_loop(self):
        """Continuously poll both sensors and detect rising edges.

        This avoids gpiozero callback/backend issues while still using the
        already verified GPIO input states.  Both sensors are sampled together
        every 2 ms, so Sensor 2 is never blocked by Sensor 1 handling.
        """
        print("\n[SPEED] High-speed polling loop started (2 ms sampling).")
        while self.running and self.sensor1 is not None and self.sensor2 is not None:
            try:
                s1 = bool(self.sensor1.is_pressed)
                s2 = bool(self.sensor2.is_pressed)

                if s1 and not self._sensor1_prev:
                    self._sensor1_triggered()

                if s2 and not self._sensor2_prev:
                    self._sensor2_triggered()

                self._sensor1_prev = s1
                self._sensor2_prev = s2
                time.sleep(0.002)
            except Exception as e:
                print(f"\n[SPEED] Polling error: {e}")
                time.sleep(0.05)

    def _sensor1_triggered(self):
        """First sensor was blocked by a vehicle."""
        now = time.monotonic()
        with self._speed_lock:
            self._sensor1_time = now
            self._sensor2_time = None
            self.maximum_speed = 0.0
        print(f"\n[SPEED] Sensor 1 triggered at {now:.6f} — waiting for Sensor 2")

    def _sensor2_triggered(self):
        """Second sensor was blocked; calculate speed from elapsed time."""
        now = time.monotonic()

        with self._speed_lock:
            start_time = self._sensor1_time

            if start_time is None:
                print("\n[SPEED] Sensor 2 triggered before Sensor 1; ignored.")
                return

            elapsed = now - start_time
            self._sensor2_time = now
            self._sensor1_time = None

            if elapsed <= 0:
                return
            # Ignore stale S1 events rather than showing a meaningless speed.
            if elapsed > 30.0:
                print(f"\n[SPEED] Sensor 2 arrived after {elapsed:.2f}s; stale sequence ignored.")
                return

            speed_mps = SENSOR_SPACING_M / elapsed
            speed_kmh = speed_mps * 3.6
            self.current_speed = round(speed_kmh, 2)
            self.maximum_speed = max(self.maximum_speed, self.current_speed)
            self._last_speed_time = now

        print(
            f"\n[SPEED] Time={elapsed:.4f} s, "
            f"Speed={self.current_speed:.2f} km/h"
        )

    # THREAD 4 — SPEED DISPLAY TIMEOUT
    def speed_thread(self):
        while self.running:
            if (
                self.current_speed > 0
                and self._last_speed_time > 0
                and time.monotonic() - self._last_speed_time
                > SPEED_DISPLAY_TIMEOUT_S
            ):
                self.current_speed = 0.0
            time.sleep(0.1)

    # SPEEDOMETER
    def animate_speedometer(self):
        c = self.speed_canvas
        c.delete("all")

        w = c.winfo_width()
        h = c.winfo_height()
        if w <= 1 or h <= 1:
            w, h = 300, 200

        r  = min(w / 2, h / 1.7) - 25
        cx = w / 2
        cy = h / 2.21 + r * 0.2

        speed_val = max(0.0, min(SPEEDOMETER_MAX_KMH, self.current_speed))
        color = "#0073F0" if speed_val <= 60 else "#FF3333"

        # Background track (230°)
        c.create_arc(cx-r, cy-r, cx+r, cy+r,
                     start=-25, extent=230,
                     style="arc", outline="#E5E7EB", width=12)

        # Active (filled) track
        active_extent = int(speed_val / SPEEDOMETER_MAX_KMH * 230)
        if active_extent > 0:
            c.create_arc(cx-r, cy-r, cx+r, cy+r,
                         start=205, extent=-active_extent,
                         style="arc", outline=color, width=12)

        # Tick marks and numeric labels
        tick_step = 20
        for s in range(0, int(SPEEDOMETER_MAX_KMH) + 1, tick_step):
            ang = math.radians(205 - (s / SPEEDOMETER_MAX_KMH) * 230)
            ox  = cx + (r - 15) * math.cos(ang); oy = cy - (r - 15) * math.sin(ang)
            ix  = cx + (r - 25) * math.cos(ang); iy = cy - (r - 25) * math.sin(ang)
            c.create_line(ix, iy, ox, oy, fill="#9CA3AF", width=2)
            tx = cx + (r - 42) * math.cos(ang);  ty = cy - (r - 42) * math.sin(ang)
            c.create_text(tx, ty, text=str(s), font=("Arial", 10, "bold"), fill="#4B5563")

        # Needle
        ang = math.radians(205 - (speed_val / SPEEDOMETER_MAX_KMH) * 230)
        nx  = cx + (r - 20) * math.cos(ang)
        ny  = cy - (r - 20) * math.sin(ang)
        c.create_line(cx, cy, nx, ny, fill=color, width=3)

        # Centre cap
        c.create_oval(cx-9, cy-9, cx+9, cy+9, fill="#1F2937", outline="#ffffff", width=2)

        # Digital readout
        c.create_text(cx, cy + r * 0.5, text=f"{speed_val:.1f}",
                      font=("Arial", 26, "bold"), fill=color)
        c.create_text(cx, cy + r * 0.8, text="km/h",
                      font=("Arial", 14, "bold"), fill="#555555")

        self.window.after(37, self.animate_speedometer)

    # BARRIER
    def _setup_barrier(self):
        if not BARRIER_SERVO_ENABLED:
            print("\n[BARRIER] Servo disabled; simulation mode.")
            return
        if Servo is None:
            print("\n[BARRIER] gpiozero Servo is unavailable; simulation mode.")
            return

        try:
            self.barrier_servo = Servo(
                BARRIER_SERVO_GPIO,
                initial_value=None,
                min_pulse_width=BARRIER_MIN_PULSE_S,
                max_pulse_width=BARRIER_MAX_PULSE_S,
                frame_width=0.020,
            )
            # Keep PWM detached at startup so the servo does not buzz immediately.
            self.barrier_servo.value = None
            print(
                f"\n[BARRIER] Servo ready on GPIO{BARRIER_SERVO_GPIO} at 50 Hz "
                f"(pulse {BARRIER_MIN_PULSE_S*1000:.1f}-{BARRIER_MAX_PULSE_S*1000:.1f} ms, startup idle)."
            )
        except Exception as e:
            self.barrier_servo = None
            print(f"\n[BARRIER] Servo setup failed; simulation mode: {e}")

    def _move_barrier_servo(self, start_value: float, end_value: float):
        print(f"\n[BARRIER] Servo move command: {start_value:.2f} -> {end_value:.2f} on GPIO{BARRIER_SERVO_GPIO}")
        for step in range(BARRIER_MOVE_STEPS + 1):
            if not self.running:
                break
            fraction = step / max(BARRIER_MOVE_STEPS, 1)
            value = start_value + (end_value - start_value) * fraction
            value = max(-1.0, min(1.0, value))
            if self.barrier_servo is not None:
                try:
                    self.barrier_servo.value = value
                except Exception as e:
                    print(f"\n[BARRIER] Servo PWM error; simulation mode: {e}")
                    try:
                        self.barrier_servo.close()
                    except Exception:
                        pass
                    self.barrier_servo = None
            time.sleep(BARRIER_STEP_DELAY_S)
        # Detach PWM after the movement so the servo does not keep buzzing.
        if self.barrier_servo is not None:
            try:
                self.barrier_servo.value = None
            except Exception:
                pass

    def open_barrier(self):
        if not self._barrier_lock.acquire(blocking=False):
            print("\n[BARRIER] Already in motion/open — ignoring duplicate command.")
            return

        self._post_ui(lambda: self._set_barrier_ui("Barrier is opening...", True))
        try:
            self._move_barrier_servo(BARRIER_CLOSED_VALUE, BARRIER_OPEN_VALUE)
            self._post_ui(lambda: self.barrier_label.configure(
                text="Barrier is opened.", text_color="#008000"))
            print(f"\nAction: Barrier Opened [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]" )

            self._close_early.clear()
            self._close_early.wait(timeout=BARRIER_OPEN_HOLD_S)
            self.close_barrier()
        except Exception as e:
            print(f"\n[BARRIER] Open/hold error: {e}")
            if self._barrier_lock.locked():
                self._barrier_lock.release()

    def _set_barrier_ui(self, text: str, is_opening: bool):
        if is_opening:
            self.barrier_label.configure(text=text, text_color="#008000")
            self.led_canvas.itemconfig(self.led, fill="#00FF00", outline="#007700")
        else:
            self.barrier_label.configure(text=text, text_color="#CC0000")
            self.led_canvas.itemconfig(self.led, fill="#FF0000", outline="#AA0000")

    def close_barrier(self):
        self._post_ui(lambda: self._set_barrier_ui("Barrier is closing...", False))
        try:
            self._move_barrier_servo(BARRIER_OPEN_VALUE, BARRIER_CLOSED_VALUE)
            self._post_ui(lambda: self.barrier_label.configure(
                text="Barrier is closed.", text_color="#CC0000"))
            print(f"\nAction: Barrier Closed [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]" )
            self.maximum_speed = 0.0
        finally:
            if self._barrier_lock.locked():
                self._barrier_lock.release()

    # ALERT
    def set_alert_state(self, active: bool):
        self.alert_open = active
        if active:
            self.alert_canvas.itemconfig(self.alert, fill="#FFCC00", outline="#DDAA00")
        else:
            self.alert_canvas.itemconfig(self.alert, fill="#CCCCCC", outline="#BBBBBB")

    def on_key_press(self, event):
        if self._barrier_lock.locked():
            self._close_early.set()
        else:
            threading.Thread(target=self.open_barrier, daemon=True).start()

    # CLEANUP
    def on_closing(self):
        self.running = False
        self._ocr_trigger.set()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        for sensor in (self.sensor1, self.sensor2):
            try:
                if sensor is not None:
                    sensor.close()
            except Exception:
                pass

        try:
            if self.barrier_servo is not None:
                self.barrier_servo.value = None
                self.barrier_servo.close()
        except Exception:
            pass

        self.window.destroy()

    # WIDGETS
    def create_widgets(self):
        surface_color = "#FFFFFF"
        text_color    = "#111111"

        # Logos
        self.TUTGO  = ctk.CTkImage(Image.open(os.path.join(FILES_DIR, "TUTGO.png")),  size=(50, 50))
        self.ECDept = ctk.CTkImage(Image.open(os.path.join(FILES_DIR, "EC.png")), size=(50, 50))
        ctk.CTkLabel(self.window, image=self.TUTGO,  text="", anchor="w").place(
            relx=0.025, rely=0.05, relwidth=0.1, relheight=0.1, anchor="w")
        ctk.CTkLabel(self.window, image=self.ECDept, text="", anchor="e").place(
            relx=0.975, rely=0.05, relwidth=0.1, relheight=0.1, anchor="e")

        # Title
        ctk.CTkLabel(self.window, text="Department of Electronic Engineering",
                     font=("Arial", 20, "bold"), text_color="#000000").place(
            relx=0.5, rely=0.05, relwidth=0.46, relheight=0.1, anchor="center")

        # Live Stream
        self.video_frame = ctk.CTkFrame(self.window, corner_radius=15,
                                        fg_color=surface_color, border_width=3,
                                        border_color="#E0E0E0")
        self.video_frame.place(relx=0.025, rely=0.1, relwidth=0.3, relheight=0.4)
        ctk.CTkLabel(self.video_frame, text="Live Stream from Webcam",
                     font=("Arial", 16, "bold"), text_color="#000000", height=30
                     ).pack(pady=(7, 0))
        self.video_label = ctk.CTkLabel(self.video_frame, bg_color="#E5E5E5", text="")
        self.video_label.pack(fill="both", expand=True, padx=15, pady=(3, 15))

        # Speedometer
        self.speed_frame = ctk.CTkFrame(self.window, corner_radius=15,
                                        fg_color=surface_color, border_width=3,
                                        border_color="#E0E0E0")
        self.speed_frame.place(relx=0.35, rely=0.1, relwidth=0.3, relheight=0.4)
        ctk.CTkLabel(self.speed_frame, text="Vehicle Real-Time Speed",
                     font=("Arial", 16, "bold"), text_color="#000000", height=30
                     ).pack(pady=(7, 0))
        self.speed_canvas = Canvas(self.speed_frame, bg=surface_color, highlightthickness=0)
        self.speed_canvas.pack(fill="both", expand=True, padx=15, pady=(3, 15))

        # Detected Plate
        self.plate_frame = ctk.CTkFrame(self.window, corner_radius=15,
                                        fg_color=surface_color, border_width=3,
                                        border_color="#E0E0E0")
        self.plate_frame.place(relx=0.025, rely=0.53, relwidth=0.625, relheight=0.4)
        ctk.CTkLabel(self.plate_frame, text="Detected License Plate",
                     font=("Arial", 16, "bold"), text_color="#000000", height=30
                     ).pack(pady=(7, 0))
        self.plate_label = ctk.CTkLabel(self.plate_frame, bg_color="#E5E5E5", text="")
        self.plate_label.pack(fill="both", expand=True, padx=15, pady=(3, 15))

        # Owner Info
        self.info_frame = ctk.CTkFrame(self.window, corner_radius=15,
                                       fg_color=surface_color, border_width=3,
                                       border_color="#E0E0E0")
        self.info_frame.place(relx=0.675, rely=0.1, relwidth=0.3, relheight=0.4)
        ctk.CTkLabel(self.info_frame, text="Vehicle Owner Info",
                     font=("Arial", 16, "bold"), text_color="#000000", height=30
                     ).pack(pady=(7, 0))
        info_str = ("Vehicle Plate : \n"
                    "Status        : \n"
                    "Owner         : \n"
                    "Region        : \n"
                    "Vehicle Type  : \n"
                    "Speed         : \n"
                    "Detected Time : ")
        self.info_text = ctk.CTkLabel(self.info_frame, justify="left",
                                      font=("Noto Sans Mono", 14, "bold"), anchor="w",
                                      text_color="#333333", text=info_str)
        self.info_text.pack(anchor="w", padx=20, pady=20)

        # Action Panel
        self.action_frame = ctk.CTkFrame(self.window, corner_radius=15,
                                         fg_color=surface_color, border_width=3,
                                         border_color="#E0E0E0")
        self.action_frame.place(relx=0.675, rely=0.53, relwidth=0.3, relheight=0.4)
        ctk.CTkLabel(self.action_frame, text="Action",
                     font=("Arial", 16, "bold"), text_color="#000000", height=30
                     ).pack(pady=(7, 0))

        self.action_inner = ctk.CTkFrame(self.action_frame, fg_color="transparent")
        self.action_inner.pack(expand=True)

        # Barrier LED
        self.led_canvas = Canvas(self.action_inner, width=60, height=60,
                                 bg=surface_color, highlightthickness=0)
        self.led = self.led_canvas.create_oval(5, 5, 55, 55,
                                               fill="#FF0000", outline="#AA0000", width=3)
        self.led_canvas.grid(row=0, column=0, padx=20, pady=5)
        ctk.CTkLabel(self.action_inner, text="Barrier",
                     text_color=text_color, font=("Arial", 14, "bold")).grid(row=1, column=0)

        # Alert LED
        self.alert_canvas = Canvas(self.action_inner, width=60, height=60,
                                   bg=surface_color, highlightthickness=0)
        self.alert = self.alert_canvas.create_oval(5, 5, 55, 55,
                                                   fill="#CCCCCC", outline="#BBBBBB", width=3)
        self.alert_canvas.grid(row=0, column=2, padx=20, pady=5)
        ctk.CTkLabel(self.action_inner, text="Alert",
                     text_color=text_color, font=("Arial", 14, "bold")).grid(row=1, column=2)

        # Barrier status label
        self.barrier_border = ctk.CTkFrame(self.action_inner, width=170, height=45,
                                           corner_radius=15, border_width=3,
                                           border_color="#E0E0E0", fg_color="#F0F0F0")
        self.barrier_border.grid(row=8, column=0, columnspan=3, pady=37)
        self.barrier_border.grid_propagate(False)

        self.barrier_label = ctk.CTkLabel(self.barrier_border,
                                          text="Barrier is closed!",
                                          text_color="#CC0000",
                                          font=("Arial", 16, "bold"),
                                          fg_color="transparent")
        self.barrier_label.place(relx=0.5, rely=0.5, anchor="center")

def main():
    window = ctk.CTk()
    app = GateVisionApp(window)
    window.protocol("WM_DELETE_WINDOW", app.on_closing)

    def start_maximized():
        try:
            if os.name == "nt":
                window.state("zoomed")
            else:
                window.attributes("-zoomed", True)
        except Exception:
            try:
                window.attributes("-fullscreen", True)
            except Exception:
                pass

    window.after(100, start_maximized)
    window.bind("<F11>", lambda e: window.attributes(
        "-fullscreen", not window.attributes("-fullscreen")))
    window.bind("<Escape>", lambda e: window.attributes("-fullscreen", False))
    window.mainloop()


if __name__ == "__main__":
    main()
