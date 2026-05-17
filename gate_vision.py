from tkinter import Canvas
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
from paddleocr import PaddleOCR
import customtkinter as ctk
import cv2, csv, os, re, threading, math, time#, serial, pigpio
import numpy as np

np.int, np.float, np.bool = int, float, bool # monkey-patch

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

        # Barrier lock — prevents simultaneous open/close commands (motor safety)
        self._barrier_lock = threading.Lock()
        self._close_early  = threading.Event()

        # Serial sensor setup
        '''self.ser = serial.Serial("/dev/ttyS0", 115200, timeout=0.01)
        self.servo_pin = 12
        self.pi = pigpio.pi()
        self.pi.set_mode(self.servo_pin,pigpio.OUTPUT)
        self.pi.set_PWM_frequency(self.servo_pin,50)
        self.pi.set_PWM_range(self.servo_pin, 1024)'''

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        # AI Models
        self.yolo_model = YOLO("./files/license_plate_yolo11n.pt")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

        # Pre-load CSV database
        self.plate_db = self._load_plate_database("./files/registered_vehicles.csv")

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
        threading.Thread(target=self.speed_thread,           daemon=True).start()

        # Start GUI loops
        self.update_gui_loop()
        self.animate_speedometer()

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
            ret, frame = self.cap.read()
            if ret:
                with self._frame_lock:
                    self.raw_frame = cv2.flip(frame, 1)
            elapsed = time.monotonic() - t0
            wait = target_interval - elapsed
            if wait > 0:
                time.sleep(wait)

    # THREAD 2 — YOLO DETECTION
    def yolo_processing_thread(self):
        while self.running:
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
                continue

            # pick the single highest-confidence detection
            confs    = boxes.conf.cpu().numpy()
            best_idx = int(confs.argmax())
            best_conf = float(confs[best_idx])

            if best_conf < 0.64:
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

                # FIX 4: GUI update must happen on the main thread
                self.window.after(0, lambda c=crop: self.update_plate_image_ui(c))

                ocr_result = self.ocr.ocr(crop, cls=True)
                if ocr_result and ocr_result[0]:
                    raw_text   = " ".join(line[1][0] for line in ocr_result[0])
                    plate_text = self._preprocess_plate(raw_text)
                    self.window.after(0, lambda t=plate_text: self.update_info(t))

            except Exception as e:
                print(f"\n[OCR Error] {e}")
            finally:
                with self._frame_lock:
                    self.plate_crop = None
                self._ocr_trigger.clear()
                self._ocr_busy.clear() # allow YOLO to resume


    # Strip spaces and non-alphanumeric chars, and change to uppercase
    @staticmethod
    def _preprocess_plate(text: str) -> str:
        text.upper()
        match = re.search(r'\d{1}[A-Z]-?\d{4}', text)
        if match:
            text = match.group(0)
        return re.sub(r'[^A-Z0-9\-]', '', text)

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
        lookup_key = re.sub(r'[^A-Z0-9]', '', plate_text)
        matched_row = self.plate_db.get(lookup_key)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if matched_row:
            info_str = (
                f"Vehicle Plate : {matched_row['plate']}\n"
                f"Status        : Registered\n"
                f"Owner         : {matched_row['name']}\n"
                f"Region        : {matched_row['region']}\n"
                f"Vehicle Type  : {matched_row['type']}\n"
                f"Max Speed     : {self.maximum_speed} m/s\n"
                f"Detected Time : {now}"
            )
            self.info_text.configure(text=info_str)

            print(f"\n[GATE OPENING FOR 5 SECONDS] {now}")
            self.set_alert_state(False)
            threading.Thread(target=self.open_barrier, daemon=True).start()

        else:
            info_str = (
                f"Vehicle Plate : {plate_text}\n"
                f"Status        : Unregistered\n"
                f"Max Speed     : {self.maximum_speed} m/s\n"
                f"Detected Time : {now}"
            )
            self.info_text.configure(text=info_str)
            print(f"\n[UNREGISTERED VEHICLE: {plate_text}]")
            self.set_alert_state(True)

    # SERIAL SENSOR
    def read_data(self) -> int:
        try:
            for _ in range(100): # max 100 attempts — prevents infinite loop
                ch = self.ser.read(1)
                if not ch:
                    continue # serial timeout, try again
                if ch[0] == 0x59:
                    data = self.ser.read(8)
                    if len(data) >= 3 and data[0] == 0x59: # length guard before indexing
                        distance = data[1] + data[2] * 256
                        self.ser.reset_input_buffer()
                        return distance if distance < 1000 else 0
        except Exception as e:
            print(f"\n[Serial Error] {e}")
        self.ser.reset_input_buffer()
        return 0

    # THREAD 4 — SPEED MEASUREMENT
    def speed_thread(self):
        while self.running:
            self.distance1 = 900#self.read_data()
            self.time1     = time.monotonic() # monotonic: never jumps backward
            time.sleep(0.1)
            self.distance2 = 800#self.read_data()
            self.time2     = time.monotonic()

            delta_d = abs(self.distance1 - self.distance2) / 100.0 # cm → m, abs prevents negative
            delta_t = self.time2 - self.time1

            if delta_t > 0 and delta_d > 0:
                speed = round(delta_d / delta_t, 1)
                self.current_speed = speed
                if speed > self.maximum_speed:
                    self.maximum_speed = speed

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

        speed_val = max(0.0, min(100.0, self.current_speed))
        color = "#0073F0" if speed_val <= 60 else "#FF3333"

        # Background track (230°)
        c.create_arc(cx-r, cy-r, cx+r, cy+r,
                     start=-25, extent=230,
                     style="arc", outline="#E5E7EB", width=12)

        # Active (filled) track
        active_extent = int(speed_val / 100 * 230)
        if active_extent > 0:
            c.create_arc(cx-r, cy-r, cx+r, cy+r,
                         start=205, extent=-active_extent,
                         style="arc", outline=color, width=12)

        # Tick marks and numeric labels
        for s in range(0, 101, 10):
            ang = math.radians(205 - (s / 100) * 230)
            ox  = cx + (r - 15) * math.cos(ang); oy = cy - (r - 15) * math.sin(ang)
            ix  = cx + (r - 25) * math.cos(ang); iy = cy - (r - 25) * math.sin(ang)
            c.create_line(ix, iy, ox, oy, fill="#9CA3AF", width=2)
            tx = cx + (r - 42) * math.cos(ang);  ty = cy - (r - 42) * math.sin(ang)
            c.create_text(tx, ty, text=str(s), font=("Arial", 10, "bold"), fill="#4B5563")

        # Needle
        ang = math.radians(205 - (speed_val / 100) * 230)
        nx  = cx + (r - 20) * math.cos(ang)
        ny  = cy - (r - 20) * math.sin(ang)
        c.create_line(cx, cy, nx, ny, fill=color, width=3)

        # Centre cap
        c.create_oval(cx-9, cy-9, cx+9, cy+9, fill="#1F2937", outline="#ffffff", width=2)

        # Digital readout
        c.create_text(cx, cy + r * 0.5, text=f"{speed_val:.1f}",
                      font=("Arial", 26, "bold"), fill=color)
        c.create_text(cx, cy + r * 0.8, text="m/s",
                      font=("Arial", 14, "bold"), fill="#555555")

        self.window.after(37, self.animate_speedometer)

    # BARRIER
    def open_barrier(self):
        if not self._barrier_lock.acquire(blocking=False):
            print("\n[BARRIER] Already in motion — ignoring.")
            return
        # Note: lock is NOT released here — close_barrier releases it after fully closed
        def _ui_opening():
            self.barrier_label.configure(text="Barrier is opening...", text_color="#008000")
            self.led_canvas.itemconfig(self.led, fill="#00FF00", outline="#007700")
        self.window.after(0, _ui_opening)
        for i in range(110, 64, -1):
            #self.pi.set_PWM_dutycycle(self.servo_pin, i)
            time.sleep(0.01)
        self.window.after(0, lambda: self.barrier_label.configure(text="Barrier is opened.", text_color="#008000"))
        print(f"\nAction: Barrier Opened [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        self._close_early.clear()
        self._close_early.wait(timeout=5)
        threading.Thread(target=self.close_barrier, daemon=True).start()

    def close_barrier(self):
        def _ui_closing():
            self.barrier_label.configure(text="Barrier is closing...", text_color="#CC0000")
            self.led_canvas.itemconfig(self.led, fill="#FF0000", outline="#AA0000")
        self.window.after(0, _ui_closing)
        try:
            for i in range(65, 111, 1):
                #self.pi.set_PWM_dutycycle(self.servo_pin, i)
                time.sleep(0.01)
            self.window.after(0, lambda: self.barrier_label.configure(text="Barrier is closed.", text_color="#CC0000"))
            print(f"\nAction: Barrier Closed [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
            self.maximum_speed = 0
        finally:
            self._barrier_lock.release()   # always release — even if an error occurs

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
        if self.cap.isOpened():
            self.cap.release()
        try:
            self.pi.set_PWM_dutycycle(self.servo_pin, 0)  # stop servo signal
            self.pi.stop()
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass
        self.window.destroy()

    # WIDGETS
    def create_widgets(self):
        surface_color = "#FFFFFF"
        text_color    = "#111111"

        # Logos
        self.TUTGO  = ctk.CTkImage(Image.open("./files/TUTGO.png"),  size=(50, 50))
        self.ECDept = ctk.CTkImage(Image.open("./files/EC.png"), size=(50, 50))
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
                    "Max Speed     : \n"
                    "Detected Time : ")
        self.info_text = ctk.CTkLabel(self.info_frame, justify="left",
                                      font=("Fira Code", 14, "bold"), anchor="w",
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

window = ctk.CTk()
app    = GateVisionApp(window)
window.protocol("WM_DELETE_WINDOW", app.on_closing)

def start_maximized():
    try:
        if os.name == 'nt':
            window.state("zoomed") # Windows
        else:
            window.attributes("-zoomed", True) # Linux
    except Exception:
        window.attributes("-fullscreen", True) # fallback

window.after(100, start_maximized)
window.bind("<F11>",  lambda e: window.attributes("-fullscreen",
                                  not window.attributes("-fullscreen")))
window.bind("<Escape>", lambda e: window.attributes("-fullscreen", False))
window.mainloop()
