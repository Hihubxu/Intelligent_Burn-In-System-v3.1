"""
Burn-In System for Headphone Conditioning (Sony WH-1000XM5 Compatible)
Version: 3.1
Author: OpenSource Contributor
License: MIT
"""

import numpy as np
import sounddevice as sd
import time
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import logging
from threading import Thread, Event

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('burnin.log'),
        logging.StreamHandler()
    ]
)

SAVE_FILE = 'burnin_state.json'

class BurnInEngine:
    def __init__(self):
        self.sr = 48000
        self.duration = 6 * 3600  # 6 hours default
        self.chunk_size = 2048

        self.start_time = 0
        self.paused_time = 0
        self.total_elapsed = 0
        self.stop_event = Event()
        self.is_paused = Event()

        self.max_daily_runtime = 8 * 3600
        self.safety_margin = 0.1

        self.f_low = 25
        self._init_filters()

    def _init_filters(self):
        self.lowpass = np.array([0.0003, 0.0042, 0.0272, 0.1015, 0.2388,
                                 0.3540, 0.2388, 0.1015, 0.0272, 0.0042, 0.0003])

    def _generate_signal(self):
        chunk = np.random.randn(self.chunk_size) * 0.3
        t = np.linspace(0, self.chunk_size / self.sr, self.chunk_size)
        chunk += 0.1 * np.sin(2 * np.pi * self.f_low * t)
        return np.clip(chunk, -0.95, 0.95).astype(np.float32)

    def _save_state(self):
        state = {
            'total_elapsed': self.total_elapsed,
            'start_time': time.time(),
            'duration': self.duration
        }
        try:
            with open(SAVE_FILE, 'w') as f:
                json.dump(state, f)
            logging.info("State saved.")
        except Exception as e:
            logging.error(f"Failed to save state: {str(e)}")

    def _load_state(self):
        if os.path.exists(SAVE_FILE):
            try:
                with open(SAVE_FILE) as f:
                    state = json.load(f)
                    elapsed = state['total_elapsed']
                    remaining = state['duration'] - elapsed
                    safe_remaining = min(remaining, self.max_daily_runtime - elapsed % (24 * 3600))
                    return elapsed, safe_remaining
            except Exception as e:
                logging.error(f"Failed to load state: {str(e)}")
                return 0, self.duration
        return 0, self.duration

    def _safety_check(self):
        daily_runtime = self.total_elapsed % (24 * 3600)
        if daily_runtime >= self.max_daily_runtime * (1 - self.safety_margin):
            logging.warning("Daily maximum runtime reached.")
            return False
        return True

    def run(self):
        try:
            self.total_elapsed, remaining = self._load_state()
            self.stop_event.clear()
            last_save_time = time.time()

            with sd.OutputStream(samplerate=self.sr, blocksize=self.chunk_size,
                                 channels=2, dtype='float32') as stream:
                start_time = time.time()

                while not self.stop_event.is_set() and self.total_elapsed < remaining:
                    if not self._safety_check():
                        break

                    if self.is_paused.is_set():
                        self.paused_time = time.time()
                        while self.is_paused.is_set():
                            time.sleep(0.5)
                        self.total_elapsed += time.time() - self.paused_time

                    data = self._generate_signal()
                    stream.write(np.column_stack((data, data)))

                    if time.time() - last_save_time >= 300:
                        self._save_state()
                        last_save_time = time.time()

                    self.total_elapsed += self.chunk_size / self.sr

        except Exception as e:
            logging.error(f"Runtime error: {str(e)}")
            raise
        finally:
            if os.path.exists(SAVE_FILE):
                try:
                    os.remove(SAVE_FILE)
                except Exception as e:
                    logging.error(f"Failed to delete state file: {str(e)}")
            self.stop_event.set()


class BurnInApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Intelligent Burn-In System v3.1")
        self.engine = BurnInEngine()
        self._create_ui()
        self.protocol("WM_DELETE_WINDOW", self._safe_exit)

        if os.path.exists(SAVE_FILE):
            self._show_recovery_dialog()

    def _create_ui(self):
        self.geometry("400x250")

        control_frame = ttk.Frame(self)
        control_frame.pack(pady=10, fill=tk.X)

        self.btn_start = ttk.Button(control_frame, text="Start", command=self.start)
        self.btn_pause = ttk.Button(control_frame, text="Pause", state=tk.DISABLED, command=self.toggle_pause)
        self.btn_stop = ttk.Button(control_frame, text="Stop", state=tk.DISABLED, command=self.stop)

        self.btn_start.pack(side=tk.LEFT, padx=5)
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        progress_frame = ttk.Labelframe(self, text="Progress")
        progress_frame.pack(pady=5, fill=tk.X)

        self.elapsed_var = tk.StringVar(value="Elapsed: 00:00:00")
        self.remaining_var = tk.StringVar(value="Remaining: 00:00:00")
        self.stage_var = tk.StringVar(value="Current Stage: Low Frequency Activation")

        ttk.Label(progress_frame, textvariable=self.elapsed_var).pack(anchor=tk.W)
        ttk.Label(progress_frame, textvariable=self.remaining_var).pack(anchor=tk.W)
        ttk.Label(progress_frame, textvariable=self.stage_var).pack(anchor=tk.W)

        self.progress = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=380, mode='determinate')
        self.progress.pack(pady=10)

        self.safety_var = tk.StringVar(value="Safety Status: Normal")
        self.safety_label = ttk.Label(self, textvariable=self.safety_var, foreground="green")
        self.safety_label.pack()

    def _show_recovery_dialog(self):
        if messagebox.askyesno("Recovery", "An interrupted session was detected. Continue?"):
            self.start(resume=True)
        else:
            try:
                os.remove(SAVE_FILE)
            except Exception as e:
                logging.error(f"Failed to delete state file: {str(e)}")

    def _update_display(self):
        try:
            def format_time(sec):
                return time.strftime("%H:%M:%S", time.gmtime(sec))

            elapsed = self.engine.total_elapsed
            remaining = max(0, self.engine.duration - elapsed)

            self.elapsed_var.set(f"Elapsed: {format_time(elapsed)}")
            self.remaining_var.set(f"Remaining: {format_time(remaining)}")
            self.progress['value'] = (elapsed / self.engine.duration) * 100

            daily_use = elapsed % 86400
            if daily_use > self.engine.max_daily_runtime * 0.8:
                self.safety_label.config(foreground="red")
                self.safety_var.set("Safety Status: Near Daily Limit!")
            else:
                self.safety_label.config(foreground="green")
                self.safety_var.set("Safety Status: Normal")

            if self.engine.is_paused.is_set():
                self.stage_var.set("Current Stage: Paused")
            else:
                self.stage_var.set("Current Stage: Running")
        except Exception as e:
            logging.error(f"UI update error: {str(e)}")
        finally:
            self.after(1000, self._update_display)

    def start(self, resume=False):
        try:
            if not resume:
                self.engine.total_elapsed = 0

            self.btn_start.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.NORMAL)

            Thread(target=self.engine.run, daemon=True).start()
            self._update_display()
        except Exception as e:
            logging.error(f"Failed to start: {str(e)}")
            messagebox.showerror("Error", "Failed to start burn-in process")

    def toggle_pause(self):
        try:
            if self.engine.is_paused.is_set():
                self.engine.is_paused.clear()
                self.btn_pause.config(text="Pause")
            else:
                self.engine.is_paused.set()
                self.btn_pause.config(text="Resume")
                self.engine._save_state()
        except Exception as e:
            logging.error(f"Pause error: {str(e)}")

    def stop(self):
        try:
            self.engine.stop_event.set()
            self.btn_start.config(state=tk.NORMAL)
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.DISABLED)
            self.engine._save_state()
        except Exception as e:
            logging.error(f"Stop error: {str(e)}")

    def _safe_exit(self):
        try:
            if self.engine.stop_event.is_set():
                self.destroy()
            else:
                if messagebox.askokcancel("Exit", "Burn-in is still running. Exit anyway?"):
                    self.stop()
                    self.destroy()
        except Exception as e:
            logging.error(f"Exit error: {str(e)}")
            self.destroy()


if __name__ == "__main__":
    try:
        app = BurnInApp()
        app.mainloop()
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        messagebox.showerror("Fatal Error", f"A critical error occurred:\n{str(e)}")