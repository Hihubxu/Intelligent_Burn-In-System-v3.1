"""
Sony WH-1000XM5 Headphone Burn-in Tool v5.1

This program automates the burn-in process for Sony WH-1000XM5 headphones to optimize audio performance.
It generates specific audio signals (sine waves, pink noise) for different burn-in phases, with customizable presets.
Features include:
- Two presets: Quick Test and Standard Burn-in
- Four phases: Low Frequency Activation, Full Frequency Expansion, Dynamic Optimization, Natural Break-in
- Audio device testing and safety checks
- Pause/Resume and Stop functionality
- Progress monitoring and temperature warnings

Requirements: Python 3, numpy, sounddevice, tkinter
Usage: Run the script to launch the GUI, select a preset and phase, and start the burn-in process.

Author: HihubXu
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
from threading import Thread
import platform
import sys

# --------------------- Configuration Setup ---------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('xm5_burnin.log'),
        logging.StreamHandler()
    ]
)

CONFIG_FILE = 'xm5_burnin_config.json'

# --------------------- Phase Definitions ---------------------
PHASE_PRESETS = {
    'Quick Test': {
        'Low Frequency Activation': {'duration': 300, 'freqs': [20, 50, 100], 'volume': 0.3, 'desc': 'Quickly activate low-frequency response'},
        'Full Frequency Expansion': {'duration': 300, 'freqs': [20, 200, 2000, 8000], 'volume': 0.4, 'desc': 'Quickly balance full frequency range'},
        'Dynamic Optimization': {'duration': 300, 'freqs': [20, 5000, 18000], 'sweep': True, 'volume': 0.35, 'desc': 'Quickly optimize dynamic range'},
        'Natural Break-in': {'duration': 300, 'music': True, 'volume': 0.5, 'desc': 'Quickly simulate music signals'}
    },
    'Standard Burn-in': {
        'Low Frequency Activation': {'duration': 10 * 3600, 'freqs': [20, 50, 100], 'volume': 0.5, 'desc': 'Deeply activate low-frequency response'},
        'Full Frequency Expansion': {'duration': 12 * 3600, 'freqs': [20, 200, 2000, 8000], 'volume': 0.6, 'desc': 'Fully balance frequency response'},
        'Dynamic Optimization': {'duration': 10 * 3600, 'freqs': [20, 5000, 18000], 'sweep': True, 'volume': 0.55, 'desc': 'Enhance transient response'},
        'Natural Break-in': {'duration': 8 * 3600, 'music': True, 'volume': 0.7, 'desc': 'Simulate real music playback'}
    }
}

# --------------------- Core Engine ---------------------
class XM5BurnInEngine:
    def __init__(self, app_ref=None):
        self.sr = 44100
        self.chunk_size = 1024
        self.app_ref = app_ref
        self.stream = None
        self.current_phase = ""
        self.phase_elapsed = 0
        self.total_elapsed = 0
        self.is_running = False
        self.is_paused = False
        self.stop_requested = False
        self.preset = "Standard Burn-in"
        self.work_interval = 45 * 60
        self.rest_interval = 15 * 60
        self.last_temp_check = 0
        self.device_id = self._init_audio_device()
        self.load_config()

    def _init_audio_device(self):
        try:
            sd._terminate()
            time.sleep(0.5)
            sd._initialize()
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if (dev['max_output_channels'] >= 2 and
                        not dev['name'].lower().startswith(('mme', 'virtual'))):
                    logging.info(f"Selected audio device: {dev['name']}")
                    return i
            default_out = sd.default.device[1]
            logging.warning(f"Using default output device: {devices[default_out]['name']}")
            return default_out
        except Exception as e:
            logging.error(f"Audio initialization failed: {str(e)}")
            return None

    def _generate_signal(self):
        phase = PHASE_PRESETS[self.preset][self.current_phase]
        if phase.get('music'):
            return self._generate_pink_noise(phase['volume'])
        t = np.linspace(0, self.chunk_size / self.sr, self.chunk_size)
        signal = np.zeros(self.chunk_size)
        if phase.get('sweep'):
            sweep_duration = 5
            sweep_range = phase['freqs'][-1] - phase['freqs'][0]
            current_pos = (self.phase_elapsed % sweep_duration) / sweep_duration
            freq = phase['freqs'][0] + current_pos * sweep_range
            signal = phase['volume'] * np.sin(2 * np.pi * freq * t)
        else:
            for f in phase['freqs']:
                signal += (phase['volume'] / len(phase['freqs'])) * np.sin(2 * np.pi * f * t)
        return np.clip(
            np.column_stack((signal, signal)),
            -0.8, 0.8
        ).astype(np.float32)

    def _generate_pink_noise(self, volume):
        white = np.random.randn(self.chunk_size)
        pink = np.cumsum(white) * (volume / 3)
        return np.column_stack((
            np.clip(pink, -0.8, 0.8),
            np.clip(pink, -0.8, 0.8)
        )).astype(np.float32)

    def _play_notification(self, freq, duration=0.5):
        try:
            if self.device_id is None:
                self.device_id = self._init_audio_device()
                if self.device_id is None:
                    raise RuntimeError("Cannot initialize audio device")
            samples = int(self.sr * duration)
            t = np.linspace(0, duration, samples, False)
            tone = 0.3 * np.sin(2 * np.pi * freq * t)
            env = np.minimum(t / duration, (duration - t) / duration)
            stereo_signal = np.column_stack((tone * env, tone * env)).astype(np.float32)
            sd.default.device = self.device_id
            sd.play(stereo_signal, samplerate=self.sr, device=self.device_id, blocking=True)
        except Exception as e:
            logging.error(f"Notification sound playback failed: {str(e)}")
            raise

    def run_phase(self, phase_name, preset):
        if not self._check_device_ready():
            raise RuntimeError("Audio device not ready")
        self.preset = preset
        self.current_phase = phase_name
        self._reset_timers()
        self._play_notification(880)
        try:
            with sd.OutputStream(
                    samplerate=self.sr,
                    blocksize=self.chunk_size,
                    channels=2,
                    dtype='float32',
                    device=self.device_id
            ) as stream:
                self.stream = stream
                start_time = time.time()
                while self._should_continue_running(phase_name):
                    if self.is_paused:
                        time.sleep(0.5)
                        continue
                    data = self._generate_signal()
                    try:
                        stream.write(data)
                    except Exception as e:
                        logging.error(f"Audio write error: {str(e)}")
                        if not self._recover_stream():
                            break
                    self._update_timers(start_time)
                    self._update_ui(phase_name)
                    self._check_safety()
                    time.sleep(0.01)
                self._play_notification(440)
        except Exception as e:
            logging.error(f"Phase execution error: {str(e)}")
            raise
        finally:
            self._cleanup()

    def _check_device_ready(self):
        try:
            test_signal = np.zeros((self.chunk_size, 2), dtype=np.float32)
            with sd.OutputStream(
                    samplerate=self.sr,
                    blocksize=self.chunk_size,
                    channels=2,
                    dtype='float32',
                    device=self.device_id
            ) as stream:
                stream.write(test_signal)
            return True
        except Exception as e:
            logging.error(f"Device check failed: {str(e)}")
            return False

    def _reset_timers(self):
        self.phase_elapsed = 0
        self.total_elapsed = 0
        self.is_running = True
        self.is_paused = False
        self.stop_requested = False

    def _should_continue_running(self, phase_name):
        phase_config = PHASE_PRESETS[self.preset][phase_name]
        return (self.phase_elapsed < phase_config['duration'] and
                not self.stop_requested)

    def _recover_stream(self):
        try:
            if self.stream:
                self.stream.stop()
                time.sleep(1)
                self.stream.start()
                return True
        except Exception as e:
            logging.error(f"Stream recovery failed: {str(e)}")
        return False

    def _update_timers(self, start_time):
        self.phase_elapsed = time.time() - start_time
        self.total_elapsed += self.chunk_size / self.sr

    def _update_ui(self, phase_name):
        if self.app_ref and time.time() % 0.2 < 0.01:
            phase_config = PHASE_PRESETS[self.preset][phase_name]
            progress = self.phase_elapsed / phase_config['duration']
            self.app_ref.update_progress(progress, phase_name)

    def _check_safety(self):
        if time.time() - self.last_temp_check > 1800:
            self.last_temp_check = time.time()
            if self.app_ref:
                self.app_ref.show_temp_warning()
        if self.phase_elapsed >= self.work_interval:
            self._take_break()

    def _take_break(self):
        if self.app_ref:
            self.app_ref.update_progress(0, "On Break")
        rest_start = time.time()
        while (time.time() - rest_start < self.rest_interval and
               not self.stop_requested):
            if not self.is_paused:
                time.sleep(1)

    def _cleanup(self):
        self.is_running = False
        self.stream = None
        self.save_config()

    def save_config(self):
        config = {
            'device_id': self.device_id,
            'total_elapsed': self.total_elapsed,
            'work_interval': self.work_interval,
            'rest_interval': self.rest_interval,
            'last_preset': self.preset
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            logging.error(f"Config save failed: {str(e)}")

    def load_config(self):
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE) as f:
                    config = json.load(f)
                    self.device_id = config.get('device_id', self._init_audio_device())
                    self.total_elapsed = config.get('total_elapsed', 0)
                    self.work_interval = config.get('work_interval', 45 * 60)
                    self.rest_interval = config.get('rest_interval', 15 * 60)
                    self.preset = config.get('last_preset', 'Standard Burn-in')
        except Exception as e:
            logging.warning(f"Config load failed: {str(e)}")

# --------------------- Modern GUI Interface ---------------------
class XM5BurnInApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Sony WH-1000XM5 Burn-in Tool v5.1 ({platform.system()})")
        self.engine = XM5BurnInEngine(app_ref=self)
        self.phase_radios = []  # Store phase radio buttons for enabling/disabling
        self._setup_ui()
        self._setup_styles()
        self.protocol("WM_DELETE_WINDOW", self._safe_exit)
        if platform.system() == 'Windows':
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)

    def _setup_styles(self):
        self.style = ttk.Style()
        phase_colors = {
            'Low Frequency Activation': '#4e79a7',
            'Full Frequency Expansion': '#f28e2b',
            'Dynamic Optimization': '#e15759',
            'Natural Break-in': '#59a14f'
        }
        for phase, color in phase_colors.items():
            self.style.configure(
                f"{phase}.TRadiobutton",
                foreground=color,
                font=('Arial', 14, 'bold')  # Reduced font size to 14
            )
        self.style.configure(
            'TFrame',
            background='#f0f0f0'
        )
        self.style.configure(
            'TLabelFrame',
            font=('Arial', 16, 'bold')
        )
        self.style.configure(
            'TButton',
            font=('Arial', 16),
            padding=6
        )
        self.style.configure(
            'Toolbutton',
            font=('Arial', 16)
        )

    def _setup_ui(self):
        self.geometry("900x680")
        self.minsize(800, 600)
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        self.btn_start = ttk.Button(
            control_frame,
            text="Start Burn-in",
            command=self._start,
            style='Accent.TButton'
        )
        self.btn_pause = ttk.Button(
            control_frame,
            text="Pause",
            state=tk.DISABLED,
            command=self._toggle_pause
        )
        self.btn_stop = ttk.Button(
            control_frame,
            text="Stop",
            state=tk.DISABLED,
            command=self._stop
        )
        self.btn_test = ttk.Button(
            control_frame,
            text="Device Test",
            command=self._test_audio
        )
        self.btn_start.pack(side=tk.LEFT, padx=(0, 16))
        self.btn_pause.pack(side=tk.LEFT, padx=(0, 16))
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 20))
        self.btn_test.pack(side=tk.RIGHT)
        preset_frame = ttk.LabelFrame(
            main_frame,
            text="Preset Options",
            padding=(15, 10)
        )
        preset_frame.pack(fill=tk.X, pady=15)
        self.preset_var = tk.StringVar(value=self.engine.preset)
        for preset in PHASE_PRESETS:
            ttk.Radiobutton(
                preset_frame,
                text=preset,
                variable=self.preset_var,
                value=preset,
                style='Toolbutton'
            ).pack(side=tk.LEFT, padx=15)
        phase_frame = ttk.LabelFrame(
            main_frame,
            text="Select Phase",
            padding=(15, 10)
        )
        phase_frame.pack(fill=tk.X, pady=15)
        self.phase_var = tk.StringVar()
        # Use a grid layout to arrange radio buttons in two rows
        inner_frame = ttk.Frame(phase_frame)
        inner_frame.pack(fill=tk.X, expand=True)
        phases = list(PHASE_PRESETS['Standard Burn-in'].keys())
        for idx, phase in enumerate(phases):
            row = idx // 2  # Two buttons per row
            col = idx % 2   # Two columns
            radio = ttk.Radiobutton(
                inner_frame,
                text=phase,
                variable=self.phase_var,
                value=phase,
                style=f"{phase}.TRadiobutton"
            )
            radio.grid(row=row, column=col, padx=10, pady=5, sticky=tk.W)
            self.phase_radios.append(radio)
        progress_frame = ttk.LabelFrame(
            main_frame,
            text="Progress Monitor",
            padding=(20, 15)
        )
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=15)
        self.phase_label = ttk.Label(
            progress_frame,
            text="Waiting to start...",
            font=('Arial', 16, 'bold'),
            anchor=tk.CENTER
        )
        self.phase_label.pack(pady=(0, 15))
        self.progress = ttk.Progressbar(
            progress_frame,
            orient=tk.HORIZONTAL,
            length=700,
            mode='determinate'
        )
        self.progress.pack(fill=tk.X, pady=5)
        time_frame = ttk.Frame(progress_frame)
        time_frame.pack(fill=tk.X, pady=15)
        ttk.Label(
            time_frame,
            text="Elapsed:",
            font=('Arial', 16)
        ).pack(side=tk.LEFT)
        self.elapsed_label = ttk.Label(
            time_frame,
            text="00:00:00",
            font=('Arial', 16, 'bold'),
            foreground='#2b2b2b'
        )
        self.elapsed_label.pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(
            time_frame,
            text="Remaining:",
            font=('Arial', 16)
        ).pack(side=tk.LEFT)
        self.remaining_label = ttk.Label(
            time_frame,
            text="00:00:00",
            font=('Arial', 16, 'bold'),
            foreground='#2b2b2b'
        )
        self.remaining_label.pack(side=tk.LEFT)
        device_frame = ttk.Frame(progress_frame)
        device_frame.pack(fill=tk.X, pady=(20, 15))
        try:
            dev = sd.query_devices(self.engine.device_id)
            device_text = f"Current Device: {dev['name']} | Sample Rate: {self.engine.sr}Hz"
            ttk.Label(
                device_frame,
                text=device_text,
                font=('Arial', 16),
                anchor=tk.E
            ).pack(side=tk.RIGHT)
        except:
            ttk.Label(
                device_frame,
                text="⚠ Device Info Unavailable",
                font=('Arial', 16),
                foreground="red",
                anchor=tk.E
            ).pack(side=tk.RIGHT)
        self.status_bar = ttk.Frame(self)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)
        self.status_var = tk.StringVar(value="✔ System Ready")
        ttk.Label(
            self.status_bar,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=('Arial', 16),
            padding=5
        ).pack(fill=tk.X)

    def _start(self):
        phase = self.phase_var.get()
        preset = self.preset_var.get()
        if not phase:
            messagebox.showwarning("Warning", "Please select a burn-in phase first")
            return
        phase_config = PHASE_PRESETS[preset][phase]
        confirm_msg = (
            f"Starting [{phase}] Phase\n\n"
            f"• Preset: {preset}\n"
            f"• Estimated Duration: {self._format_time(phase_config['duration'])}\n"
            f"• Description: {phase_config['desc']}\n\n"
            "Please confirm:\n"
            "1. Headphones are properly connected\n"
            "2. Volume is set to about 50%"
        )
        if not messagebox.askyesno("Confirm Start", confirm_msg):
            return
        # Disable phase selection radio buttons
        for radio in self.phase_radios:
            radio.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_var.set("▶ Starting burn-in phase...")
        self.update()
        Thread(
            target=self._run_safe,
            args=(phase, preset),
            daemon=True
        ).start()

    def _run_safe(self, phase, preset):
        try:
            self.engine.run_phase(phase, preset)
        except Exception as e:
            messagebox.showerror("Run Error", f"Error during burn-in:\n{str(e)}")
        finally:
            self._update_ui_state()

    def _update_ui_state(self):
        # Re-enable phase selection radio buttons
        for radio in self.phase_radios:
            radio.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="Pause")
        self.btn_stop.config(state=tk.DISABLED)
        self.status_var.set("■ Burn-in Completed")

    def _toggle_pause(self):
        if not self.engine.is_running:
            return
        if self.engine.is_paused:
            self.engine.is_paused = False
            self.btn_pause.config(text="Pause")
            self.status_var.set("▶ Running")
        else:
            self.engine.is_paused = True
            self.btn_pause.config(text="Resume")
            self.status_var.set("⏸ Paused")

    def _stop(self):
        if messagebox.askyesno("Confirm Stop", "Are you sure you want to stop the burn-in process?"):
            self.engine.stop_requested = True
            self.status_var.set("⏹ Stopping...")
            # Re-enable phase selection radio buttons when stopping
            for radio in self.phase_radios:
                radio.config(state=tk.NORMAL)

    def _test_audio(self):
        try:
            if not self.engine._check_device_ready():
                self.engine.device_id = self.engine._init_audio_device()
                if self.engine.device_id is None:
                    raise RuntimeError("Cannot initialize audio device")
            self.engine._play_notification(440, 1.0)
            messagebox.showinfo(
                "Test Successful",
                "Played 440Hz test tone\n\n"
                "Please confirm:\n"
                "1. Headphones produce sound\n"
                "2. Left and right channels are balanced\n"
                "3. No noise or distortion"
            )
        except Exception as e:
            messagebox.showerror(
                "Test Failed",
                f"Audio output test failed:\n{str(e)}\n\n"
                "Suggestions:\n"
                "1. Check headphone connection\n"
                "2. Verify system audio settings\n"
                "3. Ensure no other programs are using the audio device"
            )

    def update_progress(self, progress, phase_name):
        phase_config = PHASE_PRESETS[self.preset_var.get()][phase_name]
        total_time = phase_config['duration']
        elapsed = int(progress * total_time)
        remaining = max(0, total_time - elapsed)
        self.phase_label.config(text=phase_name)
        self.progress['value'] = progress * 100
        self.elapsed_label.config(text=self._format_time(elapsed))
        self.remaining_label.config(text=self._format_time(remaining))
        if progress >= 0.95:
            status = "✔ Nearly Complete"
        elif progress >= 0.7:
            status = "▶ Running Smoothly"
        else:
            status = "▶ Running"
        self.status_var.set(status)

    def show_temp_warning(self):
        self.status_var.set("⚠ Warning: Please check earpad temperature!")
        self.after(5000, lambda: self.status_var.set("▶ Running"))

    def _format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _safe_exit(self):
        if self.engine.is_running:
            if messagebox.askyesno(
                    "Confirm Exit",
                    "Burn-in process is still running. Are you sure you want to exit?\n\n"
                    "Note: Progress for the current phase will not be saved"
            ):
                self.engine.stop_requested = True
                self.destroy()
        else:
            self.destroy()

# --------------------- Main Program Entry ---------------------
if __name__ == "__main__":
    try:
        app = XM5BurnInApp()
        try:
            if platform.system() == 'Windows':
                app.iconbitmap(default='icon.ico')
            else:
                img = tk.PhotoImage(file='icon.png')
                app.tk.call('wm', 'iconphoto', app._w, img)
        except:
            pass
        app.mainloop()
    except Exception as e:
        logging.critical(f"Program crashed: {str(e)}")
        messagebox.showerror(
            "Fatal Error",
            f"Program terminated unexpectedly:\n\n{str(e)}\n\n"
            "Suggestions:\n"
            "1. Check audio device connection\n"
            "2. Restart the application\n"
            "3. Review log file for details"
        )
    finally:
        sd._terminate()
        logging.info("Application exited safely")
