import os
import sys
import time
import math
import wave
import tempfile
import threading
import numpy as np
import pyaudiowpatch as pyaudio

class AudioListener:
    def __init__(self, processing_queue, silence_threshold=None, silence_seconds=1.5, chunk_size=1024):
        self.processing_queue = processing_queue
        self.silence_seconds = silence_seconds
        self.chunk_size = chunk_size
        
        self.p = pyaudio.PyAudio()
        self.loopback_device = self._find_loopback_device()
        
        if not self.loopback_device:
            print("[-] Error: No WASAPI loopback device found. Cannot record system audio.", file=sys.stderr)
            self.p.terminate()
            sys.exit(1)
            
        self.rate = int(self.loopback_device["defaultSampleRate"])
        self.channels = int(self.loopback_device["maxInputChannels"])
        self.device_index = int(self.loopback_device["index"])
        
        print(f"[+] Using loopback device: {self.loopback_device['name']}")
        print(f"[+] Audio Format: {self.channels} channels, {self.rate}Hz")
        
        # Audio accumulator
        self.accumulated_frames = []
        self.is_recording = False
        self.is_speech_active = False
        self.silence_start_time = None
        self.lock = threading.Lock()
        
        # Silence threshold calibration
        self.silence_threshold = silence_threshold
        if self.silence_threshold is None:
            self._calibrate_threshold()
            
        self.thread = None
        self.stop_event = threading.Event()

    def _find_loopback_device(self):
        try:
            wasapi_info = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("[-] WASAPI is not available on this system.", file=sys.stderr)
            return None
            
        default_speakers = self.p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        
        # Search for speakers loopback
        loopback_device = None
        for loopback in self.p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                loopback_device = loopback
                break
                
        if not loopback_device:
            # Fallback 1: Get any loopback device
            for loopback in self.p.get_loopback_device_info_generator():
                loopback_device = loopback
                break
                
        if not loopback_device:
            # Fallback 2: Look manually in all devices
            for i in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(i)
                if info.get("isLoopbackDevice") or "loopback" in info["name"].lower():
                    loopback_device = info
                    break
                    
        return loopback_device

    def _calculate_rms(self, frame_bytes):
        # Convert bytes to numpy array to calculate RMS
        data = np.frombuffer(frame_bytes, dtype=np.int16)
        if len(data) == 0:
            return 0
        return np.sqrt(np.mean(data.astype(np.float64)**2))

    def _calibrate_threshold(self, duration_sec=1.5):
        print(f"[*] Calibrating silence threshold for {duration_sec} seconds... Please do not play audio.")
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size
        )
        
        rms_values = []
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            try:
                data = stream.read(self.chunk_size)
                rms = self._calculate_rms(data)
                rms_values.append(rms)
            except IOError:
                continue
                
        stream.close()
        
        if rms_values:
            avg_rms = np.mean(rms_values)
            max_rms = np.max(rms_values)
            # Set threshold slightly above the max observed noise floor to avoid false triggers
            self.silence_threshold = max(max_rms * 1.5, avg_rms + 200, 350.0)
            print(f"[+] Calibration complete. Noise Floor Avg: {avg_rms:.1f}, Max: {max_rms:.1f}. Silence Threshold set to: {self.silence_threshold:.1f}")
        else:
            self.silence_threshold = 400.0
            print(f"[!] Calibration failed. Set default Silence Threshold to: {self.silence_threshold}")

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        print("[+] Audio Listener background thread started.")

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.p.terminate()
        print("[+] Audio Listener stopped.")

    def get_current_frames(self):
        with self.lock:
            return list(self.accumulated_frames)

    def force_cut(self):
        with self.lock:
            if len(self.accumulated_frames) > 0:
                print("\n[*] [Action] Hitting cut point manually. Processing current segment...")
                self._save_and_queue_segment()
                self.is_speech_active = False
                self.silence_start_time = None

    def _save_and_queue_segment(self):
        if not self.accumulated_frames:
            return
            
        frames_to_save = list(self.accumulated_frames)
        self.accumulated_frames.clear()
        
        # Save to a thread-safe unique temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_filepath = temp_file.name
        temp_file.close()
        
        try:
            wf = wave.open(temp_filepath, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames_to_save))
            wf.close()
            
            # Push to processing queue
            self.processing_queue.put(temp_filepath)
        except Exception as e:
            print(f"[-] Error saving audio segment: {e}", file=sys.stderr)
            try:
                os.remove(temp_filepath)
            except:
                pass

    def _record_loop(self):
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size
        )
        
        print("[*] Listening to system audio...")
        
        while not self.stop_event.is_set():
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
            except IOError:
                # Overflow/Underflow error, skip chunk
                continue
                
            rms = self._calculate_rms(data)
            
            with self.lock:
                if rms > self.silence_threshold:
                    if not self.is_speech_active:
                        self.is_speech_active = True
                        # print("\n[🎙️] Speech detected...")
                    self.accumulated_frames.append(data)
                    self.silence_start_time = None
                else:
                    if self.is_speech_active:
                        self.accumulated_frames.append(data)
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()
                        elif time.time() - self.silence_start_time >= self.silence_seconds:
                            # Silence duration exceeded: Speech turn ended
                            # print("\n[⏹️] Silence detected. Turn complete.")
                            self._save_and_queue_segment()
                            self.is_speech_active = False
                            self.silence_start_time = None
                            
        stream.close()
