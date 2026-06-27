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
    def __init__(self, processing_queue, silence_threshold=None, silence_seconds=1.5, chunk_size=1024, device_mode="loopback"):
        self.processing_queue = processing_queue
        self.silence_seconds = silence_seconds
        self.chunk_size = chunk_size
        self.device_mode = device_mode
        
        self.p = pyaudio.PyAudio()
        
        if self.device_mode == "loopback":
            self.device_info = self._find_loopback_device()
            if not self.device_info:
                print("[-] Error: No WASAPI loopback device found. Cannot record system audio.", file=sys.stderr)
                self.p.terminate()
                sys.exit(1)
            print(f"[+] Using loopback device: {self.device_info['name']}")
        else:
            try:
                self.device_info = self.p.get_default_input_device_info()
            except OSError:
                self.device_info = None
                for i in range(self.p.get_device_count()):
                    info = self.p.get_device_info_by_index(i)
                    if info["maxInputChannels"] > 0 and not info.get("isLoopbackDevice", False):
                        self.device_info = info
                        break
            if not self.device_info:
                print("[-] Error: No input device (microphone) found. Cannot record audio.", file=sys.stderr)
                self.p.terminate()
                sys.exit(1)
            print(f"[+] Using microphone device: {self.device_info['name']}")
            
        self.rate = int(self.device_info["defaultSampleRate"])
        self.channels = int(self.device_info["maxInputChannels"])
        self.device_index = int(self.device_info["index"])
        
        print(f"[+] Audio Format: {self.channels} channels, {self.rate}Hz")
        
        # Audio accumulator
        self.accumulated_frames = []
        self.is_recording = False
        self.is_speech_active = False
        self.silence_start_time = None
        self.speech_chunks_count = 0
        self.current_rms = 0.0
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
        # Convert bytes to numpy float32 array to calculate RMS
        data = np.frombuffer(frame_bytes, dtype=np.float32)
        if len(data) == 0:
            return 0.0
        return np.sqrt(np.mean(data.astype(np.float64)**2))

    def _calibrate_threshold(self, duration_sec=1.5):
        if self.device_mode == "loopback":
            print(f"[*] Calibrating silence threshold for {duration_sec} seconds... Please do not play audio.")
        else:
            print(f"[*] Calibrating silence threshold for {duration_sec} seconds... Please remain silent.")
        stream = self.p.open(
            format=pyaudio.paFloat32,
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
                available = stream.get_read_available()
                if available >= self.chunk_size:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    rms = self._calculate_rms(data)
                    rms_values.append(rms)
                else:
                    # Windows loopback does not feed frames when silent. Register 0.0 RMS and wait.
                    rms_values.append(0.0)
                    time.sleep(0.01)
            except IOError:
                continue
                
        stream.close()
        
        if rms_values:
            avg_rms = np.mean(rms_values)
            max_rms = np.max(rms_values)
            # Set threshold slightly above the max observed noise floor to avoid false triggers
            # On Float32, RMS is in range [0.0, 1.0]. A safe threshold is 0.025 (equiv to 800 on int16 scale)
            self.silence_threshold = max(max_rms * 1.8, avg_rms + 0.010, 0.025)
            print(f"[+] Calibration complete. Noise Floor Avg: {avg_rms:.4f}, Max: {max_rms:.4f}. Silence Threshold set to: {self.silence_threshold:.4f}")
        else:
            self.silence_threshold = 0.03
            print(f"[!] Calibration failed. Set default Silence Threshold to: {self.silence_threshold:.4f}")

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
                self.speech_chunks_count = 0

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
            # Downmix and downsample in-memory to 16kHz mono to ensure perfect STT compatibility
            audio_bytes = b''.join(frames_to_save)
            audio_float32 = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Truncate to multiple of channel size to avoid reshape error
            num_samples = (len(audio_float32) // self.channels) * self.channels
            audio_float32 = audio_float32[:num_samples]
            audio_reshaped = audio_float32.reshape(-1, self.channels)
            mono_orig = audio_reshaped.mean(axis=1)
            
            # Downsample to exactly 16000Hz using linear interpolation
            target_rate = 16000
            if self.rate == target_rate:
                mono_16k = mono_orig
            else:
                num_target_samples = int(len(mono_orig) * target_rate / self.rate)
                x_orig = np.arange(len(mono_orig))
                x_target = np.linspace(0, len(mono_orig) - 1, num_target_samples)
                mono_16k = np.interp(x_target, x_orig, mono_orig)
            
            # Convert to int16 for standard WAV file storage
            mono_16k_int16 = (mono_16k * 32767.0).clip(-32768, 32767).astype(np.int16)
            
            wf = wave.open(temp_filepath, 'wb')
            wf.setnchannels(1) # Mono
            wf.setsampwidth(2) # 16-bit (2 bytes)
            wf.setframerate(16000) # 16kHz
            wf.writeframes(mono_16k_int16.tobytes())
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
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size
        )
        
        if self.device_mode == "loopback":
            print("[*] Listening to system audio...")
        else:
            print("[*] Listening to microphone audio...")
        
        # Skip the first 0.5s of audio to discard startup pops/clicks from WASAPI device init
        startup_chunks_skipped = 0
        startup_chunks_to_skip = int(0.5 * self.rate / self.chunk_size)
        
        while not self.stop_event.is_set():
            try:
                available = stream.get_read_available()
                if available < self.chunk_size:
                    # No data available right now. Sleep briefly to not hog CPU
                    time.sleep(0.005)
                    
                    # Track silence timeout even when there is no new data from WASAPI loopback
                    with self.lock:
                        if self.is_speech_active:
                            if self.silence_start_time is None:
                                self.silence_start_time = time.time()
                            elif time.time() - self.silence_start_time >= self.silence_seconds:
                                # Silence duration exceeded: Speech turn ended
                                min_speech_duration_sec = 0.1
                                min_speech_chunks = int(min_speech_duration_sec * self.rate / self.chunk_size)
                                if self.speech_chunks_count >= min_speech_chunks:
                                    self._save_and_queue_segment()
                                else:
                                    self.accumulated_frames.clear()
                                    
                                self.is_speech_active = False
                                self.silence_start_time = None
                                self.speech_chunks_count = 0
                    continue
                
                # Read data since it's available
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                rms = self._calculate_rms(data)
            except IOError:
                # Overflow/Underflow error, skip chunk
                continue
                
            if startup_chunks_skipped < startup_chunks_to_skip:
                startup_chunks_skipped += 1
                continue
                
            self.current_rms = rms
            
            with self.lock:
                if rms > self.silence_threshold:
                    if not self.is_speech_active:
                        self.is_speech_active = True
                    self.accumulated_frames.append(data)
                    self.speech_chunks_count += 1
                    self.silence_start_time = None
                else:
                    if self.is_speech_active:
                        # Append the quiet but real data
                        self.accumulated_frames.append(data)
                        
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()
                        elif time.time() - self.silence_start_time >= self.silence_seconds:
                            # Silence duration exceeded: Speech turn ended
                            min_speech_duration_sec = 0.1
                            min_speech_chunks = int(min_speech_duration_sec * self.rate / self.chunk_size)
                            if self.speech_chunks_count >= min_speech_chunks:
                                self._save_and_queue_segment()
                            else:
                                self.accumulated_frames.clear()
                                
                            self.is_speech_active = False
                            self.silence_start_time = None
                            self.speech_chunks_count = 0
                            
        stream.close()
