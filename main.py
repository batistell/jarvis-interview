import os
import sys
import io
import time
import queue
import threading
import msvcrt
import numpy as np

# Suppress noisy library warnings (like missing PyTorch or Symlink warning on Windows)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Enable Virtual Terminal Processing on Windows to support ANSI escape sequences natively
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        hStdOut = kernel32.GetStdHandle(-11) # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(hStdOut, ctypes.byref(mode)):
            kernel32.SetConsoleMode(hStdOut, mode.value | 0x0004) # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass

# Force stdout/stderr to utf-8 with line buffering to avoid delays and encoding errors with emojis on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from audio_listener import AudioListener
from assistant_engine import AssistantEngine

# Initialize Rich Console for beautiful styling
console = Console()

def print_welcome_panel(llm_model, whisper_model):
    welcome_text = f"""
# 🎙️ Jarvis Interview Copilot (Local & Offline)

**Ouvindo áudio do computador...** (Microfone ignorado)

## Controles do Teclado:
* **[Enter]** : Forçar processamento imediato (corta a gravação atual e gera a resposta).
* **[R]**     : Regenerar a última resposta (tenta outra variação técnica).
* **[C]**     : Limpar o histórico da conversa.
* **[Q]**     : Sair com segurança.

---
**Modelos:** STT = `faster-whisper:{whisper_model}` | LLM = `local:{llm_model}`
**Estilo:** Alinhado com `interview-example.md`
"""
    console.print(Panel(Markdown(welcome_text), title="Jarvis v1.0", border_style="cyan"))

def main():
    # Configuration
    LLM_MODEL = "qwen2.5-1.5b-ct2"
    WHISPER_MODEL_SIZE = "large-v3"
    WHISPER_DEVICE = "auto"  # Detects and uses "cuda" if Nvidia GPU is available
    LLM_DEVICE = "auto"      # Detects and uses "cuda" if Nvidia GPU is available
    
    # Thread-safe queue for audio files
    audio_queue = queue.Queue()
    
    # Initialize Assistant Engine (Local Whisper + Local Qwen)
    print_welcome_panel(LLM_MODEL, WHISPER_MODEL_SIZE)
    assistant = AssistantEngine(
        whisper_model_size=WHISPER_MODEL_SIZE,
        whisper_device=WHISPER_DEVICE,
        llm_device=LLM_DEVICE
    )
    
    # Initialize Audio Listener
    listener = AudioListener(
        processing_queue=audio_queue,
        silence_threshold=None, # Auto-calibrated
        silence_seconds=1.5
    )
    listener.live_printed = False
    
    # Worker function for processing queue
    def process_queue_worker():
        while True:
            wav_path = audio_queue.get()
            if wav_path is None:
                break
                
            try:
                # Clear the live transcription line raw via stdout to prevent [K from printing
                with listener.lock:
                    if getattr(listener, "live_printed", False):
                        sys.stdout.write("\r\x1b[K")
                        sys.stdout.flush()
                        listener.live_printed = False
                    
                    # Print the transcribing status on the same line (no leading \n)
                    msg = "[*] Transcrevendo áudio final do recrutador..."
                    console.print(f"[bold yellow]{msg}[/bold yellow]", end="")
                    sys.stdout.flush()
                    listener.live_printed = True
                
                question_text = assistant.transcribe(wav_path)
                
                # Clear the transcribing status line raw via stdout
                with listener.lock:
                    if getattr(listener, "live_printed", False):
                        sys.stdout.write("\r\x1b[K")
                        sys.stdout.flush()
                        listener.live_printed = False
                    
                    if not question_text:
                        console.print("[dim]Aguardando próxima pergunta... [Enter] para cortar | [Q] sair[/dim]")
                        continue
                        
                    # Print the final question on the same line (no leading \n)
                    console.print(f"[bold cyan]❓ Recrutador:[/bold cyan] {question_text}")
                
                console.print("[bold yellow][*] Processando resposta técnica com modelo local...[/bold yellow]")
                
                from rich.live import Live
                panel_title = "Sugestão de Resposta (Jarvis)"
                panel = Panel("", title=panel_title, border_style="green", expand=False)
                
                with Live(panel, console=console, auto_refresh=False) as live:
                    def stream_cb(current_text):
                        live.update(Panel(Markdown(current_text), title=panel_title, border_style="green", expand=False))
                        live.refresh()
                    
                    answer = assistant.generate_answer(question_text, callback=stream_cb)
                
                console.print("\n[dim]Aguardando próxima pergunta... [Enter] para cortar | [Q] sair[/dim]")
                
            except Exception as ex:
                console.print(f"\n[bold red][-] Erro ao processar áudio: {ex}[/bold red]", style="red")
            finally:
                # Clean up temp file
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
                audio_queue.task_done()
                
    # Start worker thread
    worker_thread = threading.Thread(target=process_queue_worker, daemon=True)
    worker_thread.start()
    
    # Live transcription worker for real-time console feedback
    stop_live_transcription = threading.Event()
    
    def live_transcription_worker():
        live_printed = False
        last_text = ""
        
        while not stop_live_transcription.is_set():
            if listener.is_speech_active:
                frames = listener.get_current_frames()
                # Run live STT only if we have at least 1 second of audio
                if len(frames) > int(listener.rate / listener.chunk_size * 1.0):
                    try:
                        # Convert byte frames to float32 numpy array
                        audio_bytes = b''.join(frames)
                        audio_float32 = np.frombuffer(audio_bytes, dtype=np.float32)
                        
                        # Reshape to channels (stereo) and downmix to mono by taking the mean
                        num_samples = (len(audio_float32) // listener.channels) * listener.channels
                        audio_float32 = audio_float32[:num_samples]
                        audio_reshaped = audio_float32.reshape(-1, listener.channels)
                        mono_orig = audio_reshaped.mean(axis=1)
                        
                        # Downsample to exactly 16000Hz using linear interpolation
                        target_rate = 16000
                        if listener.rate == target_rate:
                            mono_16k = mono_orig
                        else:
                            num_target_samples = int(len(mono_orig) * target_rate / listener.rate)
                            x_orig = np.arange(len(mono_orig))
                            x_target = np.linspace(0, len(mono_orig) - 1, num_target_samples)
                            mono_16k = np.interp(x_target, x_orig, mono_orig)
                        
                        audio_float32_input = mono_16k.astype(np.float32)
                        
                        # Transcribe locally with beam_size=1 and VAD filter enabled to strip silences/noise
                        segments, info = assistant.whisper.transcribe(
                            audio_float32_input, 
                            language="pt", 
                            beam_size=1,
                            vad_filter=True
                        )
                        text = " ".join([segment.text for segment in segments]).strip()
                        
                        # Apply assistant's hallucination filter
                        text = assistant._filter_hallucinations(text)
                        
                        if text:
                            # Limit the real-time text to the last 50 characters to prevent line wrapping
                            max_preview_len = 50
                            preview_text = text
                            if len(preview_text) > max_preview_len:
                                preview_text = "..." + preview_text[-max_preview_len:]
                                
                            with listener.lock:
                                sys.stdout.write("\r\x1b[K")
                                sys.stdout.flush()
                                msg = f"[🎙️] Ouvindo recrutador: {preview_text}"
                                console.print(f"[bold yellow]{msg}[/bold yellow]", end="")
                                sys.stdout.flush()
                                last_text = text
                                live_printed = True
                                listener.live_printed = True
                    except Exception as e:
                        pass
                else:
                    # Immediate feedback when speech begins but is under 1 second of buffer
                    if not live_printed or last_text == "":
                        with listener.lock:
                            sys.stdout.write("\r\x1b[K")
                            sys.stdout.flush()
                            msg = "[🎙️] Capturando áudio do recrutador..."
                            console.print(f"[bold yellow]{msg}[/bold yellow]", end="")
                            sys.stdout.flush()
                            live_printed = True
                            listener.live_printed = True
                time.sleep(0.5)  # Check more frequently (every 0.5 seconds) for faster updates
            else:
                if live_printed:
                    with listener.lock:
                        if getattr(listener, "live_printed", False):
                            sys.stdout.write("\r\x1b[K")
                            sys.stdout.flush()
                            listener.live_printed = False
                        live_printed = False
                        last_text = ""
                time.sleep(0.1)
                
    # Start live transcription thread
    live_thread = threading.Thread(target=live_transcription_worker, daemon=True)
    live_thread.start()
    
    # Start audio listener
    listener.start()
    
    console.print("\n[bold green][+] Jarvis está pronto e em prontidão![/bold green] Pode iniciar a chamada ou reproduzir som.")
    console.print("[dim]Aguardando o recrutador começar a falar...[/dim]\n")
    
    # Background generation helper
    def run_regeneration():
        console.print("\n[bold yellow][*] Regenerando última resposta com variação...[/bold yellow]")
        
        from rich.live import Live
        panel_title = "Sugestão de Resposta (Regenerada)"
        panel = Panel("", title=panel_title, border_style="green", expand=False)
        
        with Live(panel, console=console, auto_refresh=False) as live:
            def stream_cb(current_text):
                live.update(Panel(Markdown(current_text), title=panel_title, border_style="green", expand=False))
                live.refresh()
                
            answer = assistant.regenerate_last_answer(callback=stream_cb)
            
        console.print("\n[dim]Aguardando próxima pergunta...[/dim]")

    # Main keyboard listener loop (non-blocking)
    try:
        while True:
            if msvcrt.kbhit():
                char = msvcrt.getch()
                
                if char == b'\r':  # Enter Key
                    listener.force_cut()
                elif char in (b'q', b'Q'):
                    console.print("\n[bold red][*] Parando gravador e finalizando...[/bold red]")
                    break
                elif char in (b'r', b'R'):
                    # Run in background to keep key listener active
                    threading.Thread(target=run_regeneration, daemon=True).start()
                elif char in (b'c', b'C'):
                    assistant.clear_history()
                    console.print("\n[bold green][+] Histórico de conversa limpo.[/bold green]")
                    
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        listener.stop()
        stop_live_transcription.set()
        audio_queue.put(None)  # Stop worker
        worker_thread.join(timeout=2)
        live_thread.join(timeout=2)
        console.print("[bold green][+] Copiloto Jarvis finalizado com sucesso.[/bold green]")

if __name__ == "__main__":
    main()
