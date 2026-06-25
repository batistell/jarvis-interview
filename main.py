import os
import sys
import io
import time
import queue
import threading
import msvcrt

# Force stdout/stderr to utf-8 to avoid encoding errors with emojis on Windows cmd/powershell
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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
    WHISPER_MODEL_SIZE = "base"
    WHISPER_DEVICE = "cpu"  # Change to "cuda" if Nvidia GPU is available
    LLM_DEVICE = "cpu"      # Change to "cuda" if Nvidia GPU is available
    
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
    
    # Worker function for processing queue
    def process_queue_worker():
        while True:
            wav_path = audio_queue.get()
            if wav_path is None:
                break
                
            try:
                console.print("\n[bold yellow][*] Transcrevendo áudio do entrevistador...[/bold yellow]")
                question_text = assistant.transcribe(wav_path)
                
                if not question_text:
                    continue
                    
                console.print(f"\n[bold cyan]❓ Entrevistador:[/bold cyan] {question_text}")
                console.print("[bold yellow][*] Processando resposta técnica com modelo local...[/bold yellow]")
                
                answer = assistant.generate_answer(question_text)
                
                # Render answer in a beautiful markdown panel
                console.print("\n")
                console.print(Panel(Markdown(answer), title="Sugestão de Resposta (Jarvis)", border_style="green", expand=False))
                console.print("\n[dim]Aguardando próxima pergunta... [Enter] para cortar | [R] para regerar | [C] limpar | [Q] sair[/dim]")
                
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
        import tempfile
        import wave
        import pyaudiowpatch as pyaudio
        
        live_printed = False
        last_text = ""
        
        while not stop_live_transcription.is_set():
            if listener.is_speech_active:
                frames = listener.get_current_frames()
                # Run live STT only if we have at least 1 second of audio
                if len(frames) > int(listener.rate / listener.chunk_size * 1.0):
                    # Save frames to a temp live WAV
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    temp_filepath = temp_file.name
                    temp_file.close()
                    
                    try:
                        wf = wave.open(temp_filepath, 'wb')
                        wf.setnchannels(listener.channels)
                        wf.setsampwidth(listener.p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(listener.rate)
                        wf.writeframes(b''.join(frames))
                        wf.close()
                        
                        # Transcribe locally with beam_size=1 for instant performance
                        segments, info = assistant.whisper.transcribe(temp_filepath, language="pt", beam_size=1)
                        text = " ".join([segment.text for segment in segments]).strip()
                        
                        if text and text != last_text:
                            # Print on the same line using \r and clearing to the end of the line
                            console.print(f"\r\x1b[K[bold yellow][🎙️] Ouvindo:[/bold yellow] {text}", end="")
                            last_text = text
                            live_printed = True
                    except Exception:
                        pass
                    finally:
                        try:
                            os.remove(temp_filepath)
                        except Exception:
                            pass
                time.sleep(0.8)  # Check every 0.8 seconds during speech
            else:
                if live_printed:
                    # Clear the live line when speech ends
                    console.print("\r\x1b[K", end="")
                    live_printed = False
                    last_text = ""
                time.sleep(0.1)
                
    # Start live transcription thread
    live_thread = threading.Thread(target=live_transcription_worker, daemon=True)
    live_thread.start()
    
    # Start audio listener
    listener.start()
    
    # Background generation helper
    def run_regeneration():
        console.print("\n[bold yellow][*] Regenerando última resposta com variação...[/bold yellow]")
        answer = assistant.regenerate_last_answer()
        console.print("\n")
        console.print(Panel(Markdown(answer), title="Sugestão de Resposta (Regenerada)", border_style="green", expand=False))
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
