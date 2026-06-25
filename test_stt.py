import os
import sys
import io
import unittest
import urllib.request

# Force stdout/stderr to utf-8 with line buffering to avoid encoding errors with emojis on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

from assistant_engine import AssistantEngine

class TestSTT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = "https://huggingface.co/Xerror/XTTS-v2/resolve/main/samples/pt_sample.wav"
        cls.dest = "pt_sample_temp.wav"
        
        # Download the file if it doesn't exist
        if not os.path.exists(cls.dest):
            print(f"\n[*] Downloading test audio from {cls.url}...")
            urllib.request.urlretrieve(cls.url, cls.dest)
            print(f"[+] Download complete! File size: {os.path.getsize(cls.dest)} bytes")
        else:
            print(f"\n[+] Using existing test audio: {cls.dest} ({os.path.getsize(cls.dest)} bytes)")
            
        cls.engine = AssistantEngine(
            whisper_model_size="base", 
            whisper_device="auto", 
            llm_device="auto"
        )

    def test_stt_transcription_and_response(self):
        print("\n[*] Running STT Transcription Validation...")
        
        # 1. Transcribe the audio
        transcription = self.engine.transcribe(self.dest)
        print(f"[+] Transcription result: '{transcription}'")
        
        # Assert transcription text is correct
        self.assertTrue(len(transcription) > 0, "Transcription is empty!")
        
        # Normalize and check for key words in the transcription
        normalized_text = transcription.lower()
        self.assertTrue(
            "seis anos" in normalized_text or "6 anos" in normalized_text,
            f"Expected 'seis anos' or '6 anos' in transcription, got: '{transcription}'"
        )
        self.assertTrue(
            "imagem" in normalized_text,
            f"Expected 'imagem' in transcription, got: '{transcription}'"
        )
        
        # 2. Test LLM answer generation based on the transcription
        print("[*] Generating LLM Answer based on transcription...")
        answer = self.engine.generate_answer(transcription)
        print(f"[+] Generated answer:\n{answer}\n")
        
        self.assertTrue(len(answer) > 0, "Generated answer is empty!")
        self.assertNotEqual(answer, "Nenhuma fala detectada.", "LLM did not receive the transcription text.")
        self.assertNotIn("Erro", answer, f"LLM generated an error response: {answer}")

if __name__ == "__main__":
    unittest.main()
