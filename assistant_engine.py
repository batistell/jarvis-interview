import os
import sys
from pathlib import Path

# Keep DLL references alive in memory on Windows to prevent garbage collection
_dll_handles = []

if sys.platform == "win32":
    # Register NVIDIA CUDA DLLs from both our site-packages and the sister jarvis project if available
    site_packages_our = Path(sys.prefix) / "Lib" / "site-packages"
    site_packages_ref = Path("o:/Python/jarvis/.venv/Lib/site-packages")
    
    for site_pkg in [site_packages_our, site_packages_ref]:
        nvidia_dir = site_pkg / "nvidia"
        if nvidia_dir.exists():
            for bin_dir in nvidia_dir.glob("**/bin"):
                try:
                    resolved_path = str(bin_dir.resolve())
                    # Add to PATH so C++ binding DLLs (ctranslate2) can find them
                    os.environ["PATH"] = resolved_path + os.pathsep + os.environ["PATH"]
                    # Register DLL directory for Python 3.8+
                    handle = os.add_dll_directory(resolved_path)
                    _dll_handles.append(handle)
                except Exception:
                    pass

import ctranslate2
from transformers import AutoTokenizer
from faster_whisper import WhisperModel
import repo_indexer

class AssistantEngine:
    def __init__(self, ollama_model=None, whisper_model_size="base", whisper_device="auto", llm_device="auto"):
        self.whisper_model_size = whisper_model_size
        
        # Local paths
        project_root = repo_indexer.get_project_root()
        self.whisper_dir = os.path.join(project_root, "models", "whisper")
        self.llm_dir = os.path.join(project_root, "models", "llm")
        
        # Auto-detect CUDA availability
        cuda_available = ctranslate2.get_cuda_device_count() > 0
        
        # Set device dynamically if auto is selected
        if whisper_device == "auto":
            self.whisper_device = "cuda" if cuda_available else "cpu"
        else:
            self.whisper_device = whisper_device
            
        if llm_device == "auto":
            self.llm_device = "cuda" if cuda_available else "cpu"
        else:
            self.llm_device = llm_device
            
        # Set optimal compute type based on device
        self.whisper_compute = "int8_float16" if self.whisper_device == "cuda" else "int8"
        self.llm_compute = "int8_float16" if self.llm_device == "cuda" else "int8"
        
        print(f"[+] Device Selection - STT: {self.whisper_device} ({self.whisper_compute}) | LLM: {self.llm_device} ({self.llm_compute})")
        
        # 1. Initialize Whisper model locally
        print(f"[*] Loading local Whisper model '{whisper_model_size}'...")
        self.whisper = None
        
        # Try CUDA first if requested/available
        if self.whisper_device == "cuda":
            try:
                print("[*] Trying to load Whisper on GPU (CUDA)...")
                self.whisper = WhisperModel(
                    self.whisper_model_size,
                    device="cuda",
                    compute_type=self.whisper_compute,
                    download_root=self.whisper_dir
                )
                print("[+] Whisper model loaded on GPU.")
                
                # Warm up Whisper on GPU to check if DLLs are present and compile kernels
                print("[*] Warming up Whisper model on GPU...")
                import tempfile
                import wave
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    warmup_wav = temp_file.name
                try:
                    wf = wave.open(warmup_wav, 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 16-bit (2 bytes)
                    wf.setframerate(16000)
                    wf.writeframes(b'\x00' * 16000)
                    wf.close()
                    # Run a transcribe to verify CUDA execution works (force evaluation with list)
                    list(self.whisper.transcribe(warmup_wav, language="pt", beam_size=1)[0])
                    print("[+] Whisper model warmed up on GPU successfully.")
                finally:
                    try:
                        os.remove(warmup_wav)
                    except:
                        pass
            except Exception as e:
                print(f"[!] Warning: Failed to load or warm up Whisper on GPU: {e}. Falling back to CPU.", file=sys.stderr)
                self.whisper = None
                self.whisper_device = "cpu"
                self.whisper_compute = "int8"
                
        # Load on CPU if CUDA was not requested, not available, or failed
        if self.whisper is None:
            print("[*] Loading Whisper model on CPU (device='cpu', compute_type='int8')...")
            try:
                self.whisper = WhisperModel(
                    self.whisper_model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=self.whisper_dir
                )
                print("[+] Whisper model loaded on CPU successfully.")
                
                # Warm up on CPU to ensure it's ready
                print("[*] Warming up Whisper model on CPU...")
                import tempfile
                import wave
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    warmup_wav = temp_file.name
                try:
                    wf = wave.open(warmup_wav, 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(b'\x00' * 16000)
                    wf.close()
                    # Force evaluation with list
                    list(self.whisper.transcribe(warmup_wav, language="pt", beam_size=1)[0])
                    print("[+] Whisper model warmed up on CPU successfully.")
                finally:
                    try:
                        os.remove(warmup_wav)
                    except:
                        pass
            except Exception as e:
                print(f"[-] Error loading Whisper model on CPU: {e}", file=sys.stderr)
                sys.exit(1)
            
        # LLM placeholders
        self.generator = None
        self.tokenizer = None
        self.codebase_docs = None
        self.style_context = None
        self.history = []
        self.interrupt_requested = False
        import threading
        self.generation_lock = threading.Lock()
        
        # Load LLM model eagerly at startup
        self._load_llm()

    def _build_system_prompt(self, codebase_context):
        return f"""Você é o Jarvis, um copiloto de inteligência artificial. O seu papel é auxiliar o candidato a responder perguntas técnicas de entrevista sobre o projeto "Face Registry" em tempo real.

INSTRUÇÕES DE TOM E ESTILO:
1. Vá DIRETO ao ponto técnico. Não faça introduções, não dê saudações, nem use frases como "Claro, vou ajudar".
2. Responda em Português na primeira pessoa do singular ("Eu escolhi...", "Eu utilizei...", "No meu projeto..."), como o desenvolvedor da solução.
3. Forneça uma resposta PEQUENA, contínua e explicada de forma falada (como em uma conversa). Evite tópicos ou bullet points. A resposta deve ser fluida e direta para ser falada em menos de 1 minuto.

CONTEXTO RELEVANTE DO PROJETO:
{codebase_context}
"""

    def _filter_hallucinations(self, text: str) -> str:
        """Filtra alucinações ou preenchimentos comuns do Whisper causados por ruído ou silêncio."""
        cleaned = text.strip().lower()
        # Remove pontuação comum para normalizar a comparação
        for p in [".", ",", "!", "?", "-", '"', "'", "(", ")", "[", "]", "{", "}", "!", " "]:
            cleaned = cleaned.replace(p, "")
        cleaned = cleaned.strip()

        # Alucinações típicas do Whisper sob ruído ou silêncio
        hallucinations = {
            "thankyou", "thankyouverymuch", "thankyouforwatching", "thanksforwatching",
            "subtitlesbyamaraorg", "subtitles", "amaraorg", "you", "ha", "yeah", "bye",
            "please", "ok", "right", "obrigado", "obrigada", "obrigadoporassistir",
            "muitoobrigado", "tchau", "sweetheart"
        }

        # Normalize comparison by removing spaces too
        cleaned_no_spaces = cleaned.replace(" ", "")
        
        # Discard static noise containing known Whisper hallucination words
        if "screwdriver" in cleaned_no_spaces or "sweetheart" in cleaned_no_spaces:
            return ""
            
        if cleaned_no_spaces in hallucinations:
            return ""

        # Ignora ruídos que resultam em strings extremamente curtas
        if len(cleaned) < 2:
            return ""

        return text

    def transcribe(self, wav_path_or_ndarray):
        """Transcribes audio file or numpy array using local Whisper model."""
        if isinstance(wav_path_or_ndarray, str):
            if not os.path.exists(wav_path_or_ndarray):
                return ""
            
        # Transcribe with Portuguese hint and VAD filter enabled to strip silences/noise
        segments, info = self.whisper.transcribe(
            wav_path_or_ndarray, 
            language="pt", 
            beam_size=1,
            vad_filter=True
        )
        text = " ".join([segment.text for segment in segments]).strip()
        
        # Filter out hallucinations
        return self._filter_hallucinations(text)

    def _load_llm(self):
        """Loads local Qwen LLM model and indexes the codebase context lazily."""
        if self.generator is not None:
            return
            
        import ctranslate2
        from transformers import AutoTokenizer
        
        print(f"[*] Loading local Qwen LLM model from '{self.llm_dir}'...")
        
        # Try CUDA first if requested/available
        if self.llm_device == "cuda":
            try:
                print("[*] Trying to load Qwen on GPU (CUDA)...")
                self.generator = ctranslate2.Generator(
                    self.llm_dir, 
                    device="cuda",
                    compute_type=self.llm_compute
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_dir)
                print("[+] Qwen model loaded on GPU.")
                
                # Warm up Qwen on GPU to verify CUDA execution works
                print("[*] Warming up Qwen model on GPU...")
                warmup_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode("<|im_start|>system\nWarmup<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"))
                self.generator.generate_batch([warmup_tokens], max_length=5)
                print("[+] Qwen model warmed up on GPU successfully.")
            except Exception as e:
                print(f"[!] Warning: Failed to load or warm up Qwen on GPU: {e}. Falling back to CPU.", file=sys.stderr)
                self.generator = None
                self.llm_device = "cpu"
                self.llm_compute = "int8"
                
        # Load on CPU if CUDA was not requested, not available, or failed
        if self.generator is None:
            print("[*] Loading Qwen model on CPU (device='cpu', compute_type='int8')...")
            try:
                self.generator = ctranslate2.Generator(
                    self.llm_dir, 
                    device="cpu",
                    compute_type="int8"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_dir)
                print("[+] Qwen model loaded on CPU successfully.")
                
                # Warm up Qwen on CPU
                print("[*] Warming up Qwen model on CPU...")
                warmup_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode("<|im_start|>system\nWarmup<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"))
                self.generator.generate_batch([warmup_tokens], max_length=5)
                print("[+] Qwen model warmed up on CPU successfully.")
            except Exception as e:
                print(f"[-] Error loading local Qwen LLM on CPU: {e}", file=sys.stderr)
                print("[-] Please run 'python download_models.py' first to download the model files.", file=sys.stderr)
                sys.exit(1)
        
        # Ingest project files as a dictionary for RAG retrieval
        print("[*] Indexing face-registry codebase (docs only)...")
        self.codebase_docs = repo_indexer.index_codebase_as_dict(include_code=False)
        print(f"[+] Context ready. Indexed {len(self.codebase_docs)} documents.")

    def _retrieve_relevant_context(self, question, max_chars=25000):
        """Retorna os documentos mais relevantes do projeto com base nas palavras-chave da pergunta (Mini-RAG)."""
        stopwords = {
            "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "na", "no", 
            "uma", "os", "as", "se", "por", "como", "mais", "ao", "aos", "como", "foi", 
            "das", "dos", "qual", "quais", "como", "por que", "onde", "quando", "quem", "você"
        }
        # Tokenize question words
        q_words = set(w for w in question.lower().split() if w.isalnum() and w not in stopwords)
        
        if not q_words or not self.codebase_docs:
            readme = self.codebase_docs.get("README.md", "") if self.codebase_docs else ""
            return f"=== FILE: README.md ===\n{readme}\n"
            
        scores = []
        for path, content in self.codebase_docs.items():
            content_lower = content.lower()
            raw_score = sum(content_lower.count(word) for word in q_words)
            # Length normalization (prioritize focused docs over huge files)
            score = raw_score / (len(content) ** 0.5) if len(content) > 0 else 0
            scores.append((score, raw_score, path))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        
        selected_docs = []
        current_chars = 0
        
        # Always prioritize README.md as the core project entry context
        if "README.md" in self.codebase_docs:
            readme_content = self.codebase_docs["README.md"]
            selected_docs.append(f"=== FILE: README.md ===\n{readme_content}\n")
            current_chars += len(readme_content)
            
        for score, raw, path in scores:
            if path == "README.md":
                continue
            if raw == 0:
                continue # Ignore completely irrelevant files
                
            doc_str = f"=== FILE: {path} ===\n{self.codebase_docs[path]}\n"
            # Knapsack packing
            if current_chars + len(doc_str) <= max_chars:
                selected_docs.append(doc_str)
                current_chars += len(doc_str)
            else:
                continue # Try to fit other smaller files
                
        return "\n".join(selected_docs)

    def generate_answer(self, question_text, callback=None):
        """Generates suggested answer using local Qwen model based on codebase context and style."""
        if not question_text:
            return "Nenhuma fala detectada."
            
        self._load_llm()
        
        # Retrieve the relevant context dynamically (RAG)
        codebase_context = self._retrieve_relevant_context(question_text)
        system_prompt = self._build_system_prompt(codebase_context)
            
        # Limit history to the last 6 messages (3 turns) to keep context and inference window fast
        if len(self.history) > 6:
            self.history = self.history[-6:]
            
        messages = [
            {"role": "system", "content": system_prompt},
            *self.history,
            {"role": "user", "content": question_text}
        ]
        
        try:
            # Format chat prompt using Jinja template from tokenizer
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            # Tokenize input text to tokens
            tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(formatted_prompt))
            
            accumulated_ids = []
            
            def ctranslate2_callback(step_result):
                accumulated_ids.append(step_result.token_id)
                if callback:
                    # Decode accumulated tokens so far and pass to the callback
                    current_text = self.tokenizer.decode(accumulated_ids, skip_special_tokens=True).strip()
                    callback(current_text)
                return False
            
            # Generate response
            results = self.generator.generate_batch(
                [tokens],
                max_length=512,
                include_prompt_in_result=False,
                sampling_topk=20,
                sampling_temperature=0.7,
                callback=ctranslate2_callback if callback else None
            )
            
            # Decode generated output tokens
            output_tokens = results[0].sequences_ids[0]
            answer_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            
            # Store in history
            self.history.append({"role": "user", "content": question_text})
            self.history.append({"role": "assistant", "content": answer_text})
            
            return answer_text
        except Exception as e:
            return f"Erro ao gerar resposta com LLM local: {e}"

    def regenerate_last_answer(self, callback=None):
        """Regenerates the answer to the last question."""
        if len(self.history) < 2:
            return "Nenhuma resposta anterior para regenerar."
            
        # Get the last user question and pop the previous turn
        self.history.pop() # Remove last assistant answer
        last_question = self.history.pop() # Get and remove last user question
        
        # Add a subtle hint to try another phrasing/variation
        prompt_with_variation = last_question["content"] + "\n(Forneça uma variação diferente de resposta ou detalhe outro ponto da implementação)."
        
        print("[*] Regenerating last answer with variation...")
        return self.generate_answer(prompt_with_variation, callback=callback)

    def clear_history(self):
        self.history.clear()
        print("[+] Conversation history cleared.")
