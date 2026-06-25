import os
import sys
import ctranslate2
from transformers import AutoTokenizer
from faster_whisper import WhisperModel
import repo_indexer

class AssistantEngine:
    def __init__(self, ollama_model=None, whisper_model_size="base", whisper_device="cpu", llm_device="cpu"):
        self.whisper_model_size = whisper_model_size
        self.whisper_device = whisper_device
        self.llm_device = llm_device
        
        # Local paths
        project_root = repo_indexer.get_project_root()
        self.whisper_dir = os.path.join(project_root, "models", "whisper")
        self.llm_dir = os.path.join(project_root, "models", "llm")
        
        # 1. Initialize Whisper model locally
        print(f"[*] Loading local Whisper model '{whisper_model_size}' (Device: {whisper_device})...")
        try:
            self.whisper = WhisperModel(
                self.whisper_model_size,
                device=self.whisper_device,
                compute_type="int8",
                download_root=self.whisper_dir
            )
            print("[+] Whisper model loaded successfully.")
        except Exception as e:
            print(f"[-] Error loading Whisper model: {e}", file=sys.stderr)
            sys.exit(1)
            
        # 2. Initialize CTranslate2 Generator & Tokenizer locally
        print(f"[*] Loading local Qwen LLM model from '{self.llm_dir}' (Device: {llm_device})...")
        try:
            self.generator = ctranslate2.Generator(self.llm_dir, device=self.llm_device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_dir)
            print("[+] Local Qwen LLM loaded successfully.")
        except Exception as e:
            print(f"[-] Error loading local Qwen LLM: {e}", file=sys.stderr)
            print("[-] Please run 'python download_models.py' first to download the model files.", file=sys.stderr)
            sys.exit(1)
        
        # Ingest project files & style context (Docs only by default for speed on CPU)
        print("[*] Indexing face-registry codebase (docs only) and style context...")
        self.codebase_context = repo_indexer.index_codebase(include_code=False)
        self.style_context = repo_indexer.load_style_template()
        print(f"[+] Context ready. Codebase size: {len(self.codebase_context)} chars. Style template size: {len(self.style_context)} chars.")
        
        # System prompt definition
        self.system_prompt = self._build_system_prompt()
        
        # Conversation history
        self.history = []

    def _build_system_prompt(self):
        return f"""Você é o Jarvis Interview, um copiloto de inteligência artificial rodando localmente. O seu papel é auxiliar o candidato a responder perguntas técnicas de entrevista sobre o projeto "Face Registry" em tempo real.

INSTRUÇÕES CRÍTICAS DE ESTILO E TOM:
1. Fale EXATAMENTE com o mesmo tom, estilo, vocabulário e estrutura do candidato. Use o arquivo 'interview-example.md' abaixo como guia de referência absoluto para o seu estilo de escrita.
2. Evite rodeios, introduções ou "Claro, vou ajudar". Vá DIRETO ao ponto técnico.
3. Forneça respostas CURTAS e estruturadas em tópicos (talking points) fáceis de ler. O candidato precisa bater o olho na tela e conseguir falar a resposta de forma fluida em menos de 1 minuto.
4. Fale na primeira pessoa do singular ("Eu escolhi...", "Eu utilizei..."), assumindo o papel do desenvolvedor do projeto.

---
REFERÊNCIA DE ESTILO (interview-example.md):
{self.style_context}

---
CONTEXTO COMPLETO DO PROJETO (código fonte e documentações do repositório face-registry):
{self.codebase_context}
"""

    def transcribe(self, wav_path):
        """Transcribes audio file using local Whisper model."""
        if not os.path.exists(wav_path):
            return ""
            
        # Transcribe with Portuguese hint
        segments, info = self.whisper.transcribe(wav_path, language="pt", beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        return text

    def generate_answer(self, question_text):
        """Generates suggested answer using local Qwen model based on codebase context and style."""
        if not question_text:
            return "Nenhuma fala detectada."
            
        # Limit history to the last 6 messages (3 turns) to keep context and inference window fast
        if len(self.history) > 6:
            self.history = self.history[-6:]
            
        messages = [
            {"role": "system", "content": self.system_prompt},
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
            
            # Generate response
            results = self.generator.generate_batch(
                [tokens],
                max_length=512,
                sampling_topk=20,
                sampling_temperature=0.7
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

    def regenerate_last_answer(self):
        """Regenerates the answer to the last question."""
        if len(self.history) < 2:
            return "Nenhuma resposta anterior para regenerar."
            
        # Get the last user question and pop the previous turn
        self.history.pop() # Remove last assistant answer
        last_question = self.history.pop() # Get and remove last user question
        
        # Add a subtle hint to try another phrasing/variation
        prompt_with_variation = last_question["content"] + "\n(Forneça uma variação diferente de resposta ou detalhe outro ponto da implementação)."
        
        print("[*] Regenerating last answer with variation...")
        return self.generate_answer(prompt_with_variation)

    def clear_history(self):
        self.history.clear()
        print("[+] Conversation history cleared.")
