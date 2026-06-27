import os
import sys
import io
import time
import unittest

# Force stdout/stderr to utf-8 with line buffering to avoid encoding errors with emojis on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

from assistant_engine import AssistantEngine, clear_gpu_memory

class TestLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clear_gpu_memory()
        print("\n[*] Initializing AssistantEngine for LLM test...")
        start_time = time.time()
        
        # Configure both STT and LLM to run on CUDA as requested
        cls.engine = AssistantEngine(
            whisper_model_size="large-v3",
            whisper_device="cuda",
            llm_device="cuda"
        )
        cls.init_time = time.time() - start_time
        print(f"[+] AssistantEngine initialized in {cls.init_time:.2f} seconds.")

    def test_llm_generation_performance(self):
        print("\n[*] Running LLM generation performance profiling...")
        
        question = "Como é a arquitetura do projeto Face Registry?"
        print(f"[*] Question: '{question}'")
        
        # Profile context retrieval (uses default max_chars=6000)
        start_rag = time.time()
        context = self.engine._retrieve_relevant_context(question)
        rag_time = time.time() - start_rag
        print(f"[+] RAG context retrieval took {rag_time:.4f} seconds.")
        print(f"[+] Context length: {len(context)} characters.")
        
        # Formulate system prompt and count tokens
        system_prompt = self.engine._build_system_prompt(context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        formatted_prompt = self.engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = self.engine.tokenizer.convert_ids_to_tokens(self.engine.tokenizer.encode(formatted_prompt))
        num_prompt_tokens = len(tokens)
        print(f"[+] Prompt token count: {num_prompt_tokens} tokens.")
        
        # Profile response generation
        tokens_generated = []
        def callback(current_text):
            # Print a dot for each step to show activity
            sys.stdout.write(".")
            sys.stdout.flush()

        print("[*] Generating response...")
        start_gen = time.time()
        # Pass a callback to see the progress
        answer = self.engine.generate_answer(question, callback=callback)
        gen_time = time.time() - start_gen
        print("\n[+] Generation completed.")
        print(f"[+] Response time: {gen_time:.2f} seconds.")
        print(f"[+] Generated answer:\n{answer}\n")
        
        # Verify response is valid
        self.assertTrue(len(answer) > 0, "Generated answer is empty!")
        self.assertNotIn("Erro ao gerar resposta", answer, f"LLM error occurred: {answer}")
        
        # Print metrics
        print(f"--- Performance Metrics ---")
        print(f"Engine Load Time:  {self.init_time:.2f} s")
        print(f"RAG Retrieval:     {rag_time:.4f} s")
        print(f"Prompt Tokens:     {num_prompt_tokens}")
        print(f"Generation Time:   {gen_time:.2f} s")
        
        # Suggest threshold
        self.assertLess(gen_time, 15.0, "Response generation is too slow! Should be under 15 seconds.")

if __name__ == "__main__":
    unittest.main()
