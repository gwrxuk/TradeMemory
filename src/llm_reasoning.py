import numpy as np
import os
import google.generativeai as genai

class LLMReasoningModule:
    def __init__(self, model_name="gemini-3-pro-preview", embedding_model="models/embedding-001", api_key=None):
        self.model_name = model_name
        self.embedding_model = embedding_model
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            self.client_ready = True
        else:
            print("Warning: No Google API Key. Hybrid Agent will use mock embeddings.")
            self.client_ready = False
            
    def get_strategy_embedding(self, state, retrieved_context):
        """
        Returns: (embedding_vector, reasoning_text_string)
        """
        
        if not self.client_ready:
            return np.zeros(768, dtype=np.float32), "Mock: Client not ready"
            
        prompt = self._construct_prompt(state, retrieved_context)
        
        try:
            # 2. Reasoning (Generation)
            response = self.model.generate_content(
                "You are a crypto strategy advisor. Output a brief 1-sentence strategic assessment based on this data: " + prompt
            )
            strategy_text = response.text.strip()
            
            # 3. Embedding (Representation)
            emb_result = genai.embed_content(
                model=self.embedding_model,
                content=strategy_text,
                task_type="retrieval_document"
            )
            
            full_emb = np.array(emb_result['embedding'], dtype=np.float32)
            
            return full_emb, strategy_text
            
        except Exception as e:
            print(f"Gemini LLM Error: {e}")
            return np.zeros(768, dtype=np.float32), f"Error: {str(e)}"

    def _construct_prompt(self, state, retrieved_context):
        return f"Market State Vector (last 5): {state[-5:]}. Similar Past Outcomes (Action/Reward): {retrieved_context}. What is the strategy?"
