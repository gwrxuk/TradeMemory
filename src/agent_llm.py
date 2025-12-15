import numpy as np
import json
import os
from openai import OpenAI

class LLMTradingAgent:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None):
        self.model_name = model_name
        # Use provided key or fall back to environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            print("Warning: No OpenAI API Key found. Agent will run in mock mode.")
            self.client = None
        
    def get_action(self, state, retrieved_context, info=None):
        """
        Generates an action using an LLM based on state and retrieved memory.
        """
        
        # 1. Format the Prompt
        prompt = self._construct_prompt(state, retrieved_context)
        
        # 2. Call LLM
        if self.client:
            try:
                response = self._call_llm(prompt)
            except Exception as e:
                print(f"LLM Call failed: {e}. Falling back to mock.")
                response = self._mock_llm_response(prompt)
        else:
            response = self._mock_llm_response(prompt)
        
        # 3. Parse Response
        action = self._parse_response(response)
        
        return action, 0.0, 0.0

    def _construct_prompt(self, state, retrieved_context):
        """
        Creates a text prompt combining current state and retrieved examples.
        """
        # Simplify state for text representation
        # state structure: [Close_History..., Position, Balance]
        # Let's extract just the last few close prices for the prompt
        # Assuming window size is half of state length (minus pos/bal)
        
        # Approximate parsing of state vector for display
        prices = state[:-2] # roughly
        position = state[-2]
        balance = state[-1]
        
        prompt = "You are an expert cryptocurrency trading agent. Your goal is to maximize profit.\n\n"
        
        prompt += "### Current Market Situation:\n"
        prompt += f"Recent Normalized Prices: {np.round(prices[-5:], 4)}\n" # Show last 5
        prompt += f"Current Position: {'Long' if position > 0 else 'None'}\n"
        prompt += f"Current Balance: {balance:.2f}\n\n"
        
        prompt += "### Relevant Past Experiences (Memory):\n"
        prompt += "Here are outcomes from similar past market states (Action 1=Buy, 2=Sell, 0=Hold, Reward=Profit):\n"
        
        # retrieved_context is a flat vector of [Action, Reward] pairs from k neighbors
        # Let's parse it back to be readable
        # shape is (k * 2)
        try:
            k = len(retrieved_context) // 2
            for i in range(k):
                a = int(retrieved_context[i*2])
                r = retrieved_context[i*2+1]
                act_str = ["HOLD", "BUY", "SELL"][a] if a < 3 else "UNKNOWN"
                prompt += f"- Example {i+1}: Action {act_str} -> Reward {r:.2f}\n"
        except:
            prompt += "(No valid memory context available yet)\n"
            
        prompt += "\n### Instruction:\n"
        prompt += "Based on the similar past experiences and the current situation, determine the best action to maximize reward.\n"
        prompt += "Available Actions: 0 (HOLD), 1 (BUY), 2 (SELL).\n"
        prompt += "Reply ONLY with the single digit of the action (0, 1, or 2). do not include any other text."
        
        return prompt

    def _mock_llm_response(self, prompt):
        """
        Simulates an LLM response for testing without an API key.
        """
        import random
        return str(random.choice([0, 1, 2]))

    def _call_llm(self, prompt):
        """
        Actual call to OpenAI API
        """
        completion = self.client.chat.completions.create(
          model=self.model_name,
          messages=[
            {"role": "system", "content": "You are a crypto trading bot. Reply only with 0, 1, or 2."},
            {"role": "user", "content": prompt}
          ],
          temperature=0.1 # Low temp for deterministic actions
        )
        return completion.choices[0].message.content

    def _parse_response(self, response_text):
        try:
            # tailored to look for single digit
            text = response_text.strip()
            if '0' in text: return 0
            if '1' in text: return 1
            if '2' in text: return 2
            return 0 # Default to Hold
        except:
            return 0

    def update(self, rollouts):
        return 0.0
