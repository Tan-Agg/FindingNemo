import requests
from config import NVIDIA_API_KEY, NVIDIA_API_BASE, GENERALIST_MODEL

class ModelCaller:
    """
    SINGLE RESPONSIBILITY: Call AI models and get responses
    Handles both specialist and generalist calls
    """
    
    def __init__(self):
        self.api_key = NVIDIA_API_KEY
        self.base_url = NVIDIA_API_BASE
        self.generalist_model = GENERALIST_MODEL
    
    def call_specialist(self, endpoint, user_prompt, max_tokens=500):
        """
        Call a specialist model
        
        Args:
            endpoint (str): Specialist endpoint URL or model ID
            user_prompt (str): User's query
            max_tokens (int): Max tokens in response
            
        Returns:
            dict: {
                "answer": str,
                "model": str,
                "tokens_used": int,
                "error": str (if any)
            }
        """
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": endpoint,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1
                },
                timeout=60
            )
            
            response.raise_for_status()
            data = response.json()
            
            answer = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            
            return {
                "answer": answer,
                "model": endpoint,
                "tokens_used": usage.get('total_tokens', 0),
                "error": None
            }
            
        except Exception as e:
            print(f"❌ Specialist call failed: {e}")
            return {
                "answer": None,
                "model": endpoint,
                "tokens_used": 0,
                "error": str(e)
            }
    
    def call_generalist(self, user_prompt, max_tokens=500):
        """
        Call the generalist model (253B)
        
        Args:
            user_prompt (str): User's query
            max_tokens (int): Max tokens in response
            
        Returns:
            dict: {
                "answer": str,
                "model": str,
                "tokens_used": int,
                "error": str (if any)
            }
        """
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.generalist_model,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=60
            )
            
            response.raise_for_status()
            data = response.json()
            
            answer = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            
            return {
                "answer": answer,
                "model": self.generalist_model,
                "tokens_used": usage.get('total_tokens', 0),
                "error": None
            }
            
        except Exception as e:
            print(f"❌ Generalist call failed: {e}")
            return {
                "answer": None,
                "model": self.generalist_model,
                "tokens_used": 0,
                "error": str(e)
            }