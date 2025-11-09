import requests
import json
from config import NVIDIA_API_KEY, NVIDIA_API_BASE, ROUTER_MODEL

class IntentRouter:
    """
    SINGLE RESPONSIBILITY: Generate intent from user prompt
    """
    
    def __init__(self):
        self.model = ROUTER_MODEL
        self.api_key = NVIDIA_API_KEY
        self.base_url = NVIDIA_API_BASE
    
    def generate_intent(self, user_prompt):
        """
        Generate intent label and description from user prompt
        
        Args:
            user_prompt (str): User's input query
            
        Returns:
            dict: {
                "intent_label": str,
                "description": str,
                "confidence": float
            }
        """
        
        system_prompt = """You are an intent classifier for AI queries.

Given a user query, generate a JSON response with:
1. intent_label: A short, descriptive label in snake_case (2-4 words)
2. description: A detailed 1-sentence description of what this task involves
3. confidence: Your confidence score (0.0 to 1.0)

BE CONSISTENT: Similar queries should get the same intent_label.

Examples:
- "Write SQL to find top customers" → intent_label: "sql_generation"
- "Review this Python code" → intent_label: "code_review"
- "Summarize this document" → intent_label: "text_summarization"

Output ONLY valid JSON, nothing else."""

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 200
                },
                timeout=30
            )
            
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            intent_data = json.loads(content)
            
            # Validate
            if "intent_label" not in intent_data or "description" not in intent_data:
                raise ValueError("Missing required fields")
            
            if "confidence" not in intent_data:
                intent_data["confidence"] = 0.9
            
            return intent_data
            
        except Exception as e:
            print(f"❌ Router Error: {e}")
            return {
                "intent_label": "general_query",
                "description": "General query requiring generalist model",
                "confidence": 0.5,
                "error": str(e)
            }