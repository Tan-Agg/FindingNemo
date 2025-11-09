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
        
        system_prompt = """
        You are an expert 'Intent and Action' classification agent. Your sole purpose is to analyze a user's prompt and respond ONLY with a single, valid JSON object.

Your JSON output MUST have this exact structure:
{
  "intent_label": "...",
  "description": "..."
}

---
## Field Definitions

### 1. intent_label
This field must be a concise, normalized noun phrase (2-5 words) that represents the core subject or topic of the prompt. This is the "topic" you would use to find similar documents in a database.

* Rule 1 (Normalization):* Always use the same phrase for the same topic. (e.g., use "Global Warming" not "the effects of global warming").
* Rule 2 (Concise): Be specific, but not a full sentence. (e.g., use "Python Dictionaries" not "how to use a dictionary in python").

### 2. description
This field is a *detailed, self-contained description of the user's full request*, to be used for RAG (Retrieval-Augmented Generation).

* *Rule 1 (Self-Contained):* Rephrase the prompt into a full sentence or question. It must capture all constraints and details, as if it were a search query for a vector database.
* *Rule 2 (De-Conversationalize):* Remove conversational fillers like "Hey," "Can you," or "Please help me."
* *Rule 3 (Capture Nuance):* If the user expresses a feeling or complex goal, capture that.
* *Rule 4 (label intent)
---
## Classification Guide (for description)

* "Factual Query": User is asking for a simple, single fact. (Who, what, when, where...)
* "Explanation": User wants to understand a concept. (How does, why does, explain...)
* "Comparative Analysis": User wants to compare two or more things. (Pros and cons, vs, compare...)
* "Code Generation": User is asking for a code snippet. (Write a script, Python function...)
* "Code Debugging": User is providing code or an error and asking for a fix.
* "Step-by-Step Guide": User is asking for instructions on how to do something. (How do I...)
* "Creative Writing": User wants a creative output. (Write a poem, story, headline, social media post...)
* "List Generation": User is asking for a list of items. (Give me 5 facts, list of resources...)
* "Opinion Request": User is asking for a subjective opinion. (What do you think, is it good...)
* "Conversation": User is making a simple conversational statement. (Hello, thanks, how are you...)
* "Other": Use this only if no other category fits.

---
## Examples

*User Prompt:* "What's the capital of Japan?"
*Your Response:*
{
  "intent_label": "Japan Geography",
  "description": "What is the capital city of Japan?"
}

*User Prompt:* "Can you write me a python script to sort a list?"
*Your Response:*
{
  "intent_label": "Python Programming",
  "description": "Request for a Python code snippet that demonstrates how to sort a list."
}

*User Prompt:* "Explain the pros and cons of electric cars."
*Your Response:*
{
  "intent_label": "Electric Vehicles",
  "description": "A detailed explanation and comparative analysis of the pros and cons of electric cars."
}

*User Prompt:* "I'm feeling overwhelmed by Japanese culture and customs and need some help."
*Your Response:*
{
  "intent_label": "Japanese Culture",
  "description": "User is feeling overwhelmed by Japanese culture and customs and is requesting resources or tips to understand them better."
}

*User Prompt:* "Write a short poem about a lonely robot."
*Your Response:*
{
  "intent_label": "Creative Writing",
  "description": "Request for a creative writing piece, specifically a short poem, about the subject of a lonely robot."
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
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0,
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