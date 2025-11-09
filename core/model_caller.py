import requests
import json
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
        
        # Validate configuration
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is not set in .env file")
        if not self.base_url:
            raise ValueError("NVIDIA_API_BASE is not set in .env file")
        if not self.generalist_model:
            raise ValueError("GENERALIST_MODEL is not set in .env file")
        
        print(f"   Using API: {self.base_url}")
        print(f"   Generalist Model: {self.generalist_model}")
    
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
            print(f"   Calling specialist: {endpoint}")
            
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
            
            # Print response for debugging
            print(f"   Response status: {response.status_code}")
            
            response.raise_for_status()
            data = response.json()
            
            # DEBUG: Print response structure
            print(f"   Response keys: {data.keys()}")
            
            # Extract answer with fallbacks
            answer = None
            if 'choices' in data and len(data['choices']) > 0:
                choice = data['choices'][0]
                if 'message' in choice:
                    message = choice['message']
                    # Try reasoning_content first (for Ultra 253B model)
                    if 'reasoning_content' in message and message['reasoning_content']:
                        answer = message['reasoning_content']
                    # Then try regular content
                    elif 'content' in message and message['content']:
                        answer = message['content']
                elif 'text' in choice:
                    answer = choice['text']
            
            if not answer:
                print(f"   ⚠️  Could not extract answer from response: {data}")
                return {
                    "answer": f"Error: Invalid response format",
                    "model": endpoint,
                    "tokens_used": 0,
                    "error": "Invalid response format"
                }
            
            usage = data.get('usage', {})
            
            return {
                "answer": answer,
                "model": endpoint,
                "tokens_used": usage.get('total_tokens', 0),
                "error": None
            }
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f": {error_detail}"
            except:
                error_msg += f": {response.text}"
            
            print(f"❌ Specialist call failed: {error_msg}")
            return {
                "answer": f"Error calling specialist: {error_msg}",
                "model": endpoint,
                "tokens_used": 0,
                "error": error_msg
            }
            
        except Exception as e:
            print(f"❌ Specialist call failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error: {str(e)}",
                "model": endpoint,
                "tokens_used": 0,
                "error": str(e)
            }
    
    def call_generalist(self, user_prompt, max_tokens=500):
        """
        Call the generalist model
        
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
            print(f"   Making API request to: {self.base_url}/chat/completions")
            print(f"   Model: {self.generalist_model}")
            
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
            
            # Print response for debugging
            print(f"   Response status: {response.status_code}")
            
            # Check for errors
            response.raise_for_status()
            data = response.json()
            
            # DEBUG: Print FULL response to see what we're getting
            print(f"\n   === FULL API RESPONSE ===")
            print(json.dumps(data, indent=2))
            print(f"   === END RESPONSE ===\n")
            
            # DEBUG: Print response structure
            print(f"   Response keys: {list(data.keys())}")
            
            # Extract answer with multiple fallback strategies
            answer = None
            
            # Strategy 1: Standard OpenAI format
            if 'choices' in data and len(data['choices']) > 0:
                choice = data['choices'][0]
                print(f"   Choice keys: {list(choice.keys())}")
                
                if 'message' in choice:
                    message = choice['message']
                    # Try reasoning_content first (for Ultra 253B model)
                    if 'reasoning_content' in message and message['reasoning_content']:
                        answer = message['reasoning_content']
                        print(f"   ✅ Extracted from reasoning_content")
                    # Then try regular content
                    elif 'content' in message and message['content']:
                        answer = message['content']
                        print(f"   ✅ Extracted from message.content")
                    print(f"   DEBUG: answer type={type(answer)}, length={len(answer) if answer else 0}")
                elif 'text' in choice:
                    answer = choice['text']
                    print(f"   ✅ Extracted from text")
            
            # Strategy 2: Direct content field
            elif 'content' in data:
                answer = data['content']
                print(f"   ✅ Extracted from direct content field")
                print(f"   DEBUG: answer type={type(answer)}, value={repr(answer[:100]) if answer else answer}")
            
            # Strategy 3: Response field
            elif 'response' in data:
                answer = data['response']
                print(f"   ✅ Extracted from response field")
                print(f"   DEBUG: answer type={type(answer)}, value={repr(answer[:100]) if answer else answer}")
            
            # Check if answer is None or empty (but allow empty string as valid)
            if answer is None:
                # Print full response for debugging
                print(f"   ⚠️  Could not extract answer. Full response:")
                print(json.dumps(data, indent=2)[:500])  # First 500 chars
                return {
                    "answer": f"Error: Could not extract answer from API response. Response keys: {list(data.keys())}",
                    "model": self.generalist_model,
                    "tokens_used": 0,
                    "error": "Invalid response format"
                }
            
            usage = data.get('usage', {})
            answer_length = len(answer) if answer else 0
            print(f"   ✅ Got response ({answer_length} chars)")
            
            return {
                "answer": answer,
                "model": self.generalist_model,
                "tokens_used": usage.get('total_tokens', 0),
                "error": None
            }
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f": {error_detail}"
                print(f"❌ API Error Response: {error_detail}")
            except:
                error_msg += f": {response.text[:200]}"
                print(f"❌ API Error Text: {response.text[:200]}")
            
            print(f"❌ Generalist call failed: {error_msg}")
            return {
                "answer": f"API Error: {error_msg}",
                "model": self.generalist_model,
                "tokens_used": 0,
                "error": error_msg
            }
            
        except requests.exceptions.Timeout:
            error_msg = "Request timeout (60s)"
            print(f"❌ Generalist call failed: {error_msg}")
            return {
                "answer": f"Error: Request timed out after 60 seconds",
                "model": self.generalist_model,
                "tokens_used": 0,
                "error": error_msg
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            print(f"❌ Generalist call failed: {error_msg}")
            return {
                "answer": f"Error: {str(e)}",
                "model": self.generalist_model,
                "tokens_used": 0,
                "error": error_msg
            }
            
        except KeyError as e:
            error_msg = f"Invalid API response format: missing {str(e)}"
            print(f"❌ Generalist call failed: {error_msg}")
            if 'data' in locals():
                print(f"   Response data keys: {list(data.keys())}")
            return {
                "answer": f"Error: Invalid API response format - missing {str(e)}",
                "model": self.generalist_model,
                "tokens_used": 0,
                "error": error_msg
            }
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"❌ Generalist call failed: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error: {str(e)}",
                "model": self.generalist_model,
                "tokens_used": 0,
                "error": error_msg
            }