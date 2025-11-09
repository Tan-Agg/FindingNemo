import requests
from config import NVIDIA_API_KEY, NVIDIA_API_BASE, GENERALIST_MODEL, ROUTER_MODEL, REWARD_MODEL, GEN_MODEL_NAME

def test_api():
    """Test NVIDIA API connection"""
    
    # Test Generalist (253B)
    print("Testing Generalist (253B)...")
    response = requests.post(
        f"{NVIDIA_API_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": GENERALIST_MODEL,
            "messages": [{"role": "user", "content": "Say hello!"}],
            "max_tokens": 50
        }
    )
    
    if response.status_code == 200:
        print("✅ Generalist working!")
        print(f"Response: {response.json()['choices'][0]['message']['content']}\n")
    else:
        print(f"❌ Generalist failed: {response.status_code}")
        print(f"Error: {response.text}\n")
    
    # Test Router (Mixtral)
    print("Testing Router (Mixtral)...")
    response = requests.post(
        f"{NVIDIA_API_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": ROUTER_MODEL,
            "messages": [{"role": "user", "content": "Classify: Write SQL query"}],
            "max_tokens": 50
        }
    )
    
    if response.status_code == 200:
        print("✅ Router working!")
        print(f"Response: {response.json()['choices'][0]['message']['content']}\n")
    else:
        print(f"❌ Router failed: {response.status_code}")
        print(f"Error: {response.text}\n")

if __name__ == "__main__":
    test_api()