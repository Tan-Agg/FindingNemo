import os
import json
import time
import random
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
NVIDIA_API_KEY = os.getenv("NEMOTRON_KEY")
if not NVIDIA_API_KEY:
    print("Error: NVIDIA_API_KEY not set. Please set it as an environment variable.")
    exit()

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
REWARD_MODEL_NAME = "nvidia/llama-3.1-nemotron-70b-reward"

client = OpenAI(
    base_url=NVIDIA_API_BASE,
    api_key=NVIDIA_API_KEY
)

# Load test cases
with open("generated_test_cases.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

# --- Main Loop ---
for i, convo in enumerate(tqdm(data, desc="Evaluating test cases"), start=1):
    user_prompt = convo[0]["content"].strip()
    assistant_response = convo[1]["content"].strip()

    # --- NVIDIA API Call with retry and delay ---
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=REWARD_MODEL_NAME,
                messages=[
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            )
            break  # success, exit retry loop
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed for test case {i}: {e}")
            time.sleep(5 * (attempt + 1))
    else:
        print(f"❌ Skipping test case {i} after 3 failed attempts.")
        continue

    # --- Extract reward score ---
    raw_output = response.choices[0].message.content.strip()
    try:
        reward_score = float(raw_output.split()[-1])
    except ValueError:
        reward_score = None  # Non-numeric score

    results.append({
        "test_case": i,
        "user_prompt": user_prompt,
        "assistant_response": assistant_response,
        "reward_score": reward_score if reward_score is not None else raw_output
    })

    # --- Random delay to avoid rate limits ---
    time.sleep(random.uniform(0.3, 0.5))

# --- Sort results by reward_score (descending) ---
numeric_results = [r for r in results if isinstance(r["reward_score"], (int, float))]
non_numeric_results = [r for r in results if not isinstance(r["reward_score"], (int, float))]

sorted_results = sorted(numeric_results, key=lambda x: x["reward_score"], reverse=True)
sorted_results.extend(non_numeric_results)  # keep non-numeric at the end

# --- Take top 75% ---
top_k = int(len(sorted_results) * 0.75)
filtered_results = sorted_results[:top_k]

# --- Save to file ---
with open("reward_results.json", "w", encoding="utf-8") as f:
    json.dump(filtered_results, f, indent=2, ensure_ascii=False)

print(f"✅ Processed {len(results)} test cases.")
print(f"📊 Saved top {top_k} entries (75%) to reward_results.json.")
