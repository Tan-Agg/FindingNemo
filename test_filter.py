import os
import json
import time
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NEMOTRON_KEY")
if not NVIDIA_API_KEY:
    print("Error: NVIDIA_API_KEY not set. Please set it as an environment variable.")
    exit()

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

# The Reward Model ID
REWARD_MODEL_NAME = "nvidia/llama-3.1-nemotron-70b-reward"

INPUT_FILE = "generated_test_cases.json"
OUTPUT_FILE = "top_75_test_cases.json"
TOP_PERCENTAGE = 0.75
REQUEST_DELAY = 0.5

client = OpenAI(
    base_url=NVIDIA_API_BASE,
    api_key=NVIDIA_API_KEY
)

def get_reward_score(test_case_messages: List[Dict[str, str]]) -> float | None:
    """
    Calls the Nemotron-70B-Reward model to get a quality score.
    
    This specific model is called via the chat.completions endpoint,
    but it returns its score inside the 'logprobs' object.
    """
    try:
        response = client.chat.completions.create(
            model=REWARD_MODEL_NAME,
            messages=test_case_messages,
            max_tokens=1,  # We don't need a text response, just the score
            logprobs=True  # CRITICAL: This is required to get the score
        )
        
        # --- How to parse the score ---
        # The reward model returns its score as a stringified float
        # in the 'token' field of the first logprob.
        score_str = response.choices[0].logprobs.content[0].token
        
        # Convert the score string to a float
        score = float(score_str)
        return score

    except Exception as e:
        print(f"\nError calling reward model: {e}")
        # Return a very low score so it gets filtered out
        return -999.0
    
def main():
    print(f"Loading test cases from {INPUT_FILE}...")
    
    # 1. Load Data
    try:
        with open(INPUT_FILE, 'r') as f:
            all_test_cases = json.load(f)
        print(f"Successfully loaded {len(all_test_cases)} test cases.")
    except Exception as e:
        print(f"Fatal Error: Could not read {INPUT_FILE}: {e}")
        return

    # 2. Process Data: Get scores for all test cases
    print(f"Processing {len(all_test_cases)} test cases with {REWARD_MODEL_NAME}...")
    scored_test_cases = []

    # Use tqdm for a progress bar
    for test_case in tqdm(all_test_cases, desc="Scoring test cases"):
        
        # Get the score for the [ {"role": "user", ...}, {"role": "assistant", ...} ] object
        score = get_reward_score(test_case)
        
        if score is not None:
            # Store the score and the original data together
            scored_test_cases.append({
                "score": score,
                "data": test_case
            })
        
        # Add a delay to avoid hitting API rate limits
        # 0.5 seconds * 2000 cases = ~17 minutes
        time.sleep(REQUEST_DELAY)

    print("\nScoring complete.")
    
    # 3. Filter Data
    print("Sorting test cases by reward score...")
    
    # Sort the list of dictionaries by the 'score' key, from highest to lowest
    scored_test_cases.sort(key=lambda x: x['score'], reverse=True)
    
    # Calculate the 75% cutoff index
    num_to_keep = int(len(scored_test_cases) * TOP_PERCENTAGE)
    
    print(f"Total scored: {len(scored_test_cases)}. Keeping top {TOP_PERCENTAGE*100}% ({num_to_keep} cases).")
    
    # Slice the list to get only the top 75%
    top_test_cases = scored_test_cases[:num_to_keep]

    # 4. Save Data
    
    # Extract just the original test case data from the sorted list
    final_test_cases = [item['data'] for item in top_test_cases]
    
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(final_test_cases, f, indent=2)
        print(f"\nSuccessfully saved {len(final_test_cases)} high-quality test cases to {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nError saving final file: {e}")

if __name__ == "__main__":
    main()