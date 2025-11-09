import os
import json
import re
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import random

load_dotenv()

# --- Configuration ---

# **CRITICAL**: Ensure your NVIDIA API Key is set in your environment
# export NVIDIA_API_KEY="nvapi-..."
NVIDIA_API_KEY = os.getenv("NEMOTRON_KEY")

# The NVIDIA NIM API Base URL (Standard OpenAI-compatible endpoint)
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

# The specific 253B model ID confirmed via the NVIDIA API Catalog
GEN_MODEL_NAME = "mistralai/mixtral-8x7b-instruct-v0.1" 

# The specific 70B Reward Model ID confirmed via the NVIDIA API Catalog
REWARD_MODEL_NAME = "nvidia/llama-3.1-nemotron-70b-reward"

# The comprehensive system prompt for the Training Data Generator
PROMPT_GENERATOR_SYSTEM_PROMPT = """
You are a 'User Prompt Simulator'. Your task is to generate a single, natural-sounding user request based on the topic provided.
- **DO NOT** answer the request.
- **DO NOT** add any pre-amble, quotes, or introduction.
- **ONLY** output the user's request text and nothing else.
- The request should be random and diverse (e.g., it could be a simple question, a demand for code, a request for a poem, a "what if" scenario, or a comparison).
"""

RESPONSE_GENERATOR_SYSTEM_PROMPT = """
You are a helpful, expert-level AI assistant.
Provide a correct, complete, and well-explained response to the user's request.
Be thorough and clear in your explanation. The response should be informative and accurate. The response should be random and diverse (e.g., it could be a simple answer, code snippet, poem, detailed explanation, or comparison).
"""

# --- Client Initialization ---
client = OpenAI(
    base_url=NVIDIA_API_BASE,
    api_key=NVIDIA_API_KEY
)

# --- Function to Generate Training Data ---

def generate_random_prompt(topic: str) -> str | None:
    """
    API Call 1: Generates a random user prompt based on a general topic.
    """
    try:
        messages = [
            {"role": "system", "content": PROMPT_GENERATOR_SYSTEM_PROMPT},
            # The topic is the "user" message for this first call
            {"role": "user", "content": topic}
        ]

        response = client.chat.completions.create(
            model=GEN_MODEL_NAME,
            messages=messages,
            temperature=1.0,  # High temperature for creative, random prompts
            max_tokens=256,  # A user prompt doesn't need to be long
        )
        
        # Return the clean text of the generated user prompt
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"An error occurred during prompt generation with {GEN_MODEL_NAME}: {e}")
        return None

def generate_assistant_response(user_prompt: str) -> str | None:
    """
    API Call 2: Generates an assistant response to the user_prompt from Call 1.
    """
    if not user_prompt:
        return None
        
    try:
        messages = [
            {"role": "system", "content": RESPONSE_GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = client.chat.completions.create(
            model=GEN_MODEL_NAME,
            messages=messages,
            temperature=0.7,  # Moderate temperature for a helpful, correct answer
            max_tokens=512, # Allow for a full, complete answer
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"An error occurred during response generation with {GEN_MODEL_NAME}: {e}")
        return None


print(f"Attempting to connect to Trainer Generator: {GEN_MODEL_NAME}...")
num_examples_to_generate = 25
topic = "Japan"
output_filename = "generated_test_cases.json"

style_modifiers = [
    "a simple, one-sentence question about",
    "a 'what is' question about",
    "a 'how does' question about",
    "a question about the history of",
    "a question about the future of",
    "a complex 'what-if' scenario involving",
    "a 'why' question that requires a deep explanation of",
    "a philosophical question about the implications of",
    "a request to 'think step-by-step' to explain",
    "a question asking for a detailed analysis of",
    "a prompt asking for the pros and cons of",
    "a prompt asking to compare and contrast two aspects of",
    "a question from the perspective of a total beginner about",
    "a question from a skeptical expert about",
    "a prompt asking to explain {topic} as if to a 5-year-old",
    "a prompt asking to explain {topic} for a college-level paper",
    "a prompt from a teacher asking for a lesson plan on",
    "a prompt from a student who needs help with homework on",
    "a request for a bulleted list summarizing",
    "a request for a step-by-step guide on",
    "a request for a short poem about",
    "a request for a short, two-paragraph story involving",
    "a prompt asking for a table that outlines",
    "a request for a brief, 3-sentence summary of",
    "a request for a very detailed, multi-paragraph explanation of",
    "a prompt asking for a Python code snippet related to",
    "a prompt asking for a JavaScript function related to",
    "a request to explain a code concept related to",
    "a request to 'act as a senior developer' and review a concept about",
    "a prompt asking to write a simple shell script for",
    "a request to debug a hypothetical error message related to",
    "a request for a JSON example representing",
    "a request for a regular expression (regex) for",
    "a request for a metaphor or analogy to explain",
    "a request for a 'bad explanation' and then a 'good explanation' of",
    "a prompt asking for a common misconception about",
    "a prompt asking for an inspiring or motivational take on",
    "a prompt asking for a pessimistic take on",
    "a prompt asking for a funny or satirical take on",
    "a prompt asking to write a social media post about",
    "a prompt from someone feeling overwhelmed by",
    "a request for advice on how to deal with",
    "a prompt asking about the ethical considerations of",
    "a question about the emotional impact of",
    "a request to draft an email about",
    "a request to write a product description for",
    "a request for a list of 5 key facts about",
    "a request for the definition of a specific term related to",
    "a question about the main components of",
    "a prompt about the long-term effects of",
    "a question about the 'best' way to start learning about",
    "a prompt asking for a list of resources for",
    "a request to analyze the causes of",
    "a request to predict the consequences of",
    "a prompt asking for an argument *against*",
    "a prompt asking for an argument *for*",
    "a prompt asking for a neutral, encyclopedic definition of",
    "a request to write a formal statement about",
    "a request to write a casual, conversational text about",
    "a prompt asking for the origin of the term",
    "a prompt about the economic impact of",
    "a prompt about the social impact of",
    "a prompt about the environmental impact of",
    "a request to identify the key stakeholders involved in",
    "a request to solve a common problem related to",
    "a 'how-to' guide for a beginner on",
    "a 'how-to' guide for an expert on",
    "a request to roleplay as a historical figure discussing",
    "a request to write a news headline about",
    "a request to simplify a very complex aspect of"
]

all_test_cases = []

for i in range(num_examples_to_generate):
    print(f"Generating training example {i+1}/{num_examples_to_generate}")

    modifier = random.choice(style_modifiers)
    prompt_for_gen = modifier.format(topic=topic)

    if "{topic}" not in modifier:
        prompt_for_gen = f"{modifier} {topic}"
    
    user_prompt = generate_random_prompt(prompt_for_gen)
    if not user_prompt:
        print("Failed to generate user prompt. Skipping to next.")
        continue

    assistant_response = generate_assistant_response(user_prompt)
    if not assistant_response:
        print("Failed to generate assistant response. Skipping to next.")
        continue

    final_json_object = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response}
    ]

    print("\n--- Generated Training Example (JSON) ---")
    print(json.dumps(final_json_object, indent=2))
    print("-" * 40)

    all_test_cases.append(final_json_object)

try:
    with open(output_filename, 'w') as f:
        json.dump(all_test_cases, f, indent=2)
    print(f"\nSuccessfully saved {len(all_test_cases)} test cases to {output_filename}")

except Exception as e:
    print(f"\nError saving to file: {e}")
