import os
import json
from openai import OpenAI
from typing import Dict, Any, List

# --- Configuration ---

# **CRITICAL**: Ensure your NVIDIA API Key is set in your environment
# export NVIDIA_API_KEY="nvapi-..."
NVIDIA_API_KEY = os.environ.get("NEMOTRON_KEY")

# The NVIDIA NIM API Base URL (Standard OpenAI-compatible endpoint)
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

# The specific 253B model ID confirmed via the NVIDIA API Catalog
MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1"    

# The comprehensive system prompt for the Training Data Generator
TRAINER_GENERALIST_SYSTEM_PROMPT = """
You are a Training Data Generator creating 5,000 high-quality, diverse training examples for specialized AI models. Generate examples in this exact format:

json
{"role": "user", "content": "<natural user request>"},{"role": "assistant", "content": "<correct complete response>"}
Complexity Distribution
25% Basic: Simple, single-step tasks
35% Intermediate: Multi-step with some complexity
25% Advanced: Complex logic, multiple constraints
15% Expert: Edge cases, optimization, unusual requirements
User Request Style Distribution
20% Terse (5-15 words)
30% Standard (typical length/detail)
25% Detailed (specific requirements)
15% Conversational (exploratory, chatty)
10% Ambiguous (requires interpretation)
Generation Requirements
Diversity: Vary vocabulary, phrasing, structure, formality, context, and scenarios. Cover all domain sub-categories, operations (create/modify/explain/debug/optimize), complexity levels, common patterns, edge cases, and best practices. Use different verb forms and question styles.
User Realism: Generate natural requests reflecting actual user behavior—varied specificity, implicit context, typos, ambiguity, mixed technical/layman terms, 5-100+ word range, and different question styles (direct/commands/goals/conversational).
Response Quality: All responses must be correct, complete, professional, self-contained, and follow domain best practices. No placeholders or incomplete sections.
Domain Coverage: Analyze the given intent type and ensure proportional coverage across all sub-domains, operations, tools, frameworks, error handling, performance, security, and optimization techniques.

Critical Anti-Patterns to AVOID
❌ Template-like repetition (examples differing by 1-2 words only)
❌ Unrealistic requests or contrived scenarios
❌ Incomplete, incorrect, or placeholder responses
❌ Over-simplified patterns (same structure repeated)
❌ Trivial variations ("Write X for Y" × 1000)
❌ Missing edge cases or major domain areas
❌ Malformed JSON or escaping errors

Quality Criteria (Reward Model will filter for 90%+ retention)
Natural, realistic user requests appropriate to complexity level
Correct, complete, well-formatted responses
Well-matched request-response pairs
Self-contained examples (no external context needed)
Comprehensive domain coverage with balanced distribution
Process
For each example: (1) Select sub-domain/category randomly with balanced coverage, (2) Assign complexity level, (3) Generate unique, natural user request with varied vocabulary/structure, (4) Create correct response following best practices, (5) Format as valid JSON with proper escaping, (6) Verify uniqueness and diversity.
Target: 5,000 examples where ≥4,500 pass Reward Model evaluation. Prioritize diversity and quality equally—every example must be unique, purposeful, and teach something valuable.
"""

# --- Client Initialization ---
client = OpenAI(
    base_url=NVIDIA_API_BASE,
    api_key=NVIDIA_API_KEY
)

# --- Function to Generate Training Data ---

def generate_training_example(prompt: str) -> Dict[str, Any] | None:
    """
    Generates a single training example using the Trainer Generalist (253b).
    
    Args:
        prompt: A dynamic prompt to guide the model on the *kind* of example 
                to generate (e.g., "Generate a complex SQL query optimization 
                example for a sales database"). This should be varied to meet 
                the 5,000-example diversity target.
    
    Returns:
        A dictionary containing the generated user request and assistant response,
        or None if generation fails.
    """
    if not NVIDIA_API_KEY:
        print("Error: NVIDIA_API_KEY not set. Please set it as an environment variable.")
        return None

    try:
        # The system prompt ensures the model acts as the Training Data Generator
        # The user prompt gives the model its specific task for this instance
        messages = [
            {"role": "system", "content": TRAINER_GENERALIST_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        # Call the chat completions API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,  # Higher temperature promotes diversity/creativity
            max_tokens=2048, # Ensure enough capacity for the complete example
            response_format={"type": "json_object"} # Strictly enforce JSON output
        )
        
        # Parse the JSON string from the model's response
        json_content = response.choices[0].message.content
        
        # We expect a list of two JSON objects (user/assistant) or a single object 
        # that contains both roles, depending on the model's exact interpretation.
        # Given your prompt structure, it will likely return a single JSON list/object.
        return json.loads(json_content) 

    except Exception as e:
        print(f"An error occurred during API call with {MODEL_NAME}: {e}")
        return None

# --- Example of How to Call the Function ---

# print(f"Attempting to connect to Trainer Generalist: {MODEL_NAME}...")
# example_prompt_for_sql = "Generate an intermediate-level SQL query example that involves a LEFT JOIN, a SUBQUERY, and uses the DATE_TRUNC function, and provide the correct, professional SQL response."
# generated_example = generate_training_example(example_prompt_for_sql)

# if generated_example:
#     print("\n--- Generated Training Example (JSON) ---")
#     print(json.dumps(generated_example, indent=2))
# else:
#     print("Failed to generate a training example.")
