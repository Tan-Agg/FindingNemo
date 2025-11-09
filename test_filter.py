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

INPUT_FILE = "test_cases.json"
OUTPUT_FILE = "top_75_test_cases.json"
TOP_PERCENTAGE = 0.75
REQUEST_DELAY = 0.5

client = OpenAI(
    base_url=NVIDIA_API_BASE,
    api_key=NVIDIA_API_KEY
)