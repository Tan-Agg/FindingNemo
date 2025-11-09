import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Model endpoints
GENERALIST_MODEL = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
ROUTER_MODEL = "nvidia/mixtral-8x7b-instruct-v0.1"
REWARD_MODEL = "nvidia/llama-3.1-nemotron-70b-reward"

# API Base URL
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

# Thresholds
SIMILARITY_THRESHOLD = 0.85
QUERY_THRESHOLD = 5

# Costs (per 1M tokens)
COSTS = {
    "generalist_input": 0.60,
    "generalist_output": 1.80,
    "router": 0.63,
    "specialist": 0.63
}