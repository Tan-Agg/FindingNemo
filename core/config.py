import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Model endpoints
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME")
ROUTER_MODEL = os.getenv("ROUTER_MODEL")
REWARD_MODEL = os.getenv("REWARD_MODEL")
GENERALIST_MODEL = os.getenv("GENERALIST_MODEL")

# API Base URL
NVIDIA_API_BASE = os.getenv("NVIDIA_API_BASE")

# Thresholds
SIMILARITY_THRESHOLD = 0.35
QUERY_THRESHOLD = 5

# Costs (per 1M tokens)
COSTS = {
    "generalist_input": 0.60,
    "generalist_output": 1.80,
    "router": 0.63,
    "specialist": 0.63
}