# FindingNemo: Self-Evolving AI Agents 🚀

## Inspiration
Every AI today has a flaw — once deployed, it stops learning.  
FindingNemo is different: it **watches itself work, spots inefficiencies, and spawns new specialized agents automatically**, evolving continuously without humans in the loop.

## What It Does
Give it a prompt — for example, *“Write an SQL query for top 5 customers.”*  
- The **Generalist Model** handles it initially.  
- The **Router** observes repeated intents and decides when to create a specialist.  
- **Agent Creation Loop**:  
  1. Generate diverse synthetic examples  
  2. Filter top-quality examples via reward model  
  3. Fine-tune a smaller specialist model automatically  

The result: a **network of evolving agents**, each optimized for specific tasks.

## How It Works
- **Router:** Detects intent and routes queries  
- **Trainer:** Generates synthetic datasets for new intents  
- **Fine-Tuner + Reward Model:** Trains specialists autonomously  
- **Continuous Evolution:** No human labeling required

## Tech Stack
- LLMs: Nemotron-253B, Phi-3-mini, Mistral-7B  
- Reward Filtering: Nemotron-70B-Reward  
- Fine-Tuning: PEFT (LoRA), Hugging Face Accelerate  
- Routing: FAISS embeddings + cosine similarity  
- Data Management: JSONL pipelines  
- Orchestration: Python microservices  

## Challenges
- Generating realistic, non-repetitive test data  
- Stable intent clustering & semantic retrieval  
- Reward model drift and evolving intents  
- Time & compute constraints  

## Achievements
- Fully autonomous evolution loop  
- SQL, code, and other intent specialists created  
- Scalable, edge-ready architecture  
- Early self-learning AI that improves over time  

## Next Steps
- Close full agentic loop with human feedback  
- Expand specialist library to finance, legal, and more  
- Optimize for edge and cost-efficient self-training  

## Built With
FAISS, Hugging Face, JSON/JSONL, PEFT, LoRA, Mistral-7B, Nemotron, Nemotron-70B-Reward, NVIDIA NEMO/NIM API, Phi-3-mini, Python, Python microservices, SQLite, Streamlit
