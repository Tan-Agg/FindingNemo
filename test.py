import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NEMOTRON_KEY")



if api_key is None:
    raise ValueError("One or more API keys are not set in the environment variables.")
    
client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key
)

def run_test(model_name, messages):
    print("Testing model:", model_name)
    try:
        completion = client.chat.completions.create(
          model=model_name,
          messages=messages,
          temperature=0.5,
          top_p=1,
          max_tokens=1024,
          stream=False 
        )
        print(f"Successful response: \n{completion.choices[0].message.content}\n")
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}\n")

model1 = "nvidia/llama-3.1-nemotron-ultra-253b-v1"       
messages1 = [
    {"role": "user", "content": "Generate a 400 word essay on global warming."}
]
ans1 = run_test(model1, messages1)

model2 = "mistralai/mixtral-8x7b-instruct-v0.1"
messages2 = [
    {"role": "user", "content": "What is a Mixture of Experts (MoE) model in 50 words or less?"}
]
ans2 = run_test(model2, messages2)

model3 = "nvidia/llama-3.1-nemotron-70b-reward"
messages3 = [
    {"role": "user", "content": "Generate a 400 word essay on global warming."},
    {"role": "assistant", "content": ans1}
]
ans3 = run_test(model3, messages3)