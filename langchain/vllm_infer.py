from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:1122/v1"
import time

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
start = time.time()

chat_response = client.chat.completions.create(
    model="Qwen1.5-7B-Chat",
    messages=[
        
        {"role": "user", "content": "告诉我有关北京的特产"},
    ]
)
print(time.time() - start)
print("Chat response:", chat_response)
