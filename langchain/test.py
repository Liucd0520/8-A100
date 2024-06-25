from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:1122/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen1.5-7B-Chat",
    messages=[
        {"role": "system", "content": "你是一个人工智能助手."},
        {"role": "user", "content": "告诉我有关北京的特产"},
    ]
)
print("Chat response:", chat_response)
