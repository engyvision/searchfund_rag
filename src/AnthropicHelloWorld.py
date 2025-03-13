import requests
import os

# ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
# if ANTHROPIC_API_KEY is None:
#     raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")

url = "https://api.anthropic.com/v1/messages"

headers = {
    "x-api-key": "",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

data = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello, world"}
    ]
}

response = requests.post(url, json=data, headers=headers)

print(response.json())  # Print API response
