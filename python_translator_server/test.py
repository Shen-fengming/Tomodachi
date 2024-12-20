import requests
import json

url = "http://127.0.0.1:5000/analyze"
text = "そもそも衣装なんて要る？。"

response = requests.post(url, json={"text": text})

if response.status_code == 200:
    result = response.json()
    print("Tokens (formatted):")
    print(json.dumps(result, ensure_ascii=False, indent=4))
else:
    print(f"Error: {response.status_code}, {response.text}")