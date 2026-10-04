import json
import os
import urllib.request
import urllib.error


def load_env(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(path):
        return
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_env()


def chat(messages):
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    model = os.environ.get("LLM_MODEL", "qwen2.5")
    api_key = os.environ.get("LLM_API_KEY")

    url = f"{base_url}/chat/completions"
    body = json.dumps({"model": model, "messages": messages})
    data = body.encode("utf-8")

    headers  = {
        "Content-Type": "application/json",
        "User-Agent": "anime-watchlist-agent/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=data, headers=headers)

    try:
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        print(f"LLM request failed: {e.code} {e.reason}")
        return None
    except urllib.error.URLError as e:
        print("Could not reach LLM. Check your connection.")
        return None

