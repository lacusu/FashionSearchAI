import json
import time
import requests
import matplotlib.pyplot as plt
from pathlib import Path

SAVE_DIR = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "red party dress",
    "blue denim shirt men",
    "black running shoes"
]

BASE_URL = "http://localhost:8000"

def save_screenshot(data, filename):
    fig = plt.figure(figsize=(8, 6))
    plt.text(0, 1, json.dumps(data, indent=2, ensure_ascii=False),
             va="top", ha="left", fontsize=8, family="monospace")
    plt.axis("off")
    fig.savefig(SAVE_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

def capture_search_results():
    for i, q in enumerate(QUERIES):
        r = requests.get(f"{BASE_URL}/search", params={"q": q})
        save_screenshot(r.json(),
                        f"search_q{i+1}_{q.replace(' ', '_')}.png")
        time.sleep(1)

def capture_recommend_results():
    for i, q in enumerate(QUERIES):
        r = requests.post(
            f"{BASE_URL}/recommend",
            json={"query": q, "k": 3},
        )
        save_screenshot(r.json(),
                        f"recommend_q{i+1}_{q.replace(' ', '_')}.png")
        time.sleep(1)

if __name__ == "__main__":
    capture_search_results()
    capture_recommend_results()
    print("Screenshots saved in docs/screenshots/")
