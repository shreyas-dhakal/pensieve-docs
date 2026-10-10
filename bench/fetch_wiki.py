"""Fetch a labelled Wikipedia corpus for benchmarking: python bench/fetch_wiki.py [out_dir]

Writes <out>/<topic>/<title>.md (plain text, links stripped) and <out>/truth.json with each file's
topic and the Wikipedia links among the fetched articles. Text is CC BY-SA 4.0, so it is not committed.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

TOPICS = {
    "malaria": ["Malaria", "Mosquito", "Anopheles", "Plasmodium falciparum", "Artemisinin", "Insecticide-treated net",
                "Dengue fever", "Aedes aegypti", "Mosquito control", "Quinine"],
    "home-energy": ["Solar panel", "Photovoltaics", "Home energy storage", "Lithium-ion battery", "Heat pump",
                    "Net metering", "Power inverter", "Electrical grid", "Thermal insulation", "Feed-in tariff"],
    "machine-learning": ["Machine learning", "Neural network (machine learning)", "Transformer (deep learning)",
                         "Backpropagation", "Gradient descent", "Large language model", "Word embedding",
                         "Reinforcement learning", "Overfitting", "Convolutional neural network"],
    "coffee": ["Coffee", "Espresso", "Caffeine", "Coffee roasting", "Coffea arabica", "Latte", "Cold brew coffee",
               "Coffeehouse", "Decaffeination", "Coffee preparation"],
    "roman-history": ["Roman Empire", "Julius Caesar", "Augustus", "Roman Republic", "Punic Wars", "Colosseum",
                      "Roman legion", "Fall of the Western Roman Empire", "Cicero", "Roman roads"],
    "jazz": ["Jazz", "Bebop", "Miles Davis", "John Coltrane", "Swing music", "Louis Armstrong", "Blue note",
             "Jazz improvisation", "Saxophone", "Kind of Blue"],
}
API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "pensieve-docs-bench/0.1 (https://github.com/shreyas-dhakal/pensieve-docs)"}


def get(**params):
    url = API + "?" + urllib.parse.urlencode({"format": "json", "formatversion": 2, "redirects": 1, **params})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    truth, links = {"topics": {}, "links": []}, {}
    for topic, titles in TOPICS.items():
        for title in titles:
            page = get(action="query", prop="extracts|links", explaintext=1, pllimit="max", plnamespace=0,
                       titles=title)["query"]["pages"][0]
            text = re.sub(r"^(=+)\s*(.*?)\s*=+\s*$", lambda m: "#" * len(m.group(1)) + " " + m.group(2),
                          page["extract"], flags=re.M)  # == Heading == -> ## Heading
            rel = f"{topic}/{slug(page['title'])}.md"
            (out / topic).mkdir(exist_ok=True)
            (out / rel).write_text(f"# {page['title']}\n\n{text}\n", encoding="utf-8")
            truth["topics"][rel] = topic
            links[rel] = {l["title"] for l in page.get("links", [])}
            print(rel, len(text))
            time.sleep(0.2)
    by_title = {f"# {t}": rel for rel in truth["topics"] for t in [(out / rel).read_text().splitlines()[0][2:]]}
    for rel, targets in links.items():
        truth["links"] += sorted([rel, by_title[f"# {t}"]] for t in targets if f"# {t}" in by_title and by_title[f"# {t}"] != rel)
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"{len(truth['topics'])} articles, {len(truth['links'])} links among them -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "bench/data/wiki"))
