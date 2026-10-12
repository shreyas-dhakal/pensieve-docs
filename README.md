# pensieve-docs

pensieve-docs reads a folder of documents and maps how they connect: which files link to or cite each other, which ideas they share, and where each idea appears, down to the page or line.

Use it to:

- **find related documents** for a file or a piece of text, with the reasons why
- **ask the map**: what an idea connects to, how two ideas link up, and where each connection is written down
- **save an AI agent's tokens**: the agent asks pensieve-docs where to look, then reads a few sections instead of every file
- **see the big picture**: groups of related documents and the ideas that tie them together

It runs entirely on your computer. It makes no network calls and uses no AI model, so it is free, private and gives the same result every time.

Passwords, API keys, card numbers, email addresses and phone numbers are blanked out before anything is saved. Your files are never changed.

It reads `.md`, `.txt`, `.pdf` (if the PDF has text, not a scan) and `.docx`.

## Install

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/shreyas-dhakal/pensieve-docs
cd pensieve-docs
uv sync
```

## Use it

**1. Build the map of a folder**

```bash
uv run pensieve-docs build path/to/your/folder
```

This writes `pensieve-out/graph.json` (the map) and `pensieve-out/GRAPH_REPORT.md` (a readable summary). Run it again after you change files; only changed files are re-read.

**2. Find related documents**

```
$ echo "Should we add a home battery for backup power during a grid outage?" | uv run pensieve-docs related -
0.70  Review meeting, June 2024  (meetings/2024-06-review.md)
        backup power  [meetings/2024-06-review.md L5]
        grid outage  [meetings/2024-06-review.md L5]
        home battery  [meetings/2024-06-review.md L5]
0.60  Battery Storage  (notes/battery-storage.md)
        home battery  [notes/battery-storage.md L3]
        ...
```

Each result says which shared ideas connect it and where they are (`L5` is line 5). Pass a file instead of `-` to find documents related to that file. For short questions, add `--min-score 0` so weaker matches still show.

From Python:

```python
from pensieve_docs.related import Index

idx = Index("pensieve-out/graph.json")
idx.related("home battery for backup power", k=5)
```

**3. Ask the map**

```
$ uv run pensieve-docs query "should we add a home battery for backup power"
START backup power [concept, in 2 documents, ...] meetings/2024-06-review.md L5
backup power --co_occurs--> grid outage [INFERRED 0.95] meetings/2024-06-review.md L5
Battery Storage --mentions--> backup power [INFERRED 0.75] notes/battery-storage.md L3
...
$ uv run pensieve-docs path "heat pump" "net metering"
$ uv run pensieve-docs explain "net metering"
```

- `query` starts at the ideas your question names and walks out along the strongest connections, stopping at `--budget` tokens (default 2000). `--dfs` follows one chain deep instead.
- `path` shows the shortest chain of connections between two things.
- `explain` shows one idea or document: where it appears and everything it connects to.

Every line says which file and line it came from, so an agent can open just those lines. Name a document by its title or its path.

**4. Look at it** (optional)

```bash
uv sync --extra ui
uv run pensieve-docs ui
```

This opens a local page with the map: click a document or idea to see everything it connects to and where.

## What's in the map

- **Folders, documents and sections**, and which contains which.
- **Links and citations** between your documents. These are facts read straight from the files.
- **Ideas** (key terms such as "heat pump" or "net metering") and the documents and sections that mention them.
- **Ideas that go together**, with the verb that connects them when there is one ("backpropagation *computes* gradient"), and whether the text says something is *not* so.
- **"X is a kind of Y"** links, from phrases like "pests such as mosquitoes".
- **Related documents**, with the ideas they share.

Every connection is marked by how sure it is: *extracted* (stated in the file), *inferred* (worked out) or *ambiguous* (worth a second look), and records the file and line it came from.

The file format matches [graphify](https://github.com/Graphify-Labs/graphify), which builds the same kind of map for code, so tools that read one can read the other.

## How well it works

We test it on 60 Wikipedia articles across six topics (coffee, jazz, malaria, machine learning, Roman history, home energy) and on a standard set of scientific papers.

**Token savings for AI agents.** For 10 questions about the 60 articles, an agent that asks pensieve-docs first and reads only the sections it points to reads about **2,000 tokens instead of 630,000: 313 times less**. The article with the answer was in the top 3 for 9 of the 10 questions.

**Finding related documents.** Of the top 3 related articles it suggests, 99% are on the same topic. On scientific papers searched by title, it is about as good as standard keyword search. AI embedding models (which need a model download) do better there.

**Answer quality, LazyGraphRAG-style.** Following [Microsoft's LazyGraphRAG test](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/), an AI answered 20 questions from the text each method found and another AI judged the answers. On questions about one or two articles, pensieve-docs roughly ties keyword and AI-embedding search. On big-picture questions across many articles it loses (20 to 45% win rate), because it pulls text from a few documents rather than across topics. Picking text for an AI to answer from is not what it does best yet.

**Grouping.** The groups it finds never mix topics. Some topics split into smaller groups, such as mosquitoes and malaria.

To run the tests yourself:

```bash
uv run python bench/fetch_wiki.py   # download the 60 articles
uv run python bench/eval.py         # overall scores
uv run python bench/tokens.py       # token savings
uv run python bench/lazygraphrag.py # answer quality (uses the `claude` CLI and your Claude plan's usage)
```

## Limits

- No scanned PDFs, images, audio or video.
- Tables, headers and footers in `.docx` files are not read.
- It finds ideas by grammar, not meaning, so "solar panels" and "photovoltaic panels" stay separate, and it gets some links wrong.
- Comfortable up to a few thousand documents.

## More

- [docs/technical.md](docs/technical.md): every connection type, how they are scored, the file format, settings, adding your own idea extractor, and the full benchmark numbers.
- [config.example.toml](config.example.toml): every setting with its default.

## License

Apache 2.0, see [LICENSE](LICENSE). The bundled spaCy model is MIT and the wordfreq word lists are CC BY-SA 4.0.
