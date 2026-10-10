# pensieve-docs

pensieve-docs turns a folder of documents into a knowledge graph you can query. It reads your notes, papers and memos, pulls out the links, citations and concepts in them, and writes one `graph.json` that records where every edge came from, down to the page or line.

It runs on your machine. A default build makes no network calls and no LLM calls, and a test fails if that ever changes.

The idea comes from [graphify](https://github.com/Graphify-Labs/graphify), which does this for code using syntax trees. Documents have no syntax tree, so pensieve-docs uses a deterministic parser for structure (headings, links, DOIs, reference lists) and spaCy for concepts. The output uses graphify's `graph.json` shape, so tools that read one can read the other.

Supported inputs are `.md`, `.markdown`, `.txt`, `.pdf` (text layer only, no OCR) and `.docx`. Anything else is skipped and listed in the report.

## Setup

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/shreyas-dhakal/pensieve-docs
cd pensieve-docs
uv sync                 # installs dependencies, including the spaCy en_core_web_sm model (one-time download)
uv run pytest -q        # 20 tests: parsing, edge tiers, incremental rebuild, no network
```

For MiniLM document similarity instead of TF-IDF, run `uv sync --extra embeddings`, download the model once while online
(`uv run python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"`),
then set `similarity.method = "minilm"`. Builds run with `HF_HUB_OFFLINE=1`, so a missing model makes the build fail instead of downloading it.

## Usage

```bash
uv run pensieve-docs build <folder> [--config pensieve.toml] [--out pensieve-out] [-v]
```

The graph and report go to `./pensieve-out/`, and the per-file cache to `pensieve-out/.cache/`. A `pensieve.toml` inside the folder is picked up automatically. [`config.example.toml`](config.example.toml) lists every setting with its default.

## Viewer

There is a small local viewer built with [NiceGUI](https://nicegui.io). Its charts are bundled, so it makes no CDN calls.

```bash
uv sync --extra ui
uv run pensieve-docs ui             # reads ./pensieve-out/graph.json; --out DIR, --port N, --no-show
```

It serves on `127.0.0.1:8080`. The graph tab draws a force layout where colour is the community and shape is the node kind (folder ■, document ●, section ◆, concept ▲). Solid, dashed and dotted lines are EXTRACTED, INFERRED and AMBIGUOUS edges, and line width follows confidence. You can filter by tier, node kind and minimum confidence, or search for a node. Clicking a node lists its fields and every connection with tier, score and source location; clicking a connection jumps to the other node. The report tab renders `GRAPH_REPORT.md`.

The viewer only reads. Rerun `build` and reload the page to see changes.

## Example

`examples/sample/` is a small home-energy research folder. It has markdown notes with links and wikilinks, a `.txt` with setext headings, two meeting notes, a PDF paper with an outline, a DOI and a reference list, a `.docx` memo with a DOI hyperlink, a scanned PDF with no text, and a `.png`.

```
$ uv run pensieve-docs build examples/sample
WARNING skip scans/scanned-invoice.pdf: no text layer (scanned PDF?) - OCR is out of scope
9 files (9 parsed, 0 cached, 2 skipped) -> 69 nodes, 196 edges, 14 hyperedges, 6 communities in 0.9s
wrote pensieve-out/graph.json and GRAPH_REPORT.md

$ uv run pensieve-docs build examples/sample        # nothing changed, so everything comes from the cache
9 files (0 parsed, 9 cached, 2 skipped) -> 69 nodes, 196 edges, 14 hyperedges, 6 communities in 0.1s
```

Part of the generated `GRAPH_REPORT.md`:

| # | name | size | docs | cohesion | top documents |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | inverter / heat pump / Rooftop solar | 17 | 2 | 0.426 | Solar Basics, Home Energy Retrofit |
| 2 | solar energy / net metering export rate / payback period | 12 | 1 | 0.352 | Battery Storage Economics |
| 4 | lithium-ion battery / grid outage / backup power | 9 | 1 | 0.306 | Review meeting, June 2024 |

| from | relation | to | confidence | provenance |
| --- | --- | --- | ---: | --- |
| Battery Storage | cites | Battery Storage Economics | 1.0 | notes/battery-storage.md L19 |
| Battery Storage Economics | cites | Grid Integration of Residential Solar | 1.0 | papers/battery-economics.docx ¶5 |
| Home Energy Retrofit | links_to | Heat Pumps | 1.0 | index.md L9 |
| grid | co_occurs | net metering | 0.75 | notes/solar-basics.md L9 |

The `.docx` to PDF citation came from a DOI hyperlink in the memo that matches the DOI printed on the PDF's first page. The markdown to `.docx` one came from a `[2] Battery Storage Economics, internal memo.` line in a References section.

## How it works

| Step | Module | What it does |
| --- | --- | --- |
| Detect | `detect.py` | Walks the folder, skipping hidden and excluded directories, and sorts files into supported and skipped. |
| Parse | `parse.py` | Markdown and text: ATX and setext headings, frontmatter title, links, wikilinks. DOCX: heading styles and hyperlinks. PDF: text layer, outline as sections, link annotations. All types: URLs, DOIs, `[n]` references and References sections. Text is chunked by section, long sections are split on paragraph boundaries, and each chunk keeps its page, line and character offset. PDFs with no text layer are skipped. |
| Concepts | `concepts.py` | spaCy entities and noun chunks, lemmatised and lowercased. Reference lists are skipped, since they already become `cites` edges. A one-word heading ("Economics") only counts if the body uses the word, and a term found only as its own document's title is left to the document node. `X of Y` phrases are joined when they fit in `max_tokens`. Boilerplate headings are dropped. |
| Graph | `edges.py` | Every concept found anywhere is matched again across all chunks, longest match first, so a term counts wherever its text appears. Near-duplicates are folded with difflib, and a concept with an extra modifier folds into the shorter one (`shortlisted heat pump` into `heat pump`, `net metering rules` into `net metering`). Concepts made only of everyday English words (wordfreq Zipf at or above `generic_zipf`) are dropped. Then TF-IDF concept weights, PMI co-occurrence, document similarity and hyperedges. |
| Cluster | `analyze.py` | Louvain communities (networkx, seed 42) over the non-AMBIGUOUS edges, each named after its top concepts by PageRank. God nodes are concepts ranked by degree plus PageRank. Documents are left out of that ranking because, as graphify found for files, they collect edges mechanically. |
| Output | `output.py` | `graph.json` and `GRAPH_REPORT.md` |
| Cache | `build.py` | Per-file results are keyed by `sha256(parser version + extractor config + file bytes)`. Unchanged files are never parsed again and stale entries are pruned. Corpus-wide steps (TF-IDF, PMI, similarity, clustering) rerun every time because they depend on every document; they take well under a second on the sample. |

## Nodes and edges

| Node `kind` | `file_type` | Fields |
| --- | --- | --- |
| `folder` | `folder` | One per directory. The root is `folder_root`. |
| `document` | `document` (`paper` for PDFs) | A pointer to the file: `source_file`, `doc_type`, `hash`, `mtime`, `size`, `n_chunks`, `doi`, `external_links`. The text itself is not stored. |
| `section` | `document` | A heading: `level`, `chunk`, `source_location` |
| `concept` | `concept` | `norm`, `doc_count`, `score` (summed TF-IDF) |

| relation | tier | weight | rule |
| --- | --- | --- | --- |
| `contains` | EXTRACTED | 1 | folder to subfolder or document |
| `has_section` | EXTRACTED | 1 | document to section |
| `links_to` | EXTRACTED | 1 | relative link or wikilink that resolves to a document in the folder |
| `cites` | EXTRACTED | 1 | a DOI equal to another document's own DOI, or a reference entry containing another document's title (longest title wins) |
| `mentions` | INFERRED | TF-IDF | document to its top `max_per_doc` concepts |
| `co_occurs` | INFERRED or AMBIGUOUS | PMI | concepts sharing at least `min_count` chunks, top-k partners each; AMBIGUOUS below `npmi_threshold` |
| `conceptually_related_to` | INFERRED | 1 | a concept whose words contain another concept (`net metering export rate` to `net metering`) |
| `similar_to` | INFERRED or AMBIGUOUS | cosine | top-k neighbours; INFERRED at or above `threshold`, AMBIGUOUS between `ambiguous_min` and `threshold` |

A concept is kept if it appears in at least `min_docs` documents or ranks in the top `keep_top_single` of its own document. A concept in at least `hyperedge_min_docs` documents also gets a hyperedge `{concept, docs…}` with `relation: participate_in`.

Each pair of nodes has at most one edge. When two rules produce the same pair, the stronger tier wins (EXTRACTED, then INFERRED, then AMBIGUOUS), so two documents that cite each other share one `cites` edge.

## `graph.json`

The file has the same shape as graphify's, so `networkx.node_link_graph(data, edges="links")` or a graphify-style viewer can load it:

```json
{
  "directed": false, "multigraph": false,
  "graph": {"hyperedges": [...], "root": "...", "built_at": "...", "built_at_commit": null, "generator": "pensieve-docs 0.1.0", "similarity_method": "tfidf"},
  "nodes": [{"id": "papers_battery_economics", "label": "Battery Storage Economics", "community": 0, "community_name": "...",
             "file_type": "document", "kind": "document", "norm_label": "...", "source_file": "papers/battery-economics.docx",
             "source_location": null, "doc_type": "docx", "hash": "...", "mtime": "...", ...}],
  "links": [{"source": "papers_battery_economics", "target": "papers_grid_integration", "relation": "cites",
             "confidence": "EXTRACTED", "confidence_score": 1.0, "weight": 1.0,
             "source_file": "papers/battery-economics.docx", "source_location": "¶5", "chunk": 2, "evidence": "10.5555/grid.solar.2024"}],
  "hyperedges": [{"id": "concept_home_battery_docs", "label": "home battery (7 docs)", "nodes": ["concept_home_battery", "..."],
                  "relation": "participate_in", "confidence": "INFERRED", "confidence_score": 0.85,
                  "source_file": "index.md", "source_files": ["..."]}],
  "built_at_commit": null
}
```

Every edge carries these fields:

| field | meaning |
| --- | --- |
| `relation` | edge type, from the table above |
| `confidence` | tier: `EXTRACTED`, `INFERRED` or `AMBIGUOUS` |
| `confidence_score` | 1.0 for EXTRACTED; graphify's scale of 0.55, 0.65, 0.75, 0.85 or 0.95 for INFERRED; 0.1 to 0.3 for AMBIGUOUS |
| `weight` | the raw TF-IDF, PMI or cosine value, or 1 |
| `source_file`, `chunk`, `source_location` | where the edge came from: `p3` is a PDF page, `L12` a line, `¶5` a docx paragraph |

Node IDs follow graphify's convention: the path relative to the folder, without its extension, lowercased, with each run of non-alphanumeric characters replaced by `_`. Concepts are `concept_<norm>`, folders `folder_<path>` and sections `<doc id>#<heading>`. A document whose ID would start with `concept_` or `folder_` gets a `doc_` prefix, so IDs never collide. Nodes, links and keys are sorted, so two builds of the same folder give byte-identical JSON apart from `built_at`.

## Extending

Concept extraction is pluggable. `concepts.extractors` is a list of `"module:function"` paths. Each one is called once per changed file as `fn(chunks, cfg)` and returns:

```python
{"terms": [{norm: count}, ...],          # one dict per chunk
 "labels": {norm: {surface_form: count}},
 "bow": {lemma: count}}                  # for TF-IDF similarity
```

Results from all extractors are summed and cached with the file, so an extractor must give the same output for the same file.

There is also an `enrich = ["module:function"]` hook, called as `fn(graph, cfg)` after the graph is built and before clustering. Nothing uses it yet. `graph.nodes`, `graph.edges` and `graph.hyperedges` are plain dicts, and `graph.edge(...)` applies the tier rules. An LLM enricher could use it to add typed relations between the top nodes as INFERRED or AMBIGUOUS edges with provenance. It is empty by default, so builds stay offline unless you configure one.

## Benchmark

`bench/` scores the graph against a labelled corpus: 60 Wikipedia articles in six topics (malaria and mosquitoes, home energy, machine learning, coffee, Roman history, jazz). The articles are CC BY-SA, so they are downloaded rather than committed.

```bash
uv run python bench/fetch_wiki.py   # ~2.6 MB into bench/data/wiki, with each article's topic and the links between them
uv run python bench/eval.py         # builds the graph and scores it
```

It reports whether communities match topics, and whether each article's top 3 related articles (ranked by shared concepts, the way a "you might want this" suggestion would be) share its topic or are linked from it on Wikipedia. Current numbers:

| | |
| --- | --- |
| build time | 46 s cold, 6 s cached |
| top-3 related articles in the same topic | 98% |
| top-3 related articles that Wikipedia links to | 83% |
| communities | 16 for 6 topics; each is one topic (purity 1.00), and topics split into subtopics such as mosquitoes and malaria |
| score cut-off 0.2 | keeps 66% of suggestions, all of them on topic |

Six unrelated topics make this an easy test. The weak spots are generic words that survive as concepts (`technique`, `output`, `characteristics`).

## Known limits

- No OCR, images, audio or video. Scanned PDFs are skipped and reported.
- DOCX tables, headers and footers are not read. Markdown tables are read as plain text.
- PDFs are chunked one page at a time, with headings from the PDF outline. A PDF without an outline gets untitled page chunks.
- Document similarity compares every pair (O(n²)). That is fine for thousands of documents; past that it needs an approximate nearest-neighbour index.
- Concepts are only as good as `en_core_web_sm` noun chunks. Synonyms such as "solar panels" and "photovoltaic panels" stay separate. A larger `spacy_model` or your own extractor will do better.

## License

Apache 2.0, see [LICENSE](LICENSE). The bundled spaCy model is MIT and the wordfreq word lists are CC BY-SA 4.0.
