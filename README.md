# pensieve-docs

pensieve-docs turns a folder of documents into a knowledge graph you can query. It reads your notes, papers and memos, pulls out the links, citations and concepts in them, and writes one `graph.json` that records where every edge came from, down to the page or line.

It runs on your machine. A default build makes no network calls and no LLM calls, and a test fails if that ever changes.

The idea comes from [graphify](https://github.com/Graphify-Labs/graphify), which does this for code using syntax trees. Documents have no syntax tree, so pensieve-docs uses a deterministic parser for structure (headings, links, DOIs, reference lists) and spaCy for concepts. The output uses graphify's `graph.json` shape, so tools that read one can read the other.

Secrets and personal data are blanked out of the text as soon as a file is parsed, before anything is cached, indexed or written: passwords, API keys and tokens, private keys, card numbers, email addresses and phone numbers. Your files are not changed. The report lists how many of each were found, never the values.

Supported inputs are `.md`, `.markdown`, `.txt`, `.pdf` (text layer only, no OCR) and `.docx`. Anything else is skipped and listed in the report.

## Setup

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/shreyas-dhakal/pensieve-docs
cd pensieve-docs
uv sync                 # installs dependencies, including the spaCy en_core_web_sm model (one-time download)
uv run pytest -q        # 36 tests: parsing, edge tiers and evidence, incremental rebuild, no network
```

For MiniLM document similarity instead of TF-IDF, run `uv sync --extra embeddings`, download the model once while online
(`uv run python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"`),
then set `similarity.method = "minilm"`. Builds run with `HF_HUB_OFFLINE=1`, so a missing model makes the build fail instead of downloading it.

## Usage

```bash
uv run pensieve-docs build <folder> [--config pensieve.toml] [--out pensieve-out] [-v]
```

The graph and report go to `./pensieve-out/`, and the per-file cache to `pensieve-out/.cache/`. A `pensieve.toml` inside the folder is picked up automatically. [`config.example.toml`](config.example.toml) lists every setting with its default.

## Related documents

Once a folder is built, `related` finds the documents that share concepts with a file or with any text, and says why:

```
$ echo "Should we add a home battery? The payback period depends on the net metering export rate. Backup power during a grid outage matters too." | uv run pensieve-docs related -
1.17  Review meeting, June 2024  (meetings/2024-06-review.md)
        grid outage  [meetings/2024-06-review.md L5]
        backup power  [meetings/2024-06-review.md L5]
        payback period  [meetings/2024-06-review.md L5]
0.93  Battery Storage  (notes/battery-storage.md)
        home battery  [notes/battery-storage.md L3]
        ...
```

Pass a file path instead of `-` to use a document's text (any supported type); a file from the indexed folder is left out of its own results. `-k`, `--min-score` and `--json` adjust the output. Matches below 0.2 are hidden by default, so unrelated text returns nothing.

From Python, load the graph once and query as often as you like:

```python
from pensieve_docs.related import Index

idx = Index("pensieve-out/graph.json")
idx.related("thinking about a home battery for backup power", k=5)
# [{"id", "label", "source_file", "score", "reasons": [{"concept", "where"}, ...]}, ...]
```

A query matches the graph's known concepts in the text (surface words and lemmas, longest match first) and scores each document by the TF-IDF weight they share. A paragraph takes a few milliseconds once the spaCy model is loaded; a 30,000-character article takes about a third of a second.

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
9 files (9 parsed, 0 cached, 2 skipped) -> 69 nodes, 293 edges, 14 hyperedges, 5 communities in 0.6s
wrote pensieve-out/graph.json and GRAPH_REPORT.md

$ uv run pensieve-docs build examples/sample        # nothing changed, so everything comes from the cache
9 files (0 parsed, 9 cached, 2 skipped) -> 69 nodes, 293 edges, 14 hyperedges, 5 communities in 0.1s
```

Part of the generated `GRAPH_REPORT.md`:

| # | name | size | docs | cohesion | top documents |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | heat pump / Rooftop solar / solar panels | 18 | 2 | 0.444 | Kickoff meeting, May 2024, Heat Pumps |
| 1 | utility company / battery storage / net metering | 16 | 2 | 0.396 | Solar Basics, Grid Integration of Residential Solar |
| 2 | home battery / lithium-ion battery / net metering export rate | 13 | 1 | 0.4 | Review meeting, June 2024 |

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
| Redact | `redact.py` | Runs inside `parse_file`, before chunking. Each rule is a pattern plus a check: card numbers must pass the Luhn checksum and not be ISBNs, phone numbers need a `+`, a bracketed area code, a leading 0 or the 3-3-4 North American shape, and a `token:` value must be 16+ characters, so prose such as "token: a unit of text" is kept. On the 60 Wikipedia articles in `bench/` it blanks nothing. `python -m pensieve_docs.redact` runs a self-check. |
| Parse | `parse.py` | Markdown and text: ATX and setext headings, frontmatter title, links, wikilinks. DOCX: heading styles and hyperlinks. PDF: text layer, outline as sections, link annotations. All types: URLs, DOIs, `[n]` references and References sections. Text is chunked by section, long sections are split on paragraph boundaries, and each chunk keeps its page, line and character offset. PDFs with no text layer are skipped. |
| Concepts | `concepts.py` | spaCy entities and noun chunks, lemmatised and lowercased. Reference lists are skipped, since they already become `cites` edges. A one-word heading ("Economics") only counts if the body uses the word, and a term found only as its own document's title is left to the document node. `X of Y` phrases are joined when they fit in `max_tokens`. Boilerplate headings are dropped. Per sentence it also records concept pairs joined by a short dependency path (with the verb on the path and whether it is negated or hedged), "long form (ABBR)" abbreviations (Schwartz–Hearst) and is-a patterns ("X such as Y", Hearst). |
| Graph | `edges.py` | Every concept found anywhere is matched again across all chunks, longest match first, so a term counts wherever its text appears. Near-duplicates are folded with difflib, and a concept with an extra modifier folds into the shorter one (`shortlisted heat pump` into `heat pump`, `net metering rules` into `net metering`). An abbreviation folds into its long form first (`LLM` into `large language model`). Concepts made only of everyday English words (wordfreq Zipf at or above `generic_zipf`) are dropped, and so are concepts in more than half the documents or 30% of the chunks of a folder with 10+ documents. Then TF-IDF concept weights, `is_a` from Hearst patterns, co-occurrence (see [Edge scoring](#edge-scoring)), concepts shared between documents and sections, document similarity and hyperedges. |
| Cluster | `analyze.py` | Louvain communities (networkx, seed 42) over the non-AMBIGUOUS edges, each named after its top concepts by PageRank. God nodes are concepts ranked by degree plus PageRank. Documents are left out of that ranking because, as graphify found for files, they collect edges mechanically. |
| Output | `output.py` | `graph.json` and `GRAPH_REPORT.md` |
| Cache | `build.py` | Per-file results are keyed by `sha256(parser version + extractor config + file bytes)`. Unchanged files are never parsed again and stale entries are pruned. Corpus-wide steps (TF-IDF, PMI, similarity, clustering) rerun every time because they depend on every document; they take well under a second on the sample. |

## Nodes and edges

| Node `kind` | `file_type` | Fields |
| --- | --- | --- |
| `folder` | `folder` | One per directory. The root is `folder_root`. |
| `document` | `document` (`paper` for PDFs) | A pointer to the file: `source_file`, `doc_type`, `hash`, `mtime`, `size`, `n_chunks`, `doi`, `external_links`, `redactions` (counts by kind). The text itself is not stored. |
| `section` | `document` | A heading: `level`, `chunk`, `source_location` |
| `concept` | `concept` | `norm`, `doc_count`, `score` (summed TF-IDF) |

| relation | tier | weight | rule |
| --- | --- | --- | --- |
| `contains` | EXTRACTED | 1 | folder to subfolder or document |
| `has_section` | EXTRACTED | 1 | document to section |
| `links_to` | EXTRACTED | 1 | relative link or wikilink that resolves to a document in the folder |
| `cites` | EXTRACTED | 1 | a DOI equal to another document's own DOI, or a reference entry containing another document's title (longest title wins) |
| `mentions` | INFERRED | TF-IDF, or a count | document to its top `max_per_doc` concepts; section to those of its document's concepts that appear in it (its chunk and the continuation chunks after it), weight = count |
| `co_occurs` | INFERRED or AMBIGUOUS | PMI | concepts sharing at least `min_count` chunks, top-k partners each; AMBIGUOUS below `npmi_threshold` or when found in fewer than `min_docs_inferred` documents |
| `is_a` | INFERRED | matches | specific to general concept, from "X such as Y", "Y and other X", "X including Y" and "Y is a X"; 0.65, or 0.85 when matched in 3+ sentences |
| `conceptually_related_to` | INFERRED | 1 | a concept whose words contain another concept (`net metering export rate` to `net metering`) |
| `shares_concepts` | INFERRED | overlap | document to its top 5 by the score `related.py` uses (summed min TF-IDF over shared concepts), at least `related.MIN_SCORE`; `shared` lists the top 3 concepts and where each document mentions them |
| `similar_to` | INFERRED or AMBIGUOUS | cosine | top-k neighbours; INFERRED at or above `threshold`, AMBIGUOUS between `ambiguous_min` and `threshold` |

A concept is kept if it appears in at least `min_docs` documents or ranks in the top `keep_top_single` of its own document. A concept in at least `hyperedge_min_docs` documents also gets a hyperedge `{concept, docs…}` with `relation: participate_in`.

Each pair of nodes has at most one edge. Every rule that produces the pair adds an entry to the edge's `evidence` list (up to `max_evidence`, strongest first), and the edge's own fields are copied from the strongest entry: the higher tier (EXTRACTED, then INFERRED, then AMBIGUOUS), then the higher `confidence_score`, then whichever came first. So two documents that cite each other share one `cites` edge with two entries, and a `net metering export rate` / `net metering` pair keeps its co-occurrence statistics next to the containment rule. Read `e.get("evidence", [e])` to find every rule behind an edge; `relation` alone shows only the strongest.

## Edge scoring

Co-occurrence counts chunks. A shared chunk counts 1, or, if the two concepts are in one sentence with at most `max_path` dependency arcs between them, the best of

$$\omega = (1 + 0.5) \cdot (1 - 0.7\,\nu) \cdot (1 - 0.3\,\eta)$$

over those sentences, where $\nu$ is 1 when the path is negated ("do not reduce", "no", "never", "without") and $\eta$ is 1 when it is hedged (`hedge_words`: "may", "might", "could", "possibly", "potentially", "suggest"). The sum over shared chunks is $\tilde N_{ab}$ (`wcount`). With $N$ chunks and $N_a$ chunks holding $a$:

$$\text{PMI} = \ln\frac{\tilde N_{ab}/N}{(N_a/N)(N_b/N)}, \qquad \text{NPMI} = \min\left(1, \frac{\text{PMI}}{-\ln(\tilde N_{ab}/N)}\right)$$

Pairs sharing fewer than `min_count` chunks, or with PMI at or below 0, are dropped. Each concept keeps its `top_k` partners ranked by $\text{NPMI} \cdot \ln(1 + n_{docs})$, so a pair many documents share beats one a single document repeats. A kept pair is INFERRED when NPMI is at least `npmi_threshold` and it is found in at least `min_docs_inferred` documents (when the folder has more than one), otherwise AMBIGUOUS. `confidence_score` maps NPMI onto graphify's 0.55 to 0.95 scale.

`broader` is $b$ when $P(b \mid a) = N_{ab}/N_a \ge 0.8$ and $P(a \mid b) < 0.8$, with plain chunk counts.

Two documents' overlap is $S(d, d') = \sum_t \min(w_{d,t}, w_{d',t})$ over the concepts both mention, with $w$ their L2-normalised TF-IDF weights (`edges.overlap`). `related.py` ranks query results by the same sum, and `shares_concepts` has `confidence_score` = rubric$(S / \max S)$.

Louvain and PageRank weigh edges by `confidence_score` (`analyze.WEIGHT`). Weighing by `strength` instead split topics into more communities on the benchmark (inverse purity 0.70 to 0.65), because every structural EXTRACTED edge then weighs 1 against NPMI values around 0.3 to 0.6.

The settings, all in [`config.example.toml`](config.example.toml):

| key | default | effect |
| --- | --- | --- |
| `max_evidence` | 10 | evidence entries per edge, and shared chunks listed per `co_occurs` |
| `concepts.max_path` | 4 | longest dependency path, in arcs, that links two concepts |
| `concepts.hedge_words` | may, might, could, possibly, potentially, suggest | words that mark a path as hedged |
| `concepts.abbrev_max_zipf` | 4.0 | abbreviations this common as English words (`FIT`, `COP`) are not merged, since matching ignores case |
| `concepts.max_doc_frac`, `max_chunk_frac`, `common_min_docs` | 0.5, 0.3, 10 | drop concepts this widespread in a folder of at least `common_min_docs` documents; listed in `GRAPH_REPORT.md` |
| `cooccur.syntax_boost`, `negation_penalty`, `hedge_penalty` | 0.5, 0.7, 0.3 | the constants in $\omega$; all 0 gives plain chunk counts |
| `cooccur.label_share`, `label_min` | 0.6, 2 | when a verb becomes `relation_label` |
| `cooccur.min_docs_inferred` | 2 | 1 turns the single-document rule off |
| `shares.top_k` | 5 | `shares_concepts` partners per document |
| `hierarchy.broader_min` | 0.8 | threshold for `broader` |
| `hierarchy.isa_score`, `isa_strong_score`, `isa_min_strong` | 0.65, 0.85, 3 | `is_a` scores |

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
             "strength": 1.0, "source_file": "papers/battery-economics.docx", "source_location": "¶5", "chunk": 2,
             "ref": "10.5555/grid.solar.2024", "evidence": [{"relation": "cites", "...": "..."}, {"relation": "similar_to", "...": "..."}]}],
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
| `weight` | the raw TF-IDF, PMI, cosine, overlap or count, or 1 |
| `strength` | the same evidence on one 0 to 1 scale: 1 for EXTRACTED, NPMI (floored at 0) for `co_occurs`, weight over the document's (or section's) largest for `mentions`, cosine for `similar_to`, overlap over the largest overlap for `shares_concepts`, the score for `is_a` and `conceptually_related_to` |
| `source_file`, `chunk`, `source_location` | where the edge came from: `p3` is a PDF page, `L12` a line, `¶5` a docx paragraph |
| `evidence` | every rule behind the edge, strongest first, each with all of the fields above plus its own `source` and `target` (direction matters for `cites` and `is_a`). Left out when the edge itself is the only entry, so read it as `e.get("evidence", [e])` |

Some relations carry more:

| relation | fields |
| --- | --- |
| `links_to`, `cites` | `ref`: the link, DOI or reference line that made the edge (up to 200 characters) |
| `mentions` | `tfidf` (documents) or `count` (sections) |
| `co_occurs` | `pmi`, `npmi`, `count` (shared chunks), `wcount` (weighted count, below), `n_docs`, `max_doc_share` (largest share of `count` from one document), `chunks` (`[file, chunk, location]` for up to `max_evidence` shared chunks), `syntactic` (mentions joined by a short dependency path), `verbs` (verb lemma on those paths: count, top 5), `negated_frac`, `hedged_frac`, `relation_label` (the verb, when one verb has at least 60% and 2 of them; `relation` stays `co_occurs`), `p_b_given_a` and `p_a_given_b` (a = `source`, over chunks), `broader` (the concept that nearly always comes with the other, when one exists) |
| `is_a` | `matches` (sentences matched), `sentence` (character span in the chunk's cleaned text) |
| `shares_concepts` | `overlap`, `shared`: `[{concept, weight, locations: [[file, chunk, location] in each document]}]` |
| `similar_to` | `cosine`, `method` |

`relation_label` and `broader` say nothing about direction beyond what they name: `source` and `target` of a `co_occurs` edge are in name order, not subject and object.

Node IDs follow graphify's convention: the path relative to the folder, without its extension, lowercased, with each run of non-alphanumeric characters replaced by `_`. Concepts are `concept_<norm>`, folders `folder_<path>` and sections `<doc id>#<heading>`. A document whose ID would start with `concept_` or `folder_` gets a `doc_` prefix, so IDs never collide. Nodes, links and keys are sorted, so two builds of the same folder give byte-identical JSON apart from `built_at`.

## Extending

Concept extraction is pluggable. `concepts.extractors` is a list of `"module:function"` paths. Each one is called once per changed file as `fn(chunks, cfg)` and returns:

```python
{"terms": [{norm: count}, ...],          # one dict per chunk
 "labels": {norm: {surface_form: count}},
 "bow": {lemma: count},                  # for TF-IDF similarity
 # optional:
 "rels": [[a, b, verb or None, negated, hedged, chunk], ...],  # concept pairs joined by a short dependency path
 "abbrevs": [[short norm, long norm], ...],                    # "long form (ABBR)"
 "isa": [[specific, general, chunk, sentence start, sentence end], ...]}  # Hearst patterns
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
| build time | 49 s cold, 6 s cached |
| top-3 related articles in the same topic | 98% |
| top-3 related articles that Wikipedia links to | 83% |
| `shares_concepts` edges (score 0.2 or more) | 117 of 180 possible top-3 slots filled; 100% same topic, 94% linked on Wikipedia |
| communities | 13 for 6 topics; each is one topic (purity 1.00), and topics split into subtopics such as mosquitoes and malaria (inverse purity 0.70; 0.69 averaged over 10 Louvain seeds, since a single seed moves it by several points) |
| `co_occurs` | 2604 INFERRED and 804 AMBIGUOUS; 91% of each join two concepts whose documents share a topic |
| score cut-off 0.2 | keeps 66% of suggestions, all of them on topic |
| `Index.related` with each article's full text as the query | 99% same topic, 87% linked on Wikipedia |

Six unrelated topics make this an easy test. The weak spots are generic words that survive as concepts (`technique`, `output`, `characteristics`).

## Known limits

- No OCR, images, audio or video. Scanned PDFs are skipped and reported.
- DOCX tables, headers and footers are not read. Markdown tables are read as plain text.
- PDFs are chunked one page at a time, with headings from the PDF outline. A PDF without an outline gets untitled page chunks.
- Document similarity compares every pair (O(n²)). That is fine for thousands of documents; past that it needs an approximate nearest-neighbour index.
- `graph.json` is large: every edge keeps an `evidence` list, and sections have `mentions` edges. The 60-article benchmark writes 10 MB (4.7 MB before evidence and section mentions were added). Leaving section mentions out halves that but splits topics into more communities (inverse purity 0.70 to 0.67).
- Verbs, negation, hedging and is-a patterns come from `en_core_web_sm`'s parse and tags, which miss things ("diseases such as malaria" is lost when malaria is tagged as an adjective) and invent others (`is_a` from "input" to "tenor").
- Concepts are only as good as `en_core_web_sm` noun chunks. Synonyms such as "solar panels" and "photovoltaic panels" stay separate. A larger `spacy_model` or your own extractor will do better.

## License

Apache 2.0, see [LICENSE](LICENSE). The bundled spaCy model is MIT and the wordfreq word lists are CC BY-SA 4.0.
