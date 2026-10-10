"""Small local viewer for graph.json + GRAPH_REPORT.md (NiceGUI + its bundled ECharts; no CDN)."""
import json
from collections import defaultdict
from pathlib import Path

from nicegui import ui

TIER_STYLE = {"EXTRACTED": "solid", "INFERRED": "dashed", "AMBIGUOUS": "dotted"}
TIER_COLOR = {"EXTRACTED": "green", "INFERRED": "blue", "AMBIGUOUS": "orange"}
KINDS = ["folder", "document", "section", "concept"]
SYMBOL = {"folder": "roundRect", "document": "circle", "section": "diamond", "concept": "triangle"}


def load(out: Path) -> tuple[dict, str]:
    data = json.loads((out / "graph.json").read_text(encoding="utf-8"))
    report = out / "GRAPH_REPORT.md"
    return data, report.read_text(encoding="utf-8") if report.exists() else "_No GRAPH_REPORT.md found._"


def run(out: Path, host: str = "127.0.0.1", port: int = 8080, show: bool = True):
    @ui.page("/")
    def page():
        data, report = load(out)  # read per page load, so a rebuild shows up on browser refresh
        nodes = {n["id"]: n for n in data["nodes"]}
        adj = defaultdict(list)
        for e in data["links"]:
            adj[e["source"]].append(e)
            adj[e["target"]].append(e)
        communities = sorted({(n["community"], n.get("community_name", "")) for n in nodes.values()})
        cat_index = {cid: i for i, (cid, _) in enumerate(communities)}
        state = {"tiers": set(TIER_STYLE), "kinds": {"folder", "document", "concept"}, "min_score": 0.0, "selected": None}

        def options() -> dict:
            visible = {i for i, n in nodes.items() if n["kind"] in state["kinds"]}
            links = [e for e in data["links"] if e["confidence"] in state["tiers"]
                     and e["confidence_score"] >= state["min_score"] and e["source"] in visible and e["target"] in visible]
            degree = defaultdict(int)
            for e in links:
                degree[e["source"]] += 1
                degree[e["target"]] += 1
            sel = state["selected"]
            return {
                "tooltip": {},
                "legend": [{"type": "scroll", "orient": "vertical", "left": 0, "top": 0, "data": [name for _, name in communities]}],
                "series": [{
                    "type": "graph", "layout": "force", "roam": True, "draggable": True,
                    "force": {"repulsion": 220, "edgeLength": [40, 140], "gravity": 0.08},
                    "emphasis": {"focus": "adjacency"},
                    "categories": [{"name": name} for _, name in communities],
                    "label": {"position": "right", "fontSize": 11},
                    "data": [{
                        "id": i, "name": nodes[i]["label"], "category": cat_index[nodes[i]["community"]],
                        "symbol": SYMBOL.get(nodes[i]["kind"], "circle"),
                        "symbolSize": 8 + min(28, degree[i] * 2) + (10 if i == sel else 0),
                        "label": {"show": nodes[i]["kind"] in ("document", "folder") or degree[i] >= 6 or i == sel},
                        "itemStyle": {"borderColor": "#000", "borderWidth": 3} if i == sel else {},
                        "value": nodes[i]["kind"],
                    } for i in sorted(visible)],
                    "links": [{
                        "source": e["source"], "target": e["target"], "value": e["relation"],
                        "lineStyle": {"type": TIER_STYLE[e["confidence"]], "opacity": 0.35 + 0.6 * e["confidence_score"],
                                      "width": 1 + 2 * e["confidence_score"], "curveness": 0.05, "color": "source"},
                    } for e in links],
                }],
            }

        def refresh():
            chart.options.clear()
            chart.options.update(options())
            chart.update()
            count.text = f"{len(chart.options['series'][0]['data'])} nodes · {len(chart.options['series'][0]['links'])} edges shown"

        def select(nid: str | None):
            if nid not in nodes:
                return
            state["selected"] = nid
            show_details(nid)
            refresh()

        def show_details(nid: str):
            n = nodes[nid]
            details.clear()
            with details:
                ui.label(n["label"]).classes("text-lg font-bold")
                with ui.row().classes("gap-1"):
                    ui.badge(n["kind"])
                    ui.badge(n.get("community_name", ""), color="grey").classes("whitespace-normal")
                skip = {"id", "label", "norm_label", "community", "community_name", "kind"}
                rows = [{"k": k, "v": str(v)} for k, v in n.items() if k not in skip and v not in (None, "")]
                ui.table(columns=[{"name": "k", "label": "field", "field": "k", "align": "left"},
                                  {"name": "v", "label": "value", "field": "v", "align": "left"}],
                         rows=rows, row_key="k").props("dense flat hide-header").classes("w-full")
                edges = sorted(adj[nid], key=lambda e: (e["confidence"] != "EXTRACTED", -e["confidence_score"]))
                ui.label(f"{len(edges)} connections").classes("font-semibold mt-2")
                for e in edges[:60]:
                    other = e["target"] if e["source"] == nid else e["source"]
                    arrow = "→" if e["source"] == nid else "←"
                    with ui.row().classes("items-center gap-1 w-full no-wrap"):
                        ui.badge(e["confidence"][0], color=TIER_COLOR[e["confidence"]]).tooltip(
                            f"{e['confidence']} · score {e['confidence_score']} · weight {e['weight']}")
                        ui.label(f"{arrow} {e['relation']}").classes("text-xs text-gray-500")
                        ui.link(nodes[other]["label"]).classes("text-sm truncate cursor-pointer").on(
                            "click", lambda o=other: select(o))
                    prov = " ".join(str(x) for x in (e.get("source_file"), e.get("source_location")) if x)
                    if prov:
                        ui.label(prov).classes("text-xs text-gray-400 ml-6 -mt-1")

        def on_click(e):
            if e.data_type == "node":
                select(e.data["id"])

        def toggle(group: str, value: str, on: bool):
            (state[group].add if on else state[group].discard)(value)
            refresh()

        with ui.header().classes("items-center"):
            ui.label("Document graph").classes("text-lg font-bold")
            ui.label(data["graph"].get("root", "")).classes("text-sm opacity-80")
            ui.space()
            count = ui.label()

        with ui.left_drawer(value=True).classes("bg-gray-50 w-64"):
            ui.select({i: f"{n['label']} ({n['kind']})" for i, n in sorted(nodes.items(), key=lambda kv: kv[1]["label"].lower())},
                      label="Find node", with_input=True, on_change=lambda e: select(e.value)).classes("w-full")
            ui.label("Edge tiers").classes("font-semibold mt-4")
            for t in TIER_STYLE:
                ui.checkbox(f"{t} ({TIER_STYLE[t]})", value=True, on_change=lambda e, t=t: toggle("tiers", t, e.value))
            ui.label("Node kinds").classes("font-semibold mt-4")
            for k in KINDS:
                ui.checkbox(k, value=k in state["kinds"], on_change=lambda e, k=k: toggle("kinds", k, e.value))
            ui.label("Min confidence").classes("font-semibold mt-4")
            ui.slider(min=0, max=1, step=0.05, value=0).props("label").on(
                "update:model-value", lambda e: (state.update(min_score=float(e.args)), refresh()), throttle=0.3)
            ui.label(f"{len(data['hyperedges'])} hyperedges · {len(communities)} communities").classes("text-xs text-gray-500 mt-4")

        with ui.tabs() as tabs:
            graph_tab, report_tab = ui.tab("Graph"), ui.tab("Report")
        with ui.tab_panels(tabs, value=graph_tab).classes("w-full"):
            with ui.tab_panel(graph_tab).classes("p-0"):
                with ui.row().classes("w-full no-wrap gap-2"):
                    chart = ui.echart(options(), on_point_click=on_click).classes("grow").style("height: calc(100vh - 140px)")
                    with ui.scroll_area().classes("w-96 shrink-0").style("height: calc(100vh - 140px)"):
                        details = ui.column().classes("w-full p-2")
                        with details:
                            ui.label("Click a node to see its fields, connections and provenance.").classes("text-gray-500")
            with ui.tab_panel(report_tab):
                ui.markdown(report).classes("w-full")
        refresh()

    ui.run(host=host, port=port, show=show, reload=False, title="Document graph")
