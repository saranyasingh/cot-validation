"""Write a self-contained interactive HTML view of a graph (Cytoscape.js + dagre layout)."""

import json

from graph_pipeline.compile_tff import compile_graph, compile_statement, compile_step
from graph_pipeline.model import Graph
from graph_pipeline.prove import prover_verdict


def _safe(fn, *args) -> str:
    try:
        return fn(*args)
    except Exception as e:  # a broken node should still be viewable
        return f"% could not compile: {e!r}"


def graph_payload(graph: Graph) -> dict:
    elements = []
    for sid, st in graph.statements.items():
        elements.append({"data": {
            "id": sid, "type": "statement", "shape": st.get("shape"),
            "label_kind": st.get("label"), "kind": st.get("kind") or "",
            "unstated": st.get("label") == "assumption" and not st.get("stated_in_cot"),
            "display": f"{sid}  {st.get('text') or st.get('english') or ''}",
        }})
    for step_id, step in graph.steps.items():
        elements.append({"data": {
            "id": step_id, "type": "step", "status": step.get("status", "unchecked"),
            "display": f"{step_id}\n{step.get('move', '')}",
        }})
        for p in step.get("premises", []):
            if p in graph.statements:
                elements.append({"data": {"id": f"{p}->{step_id}", "source": p, "target": step_id,
                                          "role": "premise"}})
        w = step.get("warrant")
        if w in graph.statements:
            elements.append({"data": {"id": f"{w}=>{step_id}", "source": w, "target": step_id,
                                      "role": "warrant"}})
        c = step.get("conclusion")
        if c in graph.statements:
            elements.append({"data": {"id": f"{step_id}->{c}", "source": step_id, "target": c,
                                      "role": "conclusion"}})

    details = {}
    for sid, st in graph.statements.items():
        details[sid] = {**st, "tptp": _safe(compile_statement, graph, sid,
                                            "lemma" if st.get("label") == "derived" else "axiom")}
    for step_id, step in graph.steps.items():
        details[step_id] = {**step, "tptp": _safe(compile_step, graph, step_id)}

    return {
        "elements": elements,
        "details": details,
        "question": graph.question,
        "cot": graph.cot,
        "errors": graph.errors,
        "prover_verdict": prover_verdict(graph),
        "dictionary": {"sorts": graph.sorts, "entities": graph.entities,
                       "relations": graph.relations},
        "tptp": _safe(compile_graph, graph),
        "oplog": graph.oplog,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Inference graph</title>
<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         font-size: 13px; color: #222; display: flex; height: 100vh; }
  #cy { flex: 1; background: #fafafa; }
  #side { width: 460px; border-left: 1px solid #ddd; overflow-y: auto; padding: 14px 16px; }
  h2 { font-size: 15px; margin: 4px 0 10px; }
  h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #666;
       margin: 16px 0 6px; }
  pre { background: #f3f3f3; padding: 8px; border-radius: 4px; white-space: pre-wrap;
        word-break: break-word; font-size: 11.5px; margin: 0; }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 10px; margin: 0 4px 4px 0;
           font-size: 11px; border: 1px solid #ccc; background: #fff; }
  .text { line-height: 1.45; }
  .err { color: #b00020; }
  .legend span { display: inline-block; margin: 0 10px 4px 0; }
  .sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px;
        vertical-align: -2px; margin-right: 4px; border: 1px solid #888; }
  button { font-size: 12px; margin-bottom: 8px; }
</style>
</head>
<body>
<div id="cy"></div>
<div id="side"></div>
<script>
const DATA = __PAYLOAD__;

const KIND_COLORS = {
  derived: "#f6e27a", from_problem: "#9fd8a0", math_principle: "#9cc3ea",
  definitional: "#b8a9e6", commonsense: "#8fd3d0", closure: "#f2b880",
  normative: "#e8a0c8", stipulative: "#f08a8a", unjustified: "#c4c4c4", "": "#dddddd"
};
const STATUS_COLORS = { matched: "#2e9d4f", gap: "#d32f2f",
                        arithmetic_mismatch: "#e98a00", unchecked: "#999999" };
const PROVER_COLORS = { proved: "#2e9d4f", not_proved: "#d32f2f", contradictory_premises: "#d32f2f",
                        timeout: "#e98a00", unknown: "#e98a00", error: "#b00020", skipped: "#999999" };

function esc(s) {
  return String(s ?? "").replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
}

const cy = cytoscape({
  container: document.getElementById("cy"),
  elements: DATA.elements,
  wheelSensitivity: 0.2,
  style: [
    { selector: "node[type='statement']", style: {
        "shape": "round-rectangle", "label": "data(display)", "text-wrap": "wrap",
        "text-max-width": 230, "font-size": 11, "text-valign": "center", "text-halign": "center",
        "width": "label", "height": "label", "padding": 10, "border-width": 1.5,
        "border-color": "#666",
        "background-color": ele => KIND_COLORS[ele.data("label_kind") === "derived"
                                               ? "derived" : ele.data("kind")] || "#ddd" } },
    { selector: "node[shape='rule']", style: { "shape": "cut-rectangle" } },
    { selector: "node[shape='closure']", style: { "shape": "hexagon", "padding": 16 } },
    { selector: "node[?unstated]", style: { "border-style": "dashed", "border-width": 2.5,
                                             "border-color": "#333" } },
    { selector: "node[type='step']", style: {
        "shape": "diamond", "label": "data(display)", "text-wrap": "wrap", "font-size": 10,
        "font-weight": "bold", "text-valign": "center", "width": 62, "height": 62,
        "background-color": "#ffffff", "border-width": 3,
        "border-color": ele => STATUS_COLORS[ele.data("status")] || "#999" } },
    { selector: "edge", style: {
        "width": 1.6, "curve-style": "bezier", "target-arrow-shape": "triangle",
        "line-color": "#888", "target-arrow-color": "#888", "label": "data(role)",
        "font-size": 9, "color": "#666", "text-background-color": "#fafafa",
        "text-background-opacity": 1, "text-background-padding": 1 } },
    { selector: "edge[role='warrant']", style: { "line-style": "dashed",
        "line-color": "#5b7fbf", "target-arrow-color": "#5b7fbf" } },
    { selector: "edge[role='conclusion']", style: { "width": 2.4, "line-color": "#444",
        "target-arrow-color": "#444" } },
    { selector: ":selected", style: { "overlay-color": "#3b82f6", "overlay-opacity": 0.18 } },
  ],
  layout: { name: "dagre", rankDir: "LR", nodeSep: 18, rankSep: 70 },
});

function section(title, html) { return `<h3>${title}</h3>${html}`; }
function pre(obj) {
  return `<pre>${esc(typeof obj === "string" ? obj : JSON.stringify(obj, null, 2))}</pre>`;
}

function legend() {
  const kinds = Object.entries(KIND_COLORS).filter(([k]) => k)
    .map(([k, c]) => `<span><i class="sw" style="background:${c}"></i>${k}</span>`).join("");
  const statuses = Object.entries(STATUS_COLORS)
    .map(([k, c]) => `<span><i class="sw" style="border:3px solid ${c};background:#fff"></i>step ${k}</span>`).join("");
  return `<div class="legend">${kinds}<br>${statuses}<br>
    <span>rounded = fact, cut corners = rule, hexagon = closure, dashed border = unstated in CoT</span></div>`;
}

function overview() {
  const steps = Object.values(DATA.details).filter(d => d.move !== undefined);
  const counts = {};
  steps.forEach(s => counts[s.status] = (counts[s.status] || 0) + 1);
  const statusLine = Object.entries(counts).map(([k, v]) => `<span class="badge">${k}: ${v}</span>`).join("");
  const errors = DATA.errors.length
    ? DATA.errors.map(e => `<div class="err">• ${esc(e)}</div>`).join("")
    : "<div>none</div>";
  const problems = steps.filter(s => s.status !== "matched")
    .map(s => `<div><b>${esc(s.id)}</b> ${esc(s.status)}: ${esc(s.status_detail)}</div>`).join("") || "<div>none</div>";
  const proverProblems = steps.filter(s => s.prover && s.prover !== "proved")
    .map(s => `<div><b>${esc(s.id)}</b> ${esc(s.prover)}: ${esc(s.prover_detail)}</div>`).join("") || "<div>none</div>";
  const verdictColor = {VERIFIED: "#2e9d4f", "NOT VERIFIED": "#d32f2f"}[DATA.prover_verdict] || "#999";
  document.getElementById("side").innerHTML =
    `<h2>Inference graph</h2>${statusLine}` +
    `<div><span class="badge" style="border:2px solid ${verdictColor}">prover: ${esc(DATA.prover_verdict)}</span></div>` +
    section("Legend", legend()) +
    section("Validation errors", errors) +
    section("Steps that did not match", problems) +
    section("Steps not proved by Vampire", proverProblems) +
    section("Question", `<div class="text">${esc(DATA.question)}</div>`) +
    section("Chain of thought", pre(DATA.cot)) +
    section("Dictionary", pre(DATA.dictionary)) +
    section("Full TPTP", pre(DATA.tptp)) +
    section(`Edit log (${DATA.oplog.length} ops)`,
            pre(DATA.oplog.map(o => `#${o.seq} ${o.agent}: ${o.op} ${o.collection}/${o.id}`).join("\n")));
}

function showNode(id) {
  const d = DATA.details[id];
  const back = `<button onclick="cy.elements().unselect(); overview()">&larr; overview</button>`;
  let html = back;
  if (d.move !== undefined) {
    html += `<h2>${esc(id)} · step (${esc(d.move)})</h2>` +
      `<span class="badge" style="border-color:${STATUS_COLORS[d.status]}">${esc(d.status)}</span>` +
      (d.status_detail ? `<div class="err">${esc(d.status_detail)}</div>` : "") +
      (d.prover ? section("Prover (Vampire)",
        `<span class="badge" style="border-color:${PROVER_COLORS[d.prover] || "#999"}">${esc(d.prover)}</span>` +
        `<div class="text">${esc(d.prover_detail)}</div>`) : "") +
      section("Authored text", `<div class="text">${esc(d.text)}</div>`) +
      section("Generated English", `<div class="text">${esc(d.english)}</div>`) +
      section("CoT quote", `<div class="text">${esc(d.cot_quote)}</div>`) +
      section("Stated computation", `<div class="text">${esc(d.stated_computation)}</div>`) +
      section("Premises / warrant / conclusion",
              pre({premises: d.premises, warrant: d.warrant, conclusion: d.conclusion})) +
      section("Binding (computed)", pre(d.binding || {})) +
      section("TPTP problem for this step", pre(d.tptp));
  } else {
    const badges = [d.shape, d.label, d.kind, d.label === "assumption"
                     ? (d.stated_in_cot ? "stated in CoT" : "unstated in CoT") : null]
      .filter(Boolean).map(b => `<span class="badge">${esc(b)}</span>`).join("");
    html += `<h2>${esc(id)} · statement</h2>${badges}` +
      section("Authored text", `<div class="text">${esc(d.text)}</div>`) +
      section("Generated English", `<div class="text">${esc(d.english)}</div>`) +
      (d.question_quote ? section("Question quote", `<div class="text">${esc(d.question_quote)}</div>`) : "") +
      (d.cot_quote ? section("CoT quote", `<div class="text">${esc(d.cot_quote)}</div>`) : "") +
      section("Content", pre(d.content)) +
      section("TPTP", pre(d.tptp));
  }
  document.getElementById("side").innerHTML = html;
}

cy.on("tap", "node", evt => showNode(evt.target.id()));
cy.on("tap", evt => { if (evt.target === cy) overview(); });
overview();
</script>
</body>
</html>
"""


def write_html(graph: Graph, path: str):
    payload = json.dumps(graph_payload(graph), default=str).replace("</", "<\\/")
    with open(path, "w") as f:
        f.write(HTML_TEMPLATE.replace("__PAYLOAD__", payload))
