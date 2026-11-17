const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

/** Minimal offline DOM contract; this does not launch a browser. */
class Element {
  /** Store a fake node used by the dashboard's pure render functions. */
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.style = {};
    this.attributes = {};
    this.handlers = {};
    this.value = "";
    this.textContent = "";
  }

  /** Append a child and emulate a select's first-option default. */
  appendChild(child) {
    this.children.push(child);
    if (this.tag === "select" && this.value === "") this.value = child.value;
    return child;
  }

  /** Append text and element nodes without parsing HTML. */
  append(...children) { children.forEach(child => this.appendChild(child)); }

  /** Clear the rendered children before drawing another window. */
  replaceChildren() { this.children = []; }

  /** Record SVG attributes for assertions about real-time layout. */
  setAttribute(name, value) { this.attributes[name] = value; }

  /** Record callbacks so tests can exercise actual control listeners. */
  addEventListener(name, handler) { this.handlers[name] = handler; }
}

const html = fs.readFileSync(process.argv[2], "utf8");
const payload = html.match(/<script id="audit-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, new Element("div"));
    return elements.get(id);
  },
  createElement(tag) { return new Element(tag); },
  createElementNS(namespace, tag) { return new Element(tag); },
  createTextNode(text) { const node = new Element("text"); node.textContent = text; return node; }
};
for (const name of ["participant", "recipe", "window-size", "stage-filter", "error-filter"]) {
  elements.set(name, new Element("select"));
}
document.getElementById("audit-data").textContent = payload;
for (const [name, value] of Object.entries({recipe:"all", "window-size":"120",
                                          "window-start":"0", "stage-filter":"all",
                                          "error-filter":"all"})) {
  document.getElementById(name).value = value;
}
const context = vm.createContext({document, console});
vm.runInContext(script, context, {timeout: 5000});
const person = JSON.parse(payload).rows[0].participant_id;
const personRows = JSON.parse(payload).rows.filter(row => row.participant_id === person);
const rows = personRows.slice(0, 120);
assert.equal(document.getElementById("target-table").children.length, rows.length);
const initialStageNodes = document.getElementById("stage-chart").children[0].children.length;

document.getElementById("stage-filter").value = "2";
document.getElementById("stage-filter").handlers.change();
assert.equal(document.getElementById("target-table").children.length,
             rows.filter(row => row.target === 2).length);
assert.equal(document.getElementById("stage-chart").children[0].children.length,
             initialStageNodes, "Table filters must not remove timeline observations");

document.getElementById("error-filter").value = "persistent";
document.getElementById("error-filter").handlers.change();
assert.equal(document.getElementById("target-table").children.length,
             rows.filter(row => row.target === 2 && row.persistent_to_light).length);

document.getElementById("recipe").value = "prediction_reference";
document.getElementById("recipe").handlers.change();
assert.ok(document.getElementById("stage-chart").children[0].children.length < initialStageNodes);

vm.runInContext(`
  const probe = [{time:0,segment_id:0,value:1}, {time:30,segment_id:0,value:2},
                 {time:300,segment_id:1,value:3}, {time:330,segment_id:1,value:null}];
  const path = featurePath(probe, "value", value => value, value => value);
  if ((path.match(/M/g) || []).length !== 2 || (path.match(/L/g) || []).length !== 1) {
    throw new Error("Feature paths must break at gaps, segments, and missing values");
  }
  const scale = timeScale(probe);
  if (Math.abs((scale.x(300)-scale.x(30))/(scale.x(30)-scale.x(0))-9) > 1e-8) {
    throw new Error("Elapsed-time axes must retain gaps");
  }
  if (numberText(null) !== "—") throw new Error("Missing observations must not become zero");
`, context, {timeout: 5000});
console.log("Offline dashboard control, gap, segment, and missingness contracts passed");
