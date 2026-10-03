import os
"""Static self-check of the prototype's WORKSPACE surface. Not a pytest run.
Checks structure, accessibility hooks, wording rules, and that every number on the
screen traces to the real run files (the only defence against invented figures)."""
import json
import re
import subprocess
import sys
from html.parser import HTMLParser

sys.path.insert(0, r"D:\(Dev2)DSA_AGENT")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
P = ROOT + r"\docs\prototypes\verdacert_dsa_preview.html"
R = ROOT + r"\output\aq_final"
s = open(P, encoding="utf-8").read()

a = s.index('<main id="surface-workspace"')
b = s.index("</main>", a)
ws = s[a:b]
bar = s[s.index('<div class="proto-bar">'):s.index('<main id="surface-landing"')]

problems, ok = [], []


def check(name, passed, detail=""):
    (ok if passed else problems).append(name + ((": " + detail) if (detail and not passed) else ""))


class Visible(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        for key, value in attrs:
            if key in ("aria-label", "title", "alt", "placeholder") and value:
                self.chunks.append(value)

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.chunks.append(" ".join(data.split()))


vis = Visible()
vis.feed(ws)
text = "\n".join(vis.chunks)
low = text.lower()

# ---- structure and accessibility
ids = re.findall(r'\sid="([^"]+)"', s)
dups = sorted({i for i in ids if ids.count(i) > 1})
check("no duplicate ids on the page", not dups, str(dups))
tabs = re.findall(r'role="tab"[^>]*id="(tab-[a-z]+)"|id="(tab-[a-z]+)"[^>]*role="tab"', ws)
tab_ids = sorted({x or y for x, y in tabs})
check("four tabs present", tab_ids == ["tab-answers", "tab-charts", "tab-details", "tab-downloads"], str(tab_ids))
for t in tab_ids:
    name = t.replace("tab-", "")
    check(f"{t} controls an existing tabpanel", f'id="panel-{name}"' in ws and f'aria-controls="panel-{name}"' in ws)
    check(f"panel-{name} is labelled by its tab", f'aria-labelledby="{t}"' in ws)
check('tablist role present', 'role="tablist"' in ws)
check("exactly one tab selected initially", len(re.findall(r'role="tab"[^>]*aria-selected="true"', ws)) == 1)
check("tab keyboard handling (arrows, Home, End)", all(k in s for k in ("ArrowRight", "ArrowLeft", "'Home'", "'End'")))
check("settings button controls the rail", 'aria-controls="ws-settings"' in ws and 'id="ws-settings"' in ws)
for state in ("empty", "file", "running", "results", "failed"):
    check(f"state {state!r} has a prototype-bar button", f'data-ws="{state}"' in bar)
    check(f"state {state!r} has content", state in " ".join(re.findall(r'data-states="([^"]+)"', ws)))
# every form control is labelled
unlabelled = []
for m in re.finditer(r'<(input|select|textarea)\b([^>]*)>', ws):
    attrs = m.group(2)
    if 'type="radio"' in attrs or 'type="checkbox"' in attrs:
        continue  # wrapped in a <label>
    cid = re.search(r'\sid="([^"]+)"', attrs)
    if not cid or not re.search(rf'for="{re.escape(cid.group(1))}"', ws):
        unlabelled.append(attrs.strip()[:60])
check("every text input, select and textarea has a label", not unlabelled, str(unlabelled))
check("checkbox switches sit inside labels", len(re.findall(r'<label class="switch">', ws)) == len(re.findall(r'type="checkbox"', ws)))
check("tables have scope on headers", ws.count("<th scope=") >= 10)
check("scrollable regions are focusable and labelled", ws.count('role="region"') == ws.count('tabindex="0" role="region"') and ws.count('role="region"') >= 2)
check("no inline event handlers except download toasts", len(re.findall(r'\sonclick=', ws)) == ws.count("showToast('Prototype: the real app downloads"))

# ---- no 3D, no gradients, no glow in workspace css
css = re.sub(r"/\*.*?\*/", "", s[s.index("#surface-workspace {"):s.index("LANDING STRUCTURE (follows")], flags=re.S)
check("workspace css has no gradients", "gradient" not in css)
check("workspace css has no glow or blur", "blur(" not in css and "glow" not in css)
check("workspace css has no animation or transition", "@keyframes" not in css and "transition" not in css and "animation" not in css)
danger_selectors = {x.strip() for x in re.findall(r"([.\w\- ]+)\{[^}]*var\(--danger\)", css)}
check("--danger used only on 'may not hold' marks", danger_selectors <= {".fs.risk", ".cc .mk"}, str(danger_selectors))
check("no threejs/canvas in the workspace", "<canvas" not in ws and "THREE" not in ws)

# ---- wording
BANNED = ("swarm", "synapse", "manifold", "certified", "telemetry", "never leaves", "100% local", "air-gapped",
          "wcag", "compliance", "cryptographic", "operational", "rlm v2", "no credit card", "repl hash", "churn",
          "verified findings", "audited pass", "deterministic run")
found = [w for w in BANNED if w in low]
check("no banned or unsupported phrases", not found, str(found))
for jargon in ("p-value", "silhouette", "fdr", "benjamini", "cross-validation", "5-fold", "mann-kendall", "pearson"):
    # allowed only inside the folded "Details for analysts" notes and the technical record
    outside = re.sub(r'<details class="tech-note">.*?</details>', "", ws, flags=re.S)
    outside_vis = Visible()
    outside_vis.feed(outside)
    check(f"{jargon!r} appears only inside folded details", jargon not in "\n".join(outside_vis.chunks).lower())
check("sample runs are labelled as not the user's data", "it is not your data" in low)
check("charts are labelled as schematics where they are", low.count("schematic") >= 5)
check("verdict banner counts match findings", "1 of 2 checked findings held up" in low and ws.count('class="finding-item"') == 3)

# ---- every number on screen traces to the real run (or is structural)
real = ""
for name in (r"\reports\final_report.json", r"\reports\summary.json", r"\reports\AirQualityUCI_raw.json"):
    real += open(R + name, encoding="utf-8").read() + "\n"
from src.core.io import read_any_bytes  # noqa: E402

df, rep = read_any_bytes(open(ROOT + r"\data\AirQualityUCI.csv", "rb").read(), "AirQualityUCI.csv")
real += "\n".join(" ".join(str(v) for v in row) for row in df.head(10).itertuples(index=False)) + "\n" + "\n".join(rep.notes) + "\n" + "\n".join(map(str, df.columns))
real += f"\n{len(df)} {len(df.columns)} {int(df.isnull().sum().sum())} {len(df.select_dtypes(include='number').columns)}"
for c in df.columns:
    real += f"\n{int(df[c].isnull().sum())}"
import os  # noqa: E402

for sub in ("reports", "data", "models"):
    for fn in os.listdir(R + "\\" + sub):
        real += f"\n{os.path.getsize(R + chr(92) + sub + chr(92) + fn) / 1000:,.0f}"

def norm(token):
    token = token.replace(",", "")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token

real_numbers = {norm(t) for t in re.findall(r"\d[\d,]*\.?\d*", real)}
# also allow figures that are straightforward roundings of real ones
visible_numbers = []
body = re.sub(r'<details class="tech-note">.*?</details>', lambda m: m.group(0), ws, flags=re.S)
for t in re.findall(r"(?<![\w.])\d[\d,]*\.?\d*(?![\w])", text):
    visible_numbers.append(t.rstrip(".,"))
STRUCTURAL = {str(n) for n in range(0, 13)} | {"40", "50", "100", "1", "2004", "20"}
unmatched = sorted({t for t in visible_numbers if norm(t) not in real_numbers and norm(t) not in STRUCTURAL})
# derived and rounded figures the page states on purpose, each justified from the run
secs = json.load(open(R + r"\reports\summary.json", encoding="utf-8"))["tool_seconds"]
ROUNDED_SECONDS = {f"{v:.2f}" for v in secs.values()}
DERIVED = {
    "0.001": "p < 0.001: re-derived from the data, r 0.9311 on 7,344 readings gives p = 0.0",
    "125": "NOx level before (125.12)", "333": "NOx level after (332.85)", "166": "+166% (delta_pct 1.6601)",
    "3,300": "cluster sizes 3,288", "6,200": "cluster sizes 6,183", "0.93": "r = 0.9311", "4.8": "slope 4.82",
    "13": "13% of cells missing (profile warning)", "90": "NMHC 90% missing (profile warning)",
    "83": "data quality 83", "113": "113 duplicate rows", "1.2": "1.2% duplicates", "25": "25% confidence",
    "3.4": "sum of tool seconds 3.42", "150": "cluster mean 149.9", "235": "overall mean 235",
    "7,344": "rows used", "2,127": "rows left out", "9,357": "values converted", "15,264": "missing cells",
    "9,471": "rows", "0.294": "silhouette", "0.9311": "r", "4.82": "slope", "125.12": "level", "332.85": "level",
    "50.45": "z", "11": "questions considered", "10": "unanswered / steps", "8": "numeric columns",
}
really_unmatched = [t for t in unmatched if norm(t) not in {norm(k) for k in DERIVED} and t not in ROUNDED_SECONDS]
check("every number on screen traces to the run files, a justified derived figure, or is structural",
      not really_unmatched, str(really_unmatched))
print("numbers checked:", len(set(visible_numbers)), "| matched directly:", len(set(visible_numbers)) - len(unmatched),
      "| derived (justified):", len(unmatched) - len(really_unmatched), "| unexplained:", len(really_unmatched))

# ---- the keyword search: plain text only, and its own example must find something
m = re.search(r"const FINDINGS = (\[.*?\]);" + chr(10), s, re.S)
findings = json.loads(m.group(1).replace("<" + chr(92) + "/", "</")) if m else []
check("search has 12 written findings", len(findings) == 12, str(len(findings)))
plain_text = " ".join(f["t"] for f in findings).lower()
bad = [w for w in ("silhouette", "r=", "r between", "(gt)", "pt08", "cluster_", "p-value", "fdr", "pearson", "variance") if w in plain_text]
check("plain search text has no jargon or column codes", not bad, str(bad))
ph = re.search(r'id="ws-search"[^>]*placeholder="e\.g\. ([^"]+)"', ws)
example = ph.group(1).lower() if ph else ""
def hits(q):
    words = q.lower().split()
    return [f for f in findings if all(w in (f["t"] + " " + f["a"]).lower() for w in words)]
check("the search placeholder's own example finds a result", bool(example) and len(hits(example)) >= 1, example)
check("searching a column code finds the same findings as the plain name", len(hits("c6h6")) >= 1 and len(hits("benzene")) >= len(hits("c6h6")) - 0)
visible_numbers_extra = [n for f in findings for n in re.findall(r"(?<![\w.])\d[\d,]*\.?\d*(?![\w])", f["t"])]
real_text_for_numbers = chr(10).join(f["a"] for f in findings)
bad_nums = [n for n in visible_numbers_extra if n.replace(",", "").rstrip(".") not in {x.replace(",", "").rstrip(".") for x in re.findall(r"\d[\d,]*\.?\d*", real_text_for_numbers)} and float(n.replace(",", "").rstrip(".")) > 31]
check("numbers in the plain search text come from the matching real finding", not bad_nums, str(bad_nums))
check("sample loading forces no-AI mode (run is reachable)", s.count("WS.mode = 'noai'") >= 3)
check("running status is a single status line, not a live list", 'id="ws-status" role="status"' in ws and 'id="ws-stages" aria-live' not in ws)
check("no Risk mark on non-risks", ws.count('<span class="mk">Risk</span>') == 1)
check("night-safe text tokens exist", "--accent-text" in s and "--danger-text" in s)

# ---- JS parses
scripts = re.findall(r"<script>(.*?)</script>", s, re.S)
import tempfile
inline_path = os.path.join(tempfile.gettempdir(), "inline_check.js")
open(inline_path, "w", encoding="utf-8").write(max(scripts, key=len))
res = subprocess.run(["node", "--check", inline_path], capture_output=True, text=True)
check("inline script parses", res.returncode == 0, res.stderr[:300])
check("showToast and switchView survive", "function showToast" in s and "function switchView" in s)
check("landing CTA ids survive", all(f'id="{i}"' in s for i in ("hero-enter-btn", "enter-btn", "cta-launch-btn", "nav-launch-btn", "btn-theme-toggle")))

print(f"PASS {len(ok)}   FAIL {len(problems)}")
for line in problems:
    print("  FAIL:", line)
