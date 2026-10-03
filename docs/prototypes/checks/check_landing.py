import os
"""Static self-check of the prototype's LANDING surface against the original page's rules.
Reuses the parser and banned-phrase list from tests/test_landing.py. Not a pytest run."""
import json
import re
import sys
from html.parser import HTMLParser

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT)
P = os.path.join(ROOT, "docs", "prototypes", "verdacert_dsa_preview.html")
s = open(P, encoding="utf-8").read()

# Landing surface = skip link .. start of workspace, plus the footer.
landing = s[s.index('<a class="skip-link"'):s.index('<main id="surface-workspace"')]
landing += s[s.index('<footer class="site-footer"'):]


class VisibleText(HTMLParser):
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


parser = VisibleText()
parser.feed(landing)
text = "\n".join(parser.chunks)
low = text.lower()

BANNED = ("swarm", "synapse", "manifold", "undulation", "certified", "telemetry",
          "never leaves", "runs entirely on your machine", "100% local", "zero cloud",
          "cloud transmissions", "air-gapped", "wcag", "compliance", "cryptographic",
          "formal mathematical verification", "guard status", "operational",
          "rlm v2", "high risk", "no credit card")
EXTRA_BANNED = ("repl", "faq", "pricing", "per month", "annual", "bit-for-bit", "rest + mcp", "100% reproducible")

problems = []
ok = []


def check(name, passed, detail=""):
    (ok if passed else problems).append(f"{name}{(': ' + detail) if detail and not passed else ''}")


found = [b for b in BANNED + EXTRA_BANNED if b in low]
check("no banned or unsupported phrases in visible landing text", not found, str(found))

# data-path statement (test_data_path_statement_is_honest_and_complete)
statement = text[text.index("Where your data goes:"):].split("\n", 1)[0]
for need in ("processed on the server", "deleted when you start a new analysis", "AI provider you choose",
             "switch AI off", "sensitive data"):
    check(f"data-path statement contains {need!r}", need in statement)
check("data-path statement mentions academic project", "academic project" in statement.lower())
check("data-path statement makes no retention or compliance promise",
      not any(p in statement.lower() for p in ("never stored", "gdpr", "hipaa", "soc 2", "retention")))

# ids the original page's tests and navigation rely on
ids = re.findall(r'\sid="([^"]+)"', landing)
for needed in ("hero", "hero-enter-btn", "hero-explore-btn", "enter-btn", "cta-launch-btn", "nav-launch-btn",
               "btn-theme-toggle", "workflow", "audited-entry", "comparison", "features", "cta", "data-path"):
    check(f"id {needed!r} present", needed in ids)
dups = sorted({i for i in ids if ids.count(i) > 1})
check("no duplicate ids on the landing surface", not dups, str(dups))
for stage in range(4):
    check(f"data-stage={stage} present", f'data-stage="{stage}"' in landing)

# every in-page link resolves
all_ids = set(re.findall(r'\sid="([^"]+)"', s))
broken = [h for h in re.findall(r'href="#([^"]+)"', landing) if h not in all_ids]
check("every #anchor resolves to an id", not broken, str(sorted(set(broken))))

# SEO block
head = s[:s.index("</head>")]
for token in ('name="description"', 'property="og:title"', 'property="og:description"', 'name="twitter:card"', 'rel="canonical"'):
    check(f"head has {token}", token in head)
m = re.search(r'<script type="application/ld\+json">(.*?)</script>', head, re.S)
try:
    data = json.loads(m.group(1))
    types = {n.get("@type") for n in data.get("@graph", [])}
    check("JSON-LD parses and declares SoftwareApplication", "SoftwareApplication" in types)
    check("JSON-LD has no FAQPage (the FAQ is gone)", "FAQPage" not in types)
except Exception as exc:  # noqa: BLE001
    check("JSON-LD parses", False, repr(exc))
check('<html lang="en">', '<html lang="en"' in s)

# removed components must leave no markup or CSS behind
for dead in (".faq-", "faq-item", ".why-", "why-grid", ".steps", 'class="step"', "toggleFaq", "guarantee-sec", "hero-trust"):
    check(f"no remnant of {dead!r}", dead not in s)

# the 3D track is labelled as an illustration; findings are labelled as a real run
check("track labelled as an illustration", "illustration, not your data" in text)
check("findings labelled as a real run", "real run" in low)

# verdict count equals the number of audited entries shown
cards = landing.count('class="audited-entry"')
check("audited entries shown == 2 (verdict says 'the 2 findings shown')", cards == 2 and "1 of the 2 findings shown" in text)

# headings
check("exactly one h1", len(re.findall(r"<h1[\s>]", landing)) == 1)

# one name for the pipeline: no '3-step', '3 steps', 'seven-stage' only in the product sentence
check("no 3-step ribbon wording left", "3-step" not in low and "three steps" not in low)

# the original's primary CTA label is used consistently
markup = re.sub(r"<script.*?</script>", "", landing, flags=re.S)
labels = re.findall(r"<(?:a|button)\b[^>]*\bdata-enter\b[^>]*>\s*(?:<svg.*?</svg>)?\s*([^<]+?)\s*(?:<svg|</)", markup, re.S)
check("every data-enter control reads 'Open the workspace'", labels and all(l.strip().startswith("Open the workspace") for l in labels), str(labels))

print(f"PASS {len(ok)}   FAIL {len(problems)}")
for line in problems:
    print("  FAIL:", line)

# Workspace mock is out of scope; report what it still carries so it is not forgotten
ws = s[s.index('<main id="surface-workspace"'):s.index('<div id="toast"')]
wsp = VisibleText()
wsp.feed(ws)
wlow = "\n".join(wsp.chunks).lower()
print("workspace mock (not changed):", [b for b in BANNED + EXTRA_BANNED if b in wlow])
