// Fails if any island source uses dangerouslySetInnerHTML (React text nodes only).
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const src = new URL("../src/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const walk = (d) => readdirSync(d).flatMap((f) => {
  const p = join(d, f);
  return statSync(p).isDirectory() ? walk(p) : [p];
});
const bad = walk(src).filter((f) => /\.(tsx?|jsx?)$/.test(f) && /dangerouslySetInnerHTML|\.innerHTML\s*=/.test(readFileSync(f, "utf8")));
if (bad.length) { console.error("HTML injection found in:", bad.join(", ")); process.exit(1); }
console.log("ok: no HTML injection in island sources");
