// Prints gzipped sizes of every built island file and compares them to budget.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { gzipSync } from "node:zlib";

const root = new URL("../../../static/islands/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const walk = (d) => readdirSync(d).flatMap((f) => {
  const p = join(d, f);
  return statSync(p).isDirectory() ? walk(p) : [p];
});
let total = 0;
for (const f of walk(root)) {
  if (f.endsWith("manifest.json")) continue;
  const gz = gzipSync(readFileSync(f)).length;
  total += gz;
  console.log(`${f.slice(root.length).padEnd(40)} ${(statSync(f).size / 1024).toFixed(1).padStart(8)} KB raw ${(gz / 1024).toFixed(1).padStart(7)} KB gzip`);
}
const BUDGET_KB = 110;
console.log(`TOTAL gzip ${(total / 1024).toFixed(1)} KB (budget ${BUDGET_KB} KB)`);
if (total / 1024 > BUDGET_KB) { console.error("OVER BUDGET"); process.exit(1); }
