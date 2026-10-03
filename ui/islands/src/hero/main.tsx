/**
 * Hero island: (1) headline words fade in from blur once per session; (2) a static/slow-drifting
 * spotlight behind the hero. Display only. It reads no data and sends nothing to Python.
 * If anything here fails, the original headline stays exactly as the landing page rendered it.
 */
import { createRoot } from "react-dom/client";
import css from "./hero.css?inline";
import { TextAnimate, type Segment } from "./TextAnimate";

const PLAYED_KEY = "dsa.hero.played";

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function alreadyPlayed(): boolean {
  try {
    return sessionStorage.getItem(PLAYED_KEY) === "1";
  } catch {
    return false;
  }
}

function markPlayed(): void {
  try {
    sessionStorage.setItem(PLAYED_KEY, "1");
  } catch {
    /* storage blocked (iframe sandbox): animation may replay, harmless */
  }
}

function releaseGuard(): void {
  document.documentElement.classList.remove("hero-pending");
}

function injectCss(): void {
  if (document.getElementById("dsa-hero-island-css")) return;
  const el = document.createElement("style");
  el.id = "dsa-hero-island-css";
  el.textContent = css;
  document.head.appendChild(el);
}

function mountSpotlight(hero: HTMLElement): void {
  if (hero.querySelector(".dsa-hero-spot")) return;
  const spot = document.createElement("div");
  spot.className = "dsa-hero-spot";
  spot.setAttribute("aria-hidden", "true");
  hero.insertBefore(spot, hero.firstChild);

  let visible = true;
  const sync = () => {
    spot.dataset.paused = String(document.hidden || !visible);
  };
  document.addEventListener("visibilitychange", sync);
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((entries) => {
      visible = entries.some((e) => e.isIntersecting);
      sync();
    }).observe(hero);
  }
  sync();
}

function segmentsOf(h1: HTMLElement): Segment[] {
  const out: Segment[] = [];
  h1.childNodes.forEach((node) => {
    const text = (node.textContent ?? "").replace(/\s+/g, " ");
    if (text === "") return;
    out.push({ text, className: node instanceof HTMLElement ? node.className : undefined });
  });
  if (out.length) {
    out[0].text = out[0].text.trimStart();
    out[out.length - 1].text = out[out.length - 1].text.trimEnd();
  }
  return out;
}

function animateHeadline(h1: HTMLElement): void {
  const w = window as unknown as { __dsaHeroLate?: boolean };
  if (w.__dsaHeroLate || prefersReducedMotion() || alreadyPlayed()) return releaseGuard();
  const segments = segmentsOf(h1);
  if (segments.length === 0) return releaseGuard();

  const original = Array.from(h1.childNodes);
  const host = document.createElement("span");
  const root = createRoot(host);
  let finished = false;
  const restore = () => {
    if (finished) return;
    finished = true;
    root.unmount();
    h1.replaceChildren(...original);
  };
  h1.replaceChildren(host);
  markPlayed();
  root.render(<TextAnimate segments={segments} onDone={restore} />);
  releaseGuard();
  window.setTimeout(restore, 4000); // safety net: never leave the headline half-animated
}

function main(): void {
  const hero = document.getElementById("hero");
  const h1 = hero?.querySelector<HTMLElement>("h1.hero-title");
  if (!hero || !h1) return releaseGuard();
  injectCss();
  mountSpotlight(hero);
  animateHeadline(h1);
}

try {
  main();
} catch {
  releaseGuard();
}
