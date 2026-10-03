/**
 * Derived from Magic UI "text-animate" (https://magicui.design/docs/components/text-animate)
 * Copyright (c) Magic UI — MIT License (full notice in THIRD_PARTY_UI.md).
 * Snapshot taken 2026-10-03 from the public registry (https://magicui.design/r/text-animate.json;
 * the registry carries no version number, so the snapshot date is the version).
 *
 * Local modifications: trimmed to the by-word "blurIn" variant only; Tailwind and `cn` removed;
 * LazyMotion + `m` instead of the full `motion` component; staggering is driven by per-word delay;
 * the screen-reader copy is a single visually-hidden span; the caller decides whether to animate
 * (reduced motion / already-played-this-session never reach this component); `onDone` lets the
 * caller restore the original DOM when the animation completes.
 */
import { LazyMotion, domAnimation, m } from "motion/react";

export interface Segment {
  text: string;
  className?: string;
}

interface Props {
  segments: Segment[];
  onDone: () => void;
}

const STAGGER_S = 0.07;
const DURATION_S = 0.45;

export function TextAnimate({ segments, onDone }: Props) {
  const full = segments.map((s) => s.text).join("");
  // Words keep the segment's class so existing styling (the highlighted word) is preserved.
  const words: { word: string; className?: string }[] = [];
  for (const seg of segments) {
    for (const part of seg.text.split(/(\s+)/)) {
      if (part !== "") words.push({ word: part, className: seg.className });
    }
  }
  const last = words.length - 1;
  let n = 0;
  return (
    <LazyMotion features={domAnimation} strict>
      <span className="dsa-ta-sr">{full}</span>
      <span aria-hidden="true">
        {words.map((w, i) => {
          if (/^\s+$/.test(w.word)) return w.word; // real spaces keep natural wrapping
          const delay = n++ * STAGGER_S;
          return (
            <m.span
              key={i}
              className={`dsa-ta-word${w.className ? " " + w.className : ""}`}
              initial={{ opacity: 0, filter: "blur(10px)" }}
              animate={{ opacity: 1, filter: "blur(0px)" }}
              transition={{ delay, duration: DURATION_S, ease: "easeOut" }}
              onAnimationComplete={i === last ? onDone : undefined}
            >
              {w.word}
            </m.span>
          );
        })}
      </span>
    </LazyMotion>
  );
}
