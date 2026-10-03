(function initScrollEffects() {
  'use strict';

  /* Three scroll effects, all transform / opacity / clip-path only:
       1. Line-by-line text reveal on headlines and ledes (.rv)
       2. The dark developer band opens from a rounded panel to full width,
          the same gesture the pipeline card makes
       3. Gentle parallax on the proof artifacts ([data-parallax])
     Reduced motion: none of this runs and everything is shown in place.
     No JS: the .js-fx class is never added, so all text stays visible. */

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduceMotion) return;

  const root = document.documentElement;
  const landing = document.getElementById('surface-landing');
  const clamp01 = (v) => Math.max(0, Math.min(1, v));
  const smoothstep = (a, b, v) => {
    const t = clamp01((v - a) / (b - a));
    return t * t * (3 - 2 * t);
  };

  // ---- 1. line-by-line reveal -------------------------------------------
  const REVEAL_SELECTOR = '.h-display, .h-section, .solare-intro-header .lede, .sec-head .lede, .cta-banner .lede, .agents-band h2';
  const HERO_KEY = 'dsa.hero.played';
  // The hero headline plays its reveal once per session; later visits show it plain,
  // as the original page does for its hero island.
  let heroPlayed = false;
  try { heroPlayed = window.sessionStorage.getItem(HERO_KEY) === '1'; } catch (error) { /* storage blocked: play it */ }
  const revealTargets = Array.from(document.querySelectorAll(REVEAL_SELECTOR))
    .filter((element) => !(heroPlayed && element.closest('.hero')));

  // Wrap each word as <span class="w"><i>word</i></span>, keeping inline
  // elements (such as the serif accent) intact.
  function splitWords(node) {
    Array.from(node.childNodes).forEach((child) => {
      if (child.nodeType === Node.TEXT_NODE) {
        const fragment = document.createDocumentFragment();
        child.textContent.split(/(\s+)/).forEach((part) => {
          if (part === '') return;
          if (/^\s+$/.test(part)) {
            fragment.appendChild(document.createTextNode(part));
          } else {
            const word = document.createElement('span');
            const inner = document.createElement('i');
            word.className = 'w';
            inner.textContent = part;
            word.appendChild(inner);
            fragment.appendChild(word);
          }
        });
        node.replaceChild(fragment, child);
      } else if (child.nodeType === Node.ELEMENT_NODE) {
        splitWords(child);
      }
    });
  }

  // Which visual line each word landed on, so lines stagger instead of words.
  function assignLines(element) {
    let line = -1;
    let lastTop = null;
    let inLine = 0;
    element.querySelectorAll('.w').forEach((word) => {
      const top = word.offsetTop;
      if (lastTop === null || Math.abs(top - lastTop) > 2) {
        line += 1;
        lastTop = top;
        inLine = 0;
      }
      word.style.setProperty('--line', String(line));
      word.style.setProperty('--k', String(inLine));
      inLine += 1;
    });
  }

  function relayout() {
    revealTargets.forEach(assignLines);
  }

  function reveal(element) {
    element.classList.add('in');
    if (element.closest('.hero')) {
      try { window.sessionStorage.setItem(HERO_KEY, '1'); } catch (error) { /* storage blocked */ }
    }
  }

  // Text must never stay hidden: any failure here removes the effect and shows everything.
  function releaseAll() {
    root.classList.remove('js-fx');
    revealTargets.forEach((element) => element.classList.add('in'));
  }

  try {
    root.classList.add('js-fx');
    revealTargets.forEach((element) => {
      splitWords(element);
      element.classList.add('rv');
    });
    relayout();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(relayout);

    if (typeof IntersectionObserver === 'function') {
      const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          reveal(entry.target);
          revealObserver.unobserve(entry.target);
        });
      }, { threshold: 0.2, rootMargin: '0px 0px -8% 0px' });
      revealTargets.forEach((element) => revealObserver.observe(element));
      // Failsafe: anything already on screen after 2.5 s is shown, observer or not.
      window.setTimeout(() => {
        revealTargets.forEach((element) => {
          if (!element.classList.contains('in') && element.getBoundingClientRect().top < window.innerHeight) reveal(element);
        });
      }, 2500);
    } else {
      releaseAll();
    }
  } catch (error) {
    releaseAll();
  }

  let resizeTimer = 0;
  window.addEventListener('resize', () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(relayout, 150);
  }, { passive: true });
  // The landing surface is display:none while the workspace is open; measure
  // again when it comes back.
  if (landing && typeof ResizeObserver === 'function') new ResizeObserver(relayout).observe(landing);

  // ---- 2 and 3: one throttled scroll pass --------------------------------
  const band = document.querySelector('.agents-band');
  const parallaxItems = Array.from(document.querySelectorAll('[data-parallax]')).map((element) => ({
    element: element,
    factor: parseFloat(element.getAttribute('data-parallax')) || 0,
    frame: element.parentElement,
    visible: true,
    last: ''
  }));
  const wide = window.matchMedia('(min-width: 961px)');
  let lastClip = '';
  let queued = false;

  function update() {
    queued = false;
    if (landing && landing.style.display === 'none') return;
    const viewport = window.innerHeight;

    if (band) {
      const top = band.getBoundingClientRect().top;
      const open = smoothstep(viewport * 0.95, viewport * 0.3, top);
      const inset = Math.round((1 - open) * Math.min(window.innerWidth * 0.05, 64));
      const clip = open > 0.999 ? 'none' : `inset(0 ${inset}px round ${((1 - open) * 18).toFixed(1)}px)`;
      if (clip !== lastClip) {
        band.style.clipPath = clip;
        lastClip = clip;
      }
    }

    parallaxItems.forEach((item) => {
      if (!item.visible) return;
      let transform = '';
      if (wide.matches) {
        const rect = item.frame.getBoundingClientRect();
        const offset = rect.top + rect.height / 2 - viewport / 2;
        transform = `translate3d(0, ${(offset * item.factor).toFixed(1)}px, 0)`;
      }
      if (transform !== item.last) {
        item.element.style.transform = transform;
        item.last = transform;
      }
    });
  }

  function queueUpdate() {
    if (queued) return;
    queued = true;
    window.requestAnimationFrame(update);
  }

  if (typeof IntersectionObserver === 'function') {
    const parallaxObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const item = parallaxItems.find((candidate) => candidate.frame === entry.target);
        if (item) item.visible = entry.isIntersecting;
      });
      queueUpdate();
    }, { rootMargin: '120px 0px' });
    parallaxItems.forEach((item) => parallaxObserver.observe(item.frame));
  }

  window.addEventListener('scroll', queueUpdate, { passive: true });
  window.addEventListener('resize', queueUpdate, { passive: true });
  queueUpdate();
})();
