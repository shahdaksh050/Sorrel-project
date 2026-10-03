/*
 * Section pager: a small first-party replacement for the subset of fullPage.js the cinematic page used.
 * fullPage.js v4 is GPLv3 or commercial and was loaded from a CDN, which broke offline use and LOCAL_ONLY.
 * The names `fullpage` and `fullpage_api` are kept so the scene script and the nav buttons need no change.
 *
 * Supported: new fullpage(selector, options), fullpage_api.moveTo(index | anchor), moveSectionDown/Up,
 * options scrollingSpeed, easingcss3, navigation, navigationTooltips, anchors, touchSensitivity,
 * normalScrollElements, onLeave(origin, destination), afterLoad(origin, destination).
 * Indexes passed to moveTo are 1-based, as in fullPage.js; callback objects carry a 0-based `index`.
 * One transform on the container, wheel / touch / keyboard paging, and no motion under reduced-motion.
 */
(function () {
  'use strict';

  var REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function Pager(selector, opts) {
    var root = document.querySelector(selector);
    if (!root) return;
    var o = opts || {};
    var sections = Array.prototype.slice.call(root.querySelectorAll(':scope > .section'));
    if (!sections.length) return;

    var speed = REDUCED ? 0 : (o.scrollingSpeed == null ? 700 : o.scrollingSpeed);
    var easing = o.easingcss3 || 'cubic-bezier(0.22, 1, 0.36, 1)';
    var anchors = o.anchors || [];
    var current = 0;
    var busy = false;
    var touchY = null;

    document.documentElement.style.overflow = 'hidden';
    document.body.style.overflow = 'hidden';
    root.style.transition = speed ? 'transform ' + speed + 'ms ' + easing : 'none';
    root.style.willChange = 'transform';

    // Content is wrapped in the same .fp-tableCell the page's CSS already targets.
    sections.forEach(function (sec, i) {
      if (!sec.querySelector(':scope > .fp-tableCell')) {
        var cell = document.createElement('div');
        cell.className = 'fp-tableCell';
        while (sec.firstChild) cell.appendChild(sec.firstChild);
        sec.appendChild(cell);
      }
      if (anchors[i]) sec.setAttribute('data-anchor', anchors[i]);
    });
    sections[0].classList.add('active');

    function info(i) {
      return { index: i, anchor: anchors[i] || null, item: sections[i] };
    }

    // Navigation dots (fullPage's #fp-nav markup, so the page's existing styles apply).
    var nav = null;
    if (o.navigation) {
      nav = document.createElement('div');
      nav.id = 'fp-nav';
      nav.className = 'fp-right';
      nav.setAttribute('aria-label', 'Sections');
      var ul = document.createElement('ul');
      sections.forEach(function (sec, i) {
        var li = document.createElement('li');
        var a = document.createElement('a');
        a.href = '#';
        a.setAttribute('aria-label', (o.navigationTooltips && o.navigationTooltips[i]) || 'Section ' + (i + 1));
        a.appendChild(document.createElement('span'));
        a.addEventListener('click', function (e) { e.preventDefault(); go(i); });
        li.appendChild(a);
        if (o.navigationTooltips && o.navigationTooltips[i]) {
          var tip = document.createElement('div');
          tip.className = 'fp-tooltip fp-right';
          tip.textContent = o.navigationTooltips[i];
          li.appendChild(tip);
        }
        ul.appendChild(li);
      });
      nav.appendChild(ul);
      document.body.appendChild(nav);
    }

    function paintNav() {
      if (!nav) return;
      Array.prototype.forEach.call(nav.querySelectorAll('a'), function (a, i) {
        a.classList.toggle('active', i === current);
      });
    }

    function go(target) {
      target = Math.max(0, Math.min(sections.length - 1, target));
      if (target === current || busy) return;
      var origin = current;
      busy = true;
      if (o.onLeave) o.onLeave(info(origin), info(target));
      current = target;
      root.style.transform = 'translate3d(0,' + (-100 * target) + 'vh,0)';
      var finish = function () {
        sections[origin].classList.remove('active');
        sections[target].classList.add('active');
        paintNav();
        busy = false;
        if (o.afterLoad) o.afterLoad(info(origin), info(target));
      };
      if (speed) setTimeout(finish, speed + 40); else finish();
    }

    function inNormalScroll(el) {
      return !!(o.normalScrollElements && el && el.closest && el.closest(o.normalScrollElements));
    }

    window.addEventListener('wheel', function (e) {
      if (inNormalScroll(e.target)) return;
      e.preventDefault();
      if (Math.abs(e.deltaY) < 4) return;
      go(current + (e.deltaY > 0 ? 1 : -1));
    }, { passive: false });

    window.addEventListener('touchstart', function (e) { touchY = e.touches[0].clientY; }, { passive: true });
    window.addEventListener('touchend', function (e) {
      if (touchY == null || inNormalScroll(e.target)) { touchY = null; return; }
      var dy = touchY - e.changedTouches[0].clientY;
      touchY = null;
      if (Math.abs(dy) > (window.innerHeight * (o.touchSensitivity || 5)) / 100) go(current + (dy > 0 ? 1 : -1));
    }, { passive: true });

    window.addEventListener('keydown', function (e) {
      var t = e.target;
      if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
      if (e.key === 'ArrowDown' || e.key === 'PageDown') { e.preventDefault(); go(current + 1); }
      else if (e.key === 'ArrowUp' || e.key === 'PageUp') { e.preventDefault(); go(current - 1); }
      else if (e.key === 'Home') { e.preventDefault(); go(0); }
      else if (e.key === 'End') { e.preventDefault(); go(sections.length - 1); }
    });

    window.fullpage_api = {
      moveTo: function (where) {
        var i = typeof where === 'string' ? anchors.indexOf(where) : Number(where) - 1;
        if (i >= 0) go(i);
      },
      moveSectionDown: function () { go(current + 1); },
      moveSectionUp: function () { go(current - 1); },
    };

    paintNav();
    setTimeout(function () { if (o.afterLoad) o.afterLoad(null, info(0)); }, 0);
  }

  window.fullpage = Pager;
})();
