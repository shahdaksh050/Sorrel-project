import streamlit.components.v1 as components


def inject_micro_interactions() -> None:
    """
    Injects Anime.js micro-interactions into the Streamlit parent DOM.
    Designed to be robust against Streamlit's frequent re-renders by tracking
    animated elements with data attributes and injecting the library only once.
    """
    js_code = """
    <script>
        const pWin = window.parent;
        const pDoc = pWin.document;

        function runAnimations() {
            // 1. Vanilla JS Count-up effect (Runs independently of anime.js)
            const countUps = pDoc.querySelectorAll('.count-up:not([data-animated])');
            if (countUps.length > 0) {
                countUps.forEach(el => {
                    el.setAttribute('data-animated', 'true');
                    const targetVal = parseFloat(el.getAttribute('data-value') || "0");
                    const prefix = el.getAttribute('data-prefix') || "";
                    const suffix = el.getAttribute('data-suffix') || "";
                    const isInt = parseInt(el.getAttribute('data-value')) === targetVal;

                    const duration = 1500;
                    const startTime = performance.now();

                    // easeOutExpo function
                    const easeOutExpo = (t) => t === 1 ? 1 : 1 - Math.pow(2, -10 * t);

                    function updateCounter(currentTime) {
                        const elapsed = currentTime - startTime;
                        const progress = Math.min(elapsed / duration, 1);
                        const easedProgress = easeOutExpo(progress);

                        const currentVal = targetVal * easedProgress;
                        const formatted = isInt ? Math.round(currentVal) : currentVal.toFixed(1);
                        el.innerHTML = prefix + formatted + suffix;

                        if (progress < 1) {
                            requestAnimationFrame(updateCounter);
                        } else {
                            const finalFormatted = isInt ? Math.round(targetVal) : targetVal.toFixed(1);
                            el.innerHTML = prefix + finalFormatted + suffix;
                        }
                    }
                    // Add slight random delay to mimic anime stagger
                    setTimeout(() => requestAnimationFrame(updateCounter), Math.random() * 300);
                });
            }

            // The following animations require Anime.js
            if (typeof pWin.anime === 'undefined') return;
            const anime = pWin.anime;

            // 2. Message bubbles (handoff items)
            const handoffs = pDoc.querySelectorAll('.handoff-item:not([data-animated])');
            if (handoffs.length > 0) {
                handoffs.forEach(el => el.setAttribute('data-animated', 'true'));
                anime({
                    targets: handoffs,
                    translateY: [20, 0],
                    opacity: [0, 1],
                    delay: anime.stagger(100),
                    duration: 600,
                    easing: 'easeOutCubic'
                });
            }

            // 3. Metric cards (gauges / datum cells)
            const gauges = pDoc.querySelectorAll('.gauge:not([data-animated]), .datum .cell:not([data-animated])');
            if (gauges.length > 0) {
                gauges.forEach(el => el.setAttribute('data-animated', 'true'));
                anime({
                    targets: gauges,
                    scale: [0.9, 1],
                    opacity: [0, 1],
                    delay: anime.stagger(80),
                    duration: 500,
                    easing: 'easeOutBack'
                });
            }

            // 4. Hover float effect for charts
            const charts = pDoc.querySelectorAll('[data-testid="stVegaLiteChart"]:not([data-floating])');
            if (charts.length > 0) {
                charts.forEach(el => {
                    el.setAttribute('data-floating', 'true');
                    let floatAnim;
                    el.addEventListener('mouseenter', () => {
                        floatAnim = anime({
                            targets: el,
                            translateY: -4,
                            direction: 'alternate',
                            loop: true,
                            duration: 1200,
                            easing: 'easeInOutSine'
                        });
                    });
                    el.addEventListener('mouseleave', () => {
                        if (floatAnim) floatAnim.pause();
                        anime({
                            targets: el,
                            translateY: 0,
                            duration: 300,
                            easing: 'easeOutQuad'
                        });
                    });
                });
            }

            // 5. SVG Donut Gauge Animation
            const animeGauges = pDoc.querySelectorAll('.anime-gauge:not([data-animated])');
            if (animeGauges.length > 0) {
                animeGauges.forEach(el => {
                    el.setAttribute('data-animated', 'true');
                    const q = parseFloat(el.getAttribute('data-q') || "0");
                    const arcLength = parseFloat(el.getAttribute('stroke-dasharray'));
                    const dashOffset = arcLength - (arcLength * q / 100);
                    anime({
                        targets: el,
                        strokeDashoffset: [arcLength, dashOffset],
                        duration: 1800,
                        easing: 'easeOutExpo',
                        delay: anime.random(200, 400)
                    });
                });
            }
        }

        if (!pWin.__anime_script_loaded) {
            pWin.__anime_script_loaded = true;
            const script = pDoc.createElement('script');
            script.src = "https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js";
            script.onload = () => {
                pWin.__anime_ready = true;
                runAnimations();
            };
            script.onerror = () => {
                // If anime.js fails to load, at least run vanilla animations
                runAnimations();
            };
            pDoc.head.appendChild(script);
        } else {
            runAnimations();
        }

        // components.html mounts this script in a fresh iframe on every Streamlit
        // rerun, but pWin/pDoc are the parent window/document, which persist across
        // reruns — so the observer must be stored there too, and the previous one
        // disconnected first, or each rerun leaks another MutationObserver onto body.
        if (pWin.__anime_observer) {
            pWin.__anime_observer.disconnect();
        }
        pWin.__anime_observer = new MutationObserver(() => {
            runAnimations();
        });
        pWin.__anime_observer.observe(pDoc.body, { childList: true, subtree: true });
    </script>
    """
    components.html(js_code, height=0)
