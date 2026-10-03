import streamlit.components.v1 as components


def inject_micro_interactions() -> None:
    """
    Injects a pointer glow script into the Streamlit parent DOM.
    Sets --mx and --my CSS variables on hovered .bento-card and .finding-card elements.
    Throttled via requestAnimationFrame.
    """
    js_code = """
    <script>
        const pWin = window.parent;
        const pDoc = pWin.document;

        if (!pWin.__dsaPointer) {
            pWin.__dsaPointer = true;

            let ticking = false;
            let mouseX = 0;
            let mouseY = 0;
            let currentCard = null;

            pDoc.addEventListener('pointermove', (e) => {
                // Respect reduced motion: disable pointer glow
                if (pWin.matchMedia('(prefers-reduced-motion: reduce)').matches) {
                    return;
                }

                mouseX = e.clientX;
                mouseY = e.clientY;
                
                const card = e.target.closest('.bento-card, .finding-card');
                if (card !== currentCard) {
                    currentCard = card;
                }

                if (currentCard && !ticking) {
                    pWin.requestAnimationFrame(() => {
                        if (currentCard) {
                            const rect = currentCard.getBoundingClientRect();
                            const mx = mouseX - rect.left;
                            const my = mouseY - rect.top;
                            currentCard.style.setProperty('--mx', `${mx}px`);
                            currentCard.style.setProperty('--my', `${my}px`);
                        }
                        ticking = false;
                    });
                    ticking = true;
                }
            }, { passive: true });
        }
    </script>
    """
    components.html(js_code, height=0)
