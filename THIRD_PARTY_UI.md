# Third-Party UI & Licences

This project's frontend relies on the following third-party code and design resources.

## 1. Design Inspiration

**Aceternity UI** was evaluated and is used strictly for **design inspiration only**. 
- The official Aceternity licence (`https://ui.aceternity.com/licence`) forbids redistribution of source files. 
- **No Aceternity code is copied, ported, or adapted** in this repository.

## 2. Adopted UI Components (Magic UI)

Where a design pattern matched an Aceternity concept, replacements were built using **Magic UI** components or original CSS.

Magic UI is distributed under the **MIT Licence**.
Copyright (c) Magic UI.

| Component | Upstream Source | Licence | Snapshot Date / Version | Local Modifications |
|---|---|---|---|---|
| `TextAnimate` | Magic UI (`text-animate`) | MIT | 2026-10-03 (from registry) | Trimmed to by-word blurIn only; removed Tailwind/`cn`; swapped `motion` for `LazyMotion`; added screen-reader access and reduced-motion guard. |

*Note: The Magic UI registry does not provide component version numbers; the snapshot date serves as the version.*

## 3. Original Implementations

The following UI elements were written from scratch as **original CSS/JS** using the app's design tokens. No third-party source was used for these:
- **Hero Spotlight** (`radial-gradient` from `--accent`).
- **Progress RunStepper** (Original Tier-2 CSS + Native HTML).
- **Pointer-following Glow** (Tier-2 pure CSS/JS on hovered cards).
- **Hover Border Gradient** (`@property` angle).
- **Dropzone Styling** (Dashed `--rule-strong` border on `stFileUploader`).
- **Tabs / Sliding Pill** (Restyled Streamlit tabs indicator).

## 4. Frontend Dependencies (npm)

The `ui/islands` build system uses the following dependencies. These are **build-time/dev-time only** except for the compiled bundle which includes React and Motion.

| Package | Use | Licence |
|---|---|---|
| `react`, `react-dom` (v19) | Island runtime | MIT |
| `motion` (v14) | Animations (`LazyMotion` only) | MIT |
| `vite`, `typescript`, `@vitejs/plugin-react` | Build tooling (not shipped) | MIT |
| `@types/react`, `@types/react-dom`, `@types/node` | TypeScript definitions | MIT |

*TailwindCSS, Radix UI, Tabler Icons, and Lucide React are NOT used.*

## 5. Vendored Assets (CDN / Static)

The repository currently relies on several vendored or CDN-loaded assets:

- **GSAP (GreenSock)**: Used for 3D pipeline animations. Vendored in `static/vendor/gsap`. **Licence Note (rechecked 2026-10-03):** since Webflow's acquisition GSAP, including the former Club plugins, is free for commercial use under the GSAP Standard License (https://gsap.com/community/standard-license/). It is *not* open source or MIT: Webflow owns it and may change the terms, and the licence bars use inside visual no-code animation builders that compete with Webflow. This app is not such a tool.
- **anime.js**: Used for animations. Both major versions (3.x and 4.x) are vendored in `static/vendor/anime`. MIT Licence.
- **Three.js**: Vendored in `static/vendor/three`. MIT Licence.
- **Fonts**: `Bricolage Grotesque` (headings, variable), `Public Sans` (text, variable) and `IBM Plex Mono` (400, 500, data) are the workspace fonts, vendored as Latin-subset `.woff2` in `static/vendor/fonts` from the `@fontsource` packages. All are **SIL Open Font License 1.1**; the licence texts sit beside the files (`LICENSE-*.txt`). `Baloo 2` and `Mukta` (also OFL) remain vendored only for the landing page and the HTML report until they move to the new families.
- **fullPage.js**: removed (2026-10-03). It was GPLv3 or commercial and loaded from a CDN, which broke `LOCAL_ONLY` and offline use. The cinematic page now uses `ui/assets/section_pager.js`, a small first-party pager that keeps the `fullpage` / `fullpage_api` names. The cinematic page's Three.js is imported from the vendored copy (inlined as a data URL in the standalone export), not from jsdelivr.
