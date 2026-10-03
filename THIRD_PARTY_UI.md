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

- **GSAP (GreenSock)**: Used for 3D pipeline animations. It is currently loaded from a CDN. **Licence Note:** GSAP operates under a custom "Standard No Charge" licence, *not* MIT. Free for most commercial uses unless charging multiple users for a product that uses it.
- **anime.js**: Used for animations. Loaded from CDN/vendored. MIT Licence.
- **Three.js**: Vendored in `static/vendor/three`. MIT Licence.
- **Google Fonts**: `Baloo 2` and `Mukta` are used. They are covered by the **SIL Open Font License (OFL)**.
- **fullPage.js**: Used in the cinematic export. Loaded from CDN. **Known Issue:** The CDN load conflicts with `LOCAL_ONLY` network isolation. Additionally, fullPage.js v4 operates under GPLv3 or a paid commercial licence, which may conflict with the project's "free only" constraints and distribution model. (Pending resolution).
