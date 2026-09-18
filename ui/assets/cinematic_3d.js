/**
 * Beautiful 3D Frontend: Three.js + fullPage.js + Anime.js v4 Master Architecture
 * Unified 6-Stage Autonomous Data Analysis Cinematic Continuous Journey
 *
 * 7 Architectural Pillars (Verified Implementation):
 * 1. Single Source of Time (renderer.setAnimationLoop synchronously drives Anime.js engine + physics)
 * 2. Golden 4-Layer DOM Hierarchy (Containment Fix)
 * 3. Continuous Camera Spline (THREE.CatmullRomCurve3 driven by critically damped scalar s in [0, 5])
 * 4. MetaMask-Style Cursor Tracking Physics (Damped frame-rate-independent NDC Lerp with camera trucking)
 * 5. Dynamic 3D-to-2D Vector Leader Lines (Bounding-box-aware angled elbow overlay)
 * 6. Mansory-Standard Zero-Freeze Preloader (renderer.compile scene & shaders + WAAPI curtain)
 * 7. Strict GPU Performance Budget (DPR <= 2.0, InstancedMesh single-draw-call Matter & Link fields)
 */

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { engine, animate, stagger, cubicBezier } from 'https://cdn.jsdelivr.net/npm/animejs@4.5.0/dist/bundles/anime.esm.min.js';

// Accurate Cubic Bezier (0.22, 1, 0.36, 1) Solver (Defect 5 Fix)
export function createCubicBezierSolver(x1, y1, x2, y2) {
  const cx = 3.0 * x1;
  const bx = 3.0 * (x2 - x1) - cx;
  const ax = 1.0 - cx - bx;

  const cy = 3.0 * y1;
  const by = 3.0 * (y2 - y1) - cy;
  const ay = 1.0 - cy - by;

  function sampleCurveX(t) {
    return ((ax * t + bx) * t + cx) * t;
  }
  function sampleCurveY(t) {
    return ((ay * t + by) * t + cy) * t;
  }
  function sampleCurveDerivativeX(t) {
    return (3.0 * ax * t + 2.0 * bx) * t + cx;
  }

  return function solve(x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    let t = x;
    for (let i = 0; i < 8; i++) {
      const currentX = sampleCurveX(t) - x;
      if (Math.abs(currentX) < 1e-6) return sampleCurveY(t);
      const dX = sampleCurveDerivativeX(t);
      if (Math.abs(dX) < 1e-6) break;
      t -= currentX / dX;
    }
    let t0 = 0.0;
    let t1 = 1.0;
    t = x;
    while (t0 < t1) {
      const currentX = sampleCurveX(t);
      if (Math.abs(currentX - x) < 1e-6) return sampleCurveY(t);
      if (x > currentX) t0 = t;
      else t1 = t;
      t = (t1 + t0) * 0.5;
    }
    return sampleCurveY(t);
  };
}

export const easeOutDecel = createCubicBezierSolver(0.22, 1, 0.36, 1);

/* -------------------------------------------------------------------------- */
/* PILLAR 1 & SECTION 6: MOTION TOKENS CONTRACT                               */
/* -------------------------------------------------------------------------- */
export const MOTION = {
  dur: {
    micro: 180,
    ui: 320,
    stage: 700,
    cinematic: 1400,
  },
  ease: {
    out: 'outCubic',
    inOut: 'easeInOutCubic',
    spring: 'spring(1, 80, 10, 0)',
  },
  damp: {
    cursor: 3.5,    // lambda, 1/sec for mouse rotation & truck
    journey: 7.5,   // omega, spring stiffness for journey scalar s
    velocity: 4.0,  // lambda, decay rate for scroll momentum & velocity feedback
  },
  stagger: 40,
};


export function smoothstep(x) {
  const c = Math.min(Math.max(x, 0), 1);
  return c * c * (3 - 2 * c);
}

// Decouple Anime.js standalone ticker (Single source of time: Three.js render loop drives it)
if (engine && typeof engine === 'object') {
  engine.useDefaultMainLoop = false;
}

/* -------------------------------------------------------------------------- */
/* STATE & PALETTE MANAGEMENT (Warm Ledger Aesthetic)                         */
/* -------------------------------------------------------------------------- */
const RAW_STATE = window.__CINEMATIC_STATE__ || {};
// Compact mode (workspace hero box) prepends a step 0 section, so fullPage section N maps to
// journey stage N - SECTION_OFFSET. The journey scalar s in [0, 5] itself never changes.
const HAS_INTRO = !!RAW_STATE.compact;
const SECTION_OFFSET = HAS_INTRO ? 1 : 0;
let THEME = RAW_STATE.theme || 'night';
let isNight = THEME === 'night';
const P = RAW_STATE.palette || {};

const PALETTES = {
  night: {
    stock: (isNight && P.stock) || '#241c14',
    sheet: (isNight && P.sheet) || '#1c1610',
    pen: (isNight && P.pen) || '#f0a24a',
    accent: (isNight && P.accent) || '#4fc3f7',
    risk: (isNight && P.risk) || '#e2685a',
    positive: (isNight && P.positive) || '#7fb77e',
    graphite: (isNight && P.graphite) || '#bdae97',
    grid: (isNight && P.grid) || '#4a3c28',
    coreEmissive: 0x6e3805,
    ambient: 0x2e2419,
    ambientInt: 1.4,
    key: 0xffd9a6,
    keyInt: 1.8,
    fill: 0x4fc3f7,
    fillInt: 0.75,
    rim: 0xf0a24a,
    rimInt: 0.9,
    fogDensity: 0.024,
    gridOpacity: 0.14,
  },
  day: {
    stock: (!isNight && P.stock) || '#f7eedd',
    sheet: (!isNight && P.sheet) || '#fffbf2',
    pen: (!isNight && P.pen) || '#a34f20',
    accent: (!isNight && P.accent) || '#e08a3e',
    risk: (!isNight && P.risk) || '#a33526',
    positive: (!isNight && P.positive) || '#5b8c5a',
    graphite: (!isNight && P.graphite) || '#8a7660',
    grid: (!isNight && P.grid) || '#e4d4bc',
    coreEmissive: 0x3d1c00,
    ambient: 0xfbf4e6,
    ambientInt: 1.1,
    key: 0xfffaed,
    keyInt: 1.2,
    fill: 0xd8edf5,
    fillInt: 0.55,
    rim: 0xa34f20,
    rimInt: 0.6,
    fogDensity: 0.022,
    gridOpacity: 0.09,
  },
};

function getActivePalette() {
  return isNight ? PALETTES.night : PALETTES.day;
}

// Prefers-reduced-motion listener (Defect 11 Fix)
const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
let prefersReducedMotion = motionQuery.matches;
motionQuery.addEventListener('change', (e) => {
  prefersReducedMotion = e.matches;
});

/* -------------------------------------------------------------------------- */
/* PILLAR 7: GPU PERFORMANCE BUDGET - THREE.JS SETUP                          */
/* -------------------------------------------------------------------------- */
const canvas = document.getElementById('webgl-canvas');
const initialStockColor = new THREE.Color(getActivePalette().stock);
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
  powerPreference: 'high-performance',
});

renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2.0));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(initialStockColor, 1);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = isNight ? 1.15 : 1.0;

const scene = new THREE.Scene();
scene.background = initialStockColor;
scene.fog = new THREE.FogExp2(initialStockColor.getHex(), getActivePalette().fogDensity);

const BASE_FOV = 46;
const camera = new THREE.PerspectiveCamera(BASE_FOV, window.innerWidth / window.innerHeight, 0.1, 1000);

// Global Scene Hierarchy: Master World & Cursor Parallax Rig (Pillar 4)
const worldGroup = new THREE.Group();
scene.add(worldGroup);

const parallaxGroup = new THREE.Group();
worldGroup.add(parallaxGroup);

// Subject centered at origin; camera.setViewOffset positions it in the right pane
parallaxGroup.position.set(0, 0, 0);

/* -------------------------------------------------------------------------- */
/* LIGHTING RIG                                                               */
/* -------------------------------------------------------------------------- */
const pal = getActivePalette();
const ambientLight = new THREE.AmbientLight(pal.ambient, pal.ambientInt);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(pal.key, pal.keyInt);
keyLight.position.set(10, 16, 12);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(pal.fill, pal.fillInt);
fillLight.position.set(-12, -6, -8);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(pal.rim, pal.rimInt);
rimLight.position.set(0, 14, -12);
scene.add(rimLight);

// Floor Engineering Grid Plane
const gridHelper = new THREE.GridHelper(44, 44, new THREE.Color(pal.grid), new THREE.Color(pal.grid));
gridHelper.position.set(0, -3.2, 0);
gridHelper.material.transparent = true;
gridHelper.material.opacity = pal.gridOpacity;
worldGroup.add(gridHelper);

/* -------------------------------------------------------------------------- */
/* SECTION 4.3: CONTINUOUS MATTER & LINK SUBSTRATE                            */
/* -------------------------------------------------------------------------- */
const MATTER_COUNT = 2048;
const LINK_PAIRS = 256;

// Precomputed coordinate target buffers for each of the 6 stages [0..5]
const stageBuffers = [
  new Float32Array(MATTER_COUNT * 3), // Stage 0: Gimbal Core & Orbital Rings
  new Float32Array(MATTER_COUNT * 3), // Stage 1: Orderly 3D Grid Lattice (Defect 13 Fix)
  new Float32Array(MATTER_COUNT * 3), // Stage 2: Feature Scatter Constellation
  new Float32Array(MATTER_COUNT * 3), // Stage 3: Stratified Decision Manifolds
  new Float32Array(MATTER_COUNT * 3), // Stage 4: Anti-Overfit Generalization Envelope
  new Float32Array(MATTER_COUNT * 3), // Stage 5: Sealed Executive Ledger Dossier
];

// Stage Colors per instance
const stageColors = [
  new Float32Array(MATTER_COUNT * 3),
  new Float32Array(MATTER_COUNT * 3),
  new Float32Array(MATTER_COUNT * 3),
  new Float32Array(MATTER_COUNT * 3),
  new Float32Array(MATTER_COUNT * 3),
  new Float32Array(MATTER_COUNT * 3),
];

const cPen = new THREE.Color(pal.pen);
const cAccent = new THREE.Color(pal.accent);
const cPositive = new THREE.Color(pal.positive);
const cGraphite = new THREE.Color(pal.graphite);

// Initialize Target Buffers
// -- Stage 0: Gimbal Core Surface + Concentric Rings
for (let i = 0; i < MATTER_COUNT; i++) {
  const i3 = i * 3;
  if (i < 768) {
    // Surface of central core sphere/dodecahedron (Fibonacci sphere distribution)
    const phi = Math.acos(1 - (2 * (i + 0.5)) / 768);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    const r = 1.48;
    stageBuffers[0][i3] = r * Math.sin(phi) * Math.cos(theta);
    stageBuffers[0][i3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    stageBuffers[0][i3 + 2] = r * Math.cos(phi);
    stageColors[0][i3] = cPen.r;
    stageColors[0][i3 + 1] = cPen.g;
    stageColors[0][i3 + 2] = cPen.b;
  } else if (i < 1200) {
    // Ring 1 (Torus, R = 2.35, XZ plane)
    const angle = ((i - 768) / (1200 - 768)) * Math.PI * 2;
    stageBuffers[0][i3] = 2.35 * Math.cos(angle);
    stageBuffers[0][i3 + 1] = (Math.random() - 0.5) * 0.08;
    stageBuffers[0][i3 + 2] = 2.35 * Math.sin(angle);
    stageColors[0][i3] = cPen.r;
    stageColors[0][i3 + 1] = cPen.g;
    stageColors[0][i3 + 2] = cPen.b;
  } else if (i < 1630) {
    // Ring 2 (Torus, R = 2.95, XY plane)
    const angle = ((i - 1200) / (1630 - 1200)) * Math.PI * 2;
    stageBuffers[0][i3] = 2.95 * Math.cos(angle);
    stageBuffers[0][i3 + 1] = 2.95 * Math.sin(angle);
    stageBuffers[0][i3 + 2] = (Math.random() - 0.5) * 0.08;
    stageColors[0][i3] = cAccent.r;
    stageColors[0][i3 + 1] = cAccent.g;
    stageColors[0][i3 + 2] = cAccent.b;
  } else {
    // Ring 3 (Torus, R = 2.45, YZ plane)
    const angle = ((i - 1630) / (MATTER_COUNT - 1630)) * Math.PI * 2;
    stageBuffers[0][i3] = (Math.random() - 0.5) * 0.08;
    stageBuffers[0][i3 + 1] = 2.45 * Math.cos(angle);
    stageBuffers[0][i3 + 2] = 2.45 * Math.sin(angle);
    stageColors[0][i3] = cGraphite.r;
    stageColors[0][i3 + 1] = cGraphite.g;
    stageColors[0][i3 + 2] = cGraphite.b;
  }
}

// -- Stage 1: Orderly 3D Grid Lattice (16 x 8 x 16 = 2048 points) (Defect 13 Fix)
let latticeIdx = 0;
for (let ix = 0; ix < 16; ix++) {
  for (let iy = 0; iy < 8; iy++) {
    for (let iz = 0; iz < 16; iz++) {
      if (latticeIdx >= MATTER_COUNT) break;
      const i3 = latticeIdx * 3;
      stageBuffers[1][i3] = (ix - 7.5) * 0.22;
      stageBuffers[1][i3 + 1] = (iy - 3.5) * 0.30;
      stageBuffers[1][i3 + 2] = (iz - 7.5) * 0.22;

      const colChoice = (ix + iy + iz) % 3;
      const chosenColor = colChoice === 0 ? cAccent : colChoice === 1 ? cPen : cPositive;
      stageColors[1][i3] = chosenColor.r;
      stageColors[1][i3 + 1] = chosenColor.g;
      stageColors[1][i3 + 2] = chosenColor.b;
      latticeIdx++;
    }
  }
}

// -- Stage 2: Feature Scatter Constellation & Clustered Galaxies
for (let i = 0; i < MATTER_COUNT; i++) {
  const i3 = i * 3;
  const cluster = i % 3;
  const progress = i / MATTER_COUNT;
  const radius = 0.5 + Math.pow(Math.random(), 0.5) * 2.2;
  const theta = progress * Math.PI * 8.0 + (cluster * (Math.PI * 2 / 3));
  const height = (Math.random() - 0.5) * (3.0 - radius * 0.5);

  stageBuffers[2][i3] = radius * Math.cos(theta) + (Math.random() - 0.5) * 0.35;
  stageBuffers[2][i3 + 1] = height;
  stageBuffers[2][i3 + 2] = radius * Math.sin(theta) + (Math.random() - 0.5) * 0.35;

  const chosenColor = cluster === 0 ? cPen : cluster === 1 ? cAccent : cPositive;
  stageColors[2][i3] = chosenColor.r;
  stageColors[2][i3 + 1] = chosenColor.g;
  stageColors[2][i3 + 2] = chosenColor.b;
}

// -- Stage 3: Stratified Decision Manifolds (5 Hyperplane Sheets)
const FOLD_COUNT = 5;
const ptsPerFold = Math.floor(MATTER_COUNT / FOLD_COUNT);
for (let i = 0; i < MATTER_COUNT; i++) {
  const i3 = i * 3;
  const fold = Math.min(Math.floor(i / ptsPerFold), FOLD_COUNT - 1);
  const localIdx = i - fold * ptsPerFold;
  const u = (localIdx % 21) / 20.0;
  const v = Math.floor(localIdx / 21) / 19.0;

  const vx = (u - 0.5) * 3.4;
  const vy = (v - 0.5) * 2.2;
  const vz = (fold - 2) * 0.55 + Math.sin(vx * 0.85 + fold) * 0.22 + Math.cos(vy * 0.85) * 0.16;

  stageBuffers[3][i3] = vx + (fold - 2) * 0.32;
  stageBuffers[3][i3 + 1] = vy + (fold - 2) * 0.18;
  stageBuffers[3][i3 + 2] = vz;

  const isBest = fold === 2;
  const chosenColor = isBest ? cPen : cAccent;
  stageColors[3][i3] = chosenColor.r;
  stageColors[3][i3 + 1] = chosenColor.g;
  stageColors[3][i3 + 2] = chosenColor.b;
}

// -- Stage 4: Anti-Overfit Generalization Envelope (Octahedron Faces/Edges + Guard Ring)
for (let i = 0; i < MATTER_COUNT; i++) {
  const i3 = i * 3;
  if (i < 900) {
    let x = Math.random() - 0.5;
    let y = Math.random() - 0.5;
    let z = Math.random() - 0.5;
    const l1 = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
    const r = 2.05;
    stageBuffers[4][i3] = (x / l1) * r;
    stageBuffers[4][i3 + 1] = (y / l1) * r;
    stageBuffers[4][i3 + 2] = (z / l1) * r;
    stageColors[4][i3] = cPositive.r;
    stageColors[4][i3 + 1] = cPositive.g;
    stageColors[4][i3 + 2] = cPositive.b;
  } else if (i < 1550) {
    let x = Math.random() - 0.5;
    let y = Math.random() - 0.5;
    let z = Math.random() - 0.5;
    const l1 = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
    const r = 1.65;
    stageBuffers[4][i3] = (x / l1) * r;
    stageBuffers[4][i3 + 1] = (y / l1) * r;
    stageBuffers[4][i3 + 2] = (z / l1) * r;
    stageColors[4][i3] = cPen.r;
    stageColors[4][i3 + 1] = cPen.g;
    stageColors[4][i3 + 2] = cPen.b;
  } else {
    const angle = ((i - 1550) / (MATTER_COUNT - 1550)) * Math.PI * 2;
    stageBuffers[4][i3] = 2.45 * Math.cos(angle);
    stageBuffers[4][i3 + 1] = (Math.random() - 0.5) * 0.08;
    stageBuffers[4][i3 + 2] = 2.45 * Math.sin(angle);
    stageColors[4][i3] = cPositive.r;
    stageColors[4][i3 + 1] = cPositive.g;
    stageColors[4][i3 + 2] = cPositive.b;
  }
}

// -- Stage 5: Executive Dossier Tablet Plane & Concentric Wax Seal
for (let i = 0; i < MATTER_COUNT; i++) {
  const i3 = i * 3;
  if (i < 1440) {
    const u = (i % 48) / 47.0;
    const v = Math.floor(i / 48) / 29.0;
    stageBuffers[5][i3] = (u - 0.5) * 3.4;
    stageBuffers[5][i3 + 1] = (v - 0.5) * 2.4;
    stageBuffers[5][i3 + 2] = (Math.random() - 0.5) * 0.05;
    stageColors[5][i3] = cGraphite.r;
    stageColors[5][i3 + 1] = cGraphite.g;
    stageColors[5][i3 + 2] = cGraphite.b;
  } else {
    const ringIdx = (i - 1440) / (MATTER_COUNT - 1440);
    const r = Math.sqrt(ringIdx) * 0.42;
    const angle = (i - 1440) * 2.39996;
    stageBuffers[5][i3] = 1.2 + r * Math.cos(angle);
    stageBuffers[5][i3 + 1] = 0.75 + r * Math.sin(angle);
    stageBuffers[5][i3 + 2] = 0.08 + (Math.random() - 0.5) * 0.03;
    stageColors[5][i3] = cPen.r;
    stageColors[5][i3 + 1] = cPen.g;
    stageColors[5][i3 + 2] = cPen.b;
  }
}

// -- Step 0 (compact mode): loose, unformed cloud of raw rows that assembles into stage 0.
// Blended over the journey morph by the intro weight, so it needs no slot in stageBuffers.
const cloudBuffer = new Float32Array(MATTER_COUNT * 3);
const cloudColors = new Float32Array(MATTER_COUNT * 3);
for (let i = 0; i < MATTER_COUNT; i++) {
  const i3 = i * 3;
  // Uniform direction, radius biased toward the middle so the cloud reads as a soft volume
  const u = Math.random() * 2 - 1;
  const phi = Math.random() * Math.PI * 2;
  const r = Math.pow(Math.random(), 0.7);
  const sq = Math.sqrt(1 - u * u);
  cloudBuffer[i3] = r * sq * Math.cos(phi) * 2.7;
  cloudBuffer[i3 + 1] = r * u * 2.1;
  cloudBuffer[i3 + 2] = r * sq * Math.sin(phi) * 2.4;
  const roll = Math.random();
  const c = roll < 0.72 ? cGraphite : roll < 0.92 ? cPen : cAccent;
  cloudColors[i3] = c.r;
  cloudColors[i3 + 1] = c.g;
  cloudColors[i3 + 2] = c.b;
}

// Instantiate The Matter Field (Single Draw Call, Alive Whole Journey)
const matterGeo = new THREE.SphereGeometry(0.044, 6, 6);
const matterMat = new THREE.MeshStandardMaterial({
  color: 0xffffff,
  roughness: 0.22,
  metalness: 0.85,
  transparent: true,
  opacity: 0.92,
});
const matterMesh = new THREE.InstancedMesh(matterGeo, matterMat, MATTER_COUNT);
matterMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
parallaxGroup.add(matterMesh);

// Instantiate The Link Field (Single Draw Call LineSegments, Fixed Vertex Count)
const linkGeo = new THREE.BufferGeometry();
const linkPositions = new Float32Array(LINK_PAIRS * 2 * 3);
const linkColors = new Float32Array(LINK_PAIRS * 2 * 3);
linkGeo.setAttribute('position', new THREE.BufferAttribute(linkPositions, 3));
linkGeo.setAttribute('color', new THREE.BufferAttribute(linkColors, 3));
linkGeo.attributes.position.setUsage(THREE.DynamicDrawUsage);
linkGeo.attributes.color.setUsage(THREE.DynamicDrawUsage);

const linkMat = new THREE.LineBasicMaterial({
  vertexColors: true,
  transparent: true,
  opacity: 0.72,
  blending: THREE.NormalBlending,
});
const linkSegments = new THREE.LineSegments(linkGeo, linkMat);
parallaxGroup.add(linkSegments);

// Precompute Link Pair Index Mappings for each stage
const linkPairsMap = [];
for (let sIdx = 0; sIdx < 6; sIdx++) {
  const pairs = [];
  for (let k = 0; k < LINK_PAIRS; k++) {
    let pA = 0;
    let pB = 0;
    if (sIdx === 0) {
      pA = k % 300;
      pB = (k + 1) % 300;
    } else if (sIdx === 1) {
      pA = k * 4;
      pB = Math.min(k * 4 + 1, MATTER_COUNT - 1);
    } else if (sIdx === 2) {
      pA = (k * 7) % MATTER_COUNT;
      pB = (k * 7 + 13) % MATTER_COUNT;
    } else if (sIdx === 3) {
      pA = (k * 6) % MATTER_COUNT;
      pB = Math.min(pA + 21, MATTER_COUNT - 1);
    } else if (sIdx === 4) {
      pA = (k * 5) % 900;
      pB = (k * 5 + 12) % 900;
    } else {
      pA = (k * 4) % 1440;
      pB = (k * 4 + 48) % 1440;
    }
    pairs.push([pA, pB]);
  }
  linkPairsMap.push(pairs);
}

/* -------------------------------------------------------------------------- */
/* SOLID HERO MESHES (Cross-fading on top of Matter morph)                    */
/* -------------------------------------------------------------------------- */
// Stage 0: Gimbal Core Group
const rlmGroup = new THREE.Group();
parallaxGroup.add(rlmGroup);

const dodecaGeo = new THREE.DodecahedronGeometry(1.45, 1);
const dodecaMat = new THREE.MeshStandardMaterial({
  color: new THREE.Color(pal.pen),
  emissive: new THREE.Color(pal.coreEmissive),
  roughness: 0.22,
  metalness: 0.88,
  transparent: true,
  opacity: 1.0,
});
const rlmCore = new THREE.Mesh(dodecaGeo, dodecaMat);
rlmGroup.add(rlmCore);

const dodecaEdgesMat = new THREE.LineBasicMaterial({
  color: isNight ? 0xffe2b8 : 0x5a2d0d,
  transparent: true,
  opacity: 0.9,
});
const dodecaEdges = new THREE.LineSegments(new THREE.EdgesGeometry(dodecaGeo), dodecaEdgesMat);
rlmCore.add(dodecaEdges);

const ring1Geo = new THREE.TorusGeometry(2.35, 0.042, 16, 120);
const ring1Mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(pal.pen), roughness: 0.18, metalness: 0.92, transparent: true, opacity: 0.9 });
const ring1 = new THREE.Mesh(ring1Geo, ring1Mat);
rlmGroup.add(ring1);

const ring2Geo = new THREE.TorusGeometry(2.95, 0.038, 16, 120);
const ring2Mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(pal.accent), roughness: 0.22, metalness: 0.85, transparent: true, opacity: 0.85 });
const ring2 = new THREE.Mesh(ring2Geo, ring2Mat);
ring2.rotation.x = Math.PI / 2;
rlmGroup.add(ring2);

const ring3Geo = new THREE.TorusGeometry(2.45, 0.024, 16, 120);
const ring3Mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(pal.graphite), roughness: 0.4, metalness: 0.7, transparent: true, opacity: 0.75 });
const ring3 = new THREE.Mesh(ring3Geo, ring3Mat);
ring3.rotation.y = Math.PI / 3;
rlmGroup.add(ring3);

// Stage 1: Ingestion Group (Sweeping Cyan Laser Plane & Frame)
const ingestionGroup = new THREE.Group();
parallaxGroup.add(ingestionGroup);

const laserGeo = new THREE.BoxGeometry(5.4, 0.08, 3.8);
const laserMat = new THREE.MeshStandardMaterial({
  color: new THREE.Color(pal.accent),
  emissive: new THREE.Color(pal.accent),
  emissiveIntensity: 0.85,
  roughness: 0.2,
  metalness: 0.8,
  transparent: true,
  opacity: 0.80,
  side: THREE.DoubleSide,
});
const laserPlane = new THREE.Mesh(laserGeo, laserMat);
ingestionGroup.add(laserPlane);

const laserFrameGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(5.4, 3.8, 3.8));
const laserFrameMat = new THREE.LineBasicMaterial({ color: new THREE.Color(pal.accent), transparent: true, opacity: 0.65 });
const laserFrame = new THREE.LineSegments(laserFrameGeo, laserFrameMat);
ingestionGroup.add(laserFrame);

// Stage 2: Statistical Testing Group
const statsGroup = new THREE.Group();
parallaxGroup.add(statsGroup);

// Stage 3: ML Pipeline Manifold Plates (Defect 7 Fix: Stored Base Positions)
const pipelineGroup = new THREE.Group();
parallaxGroup.add(pipelineGroup);

const manifoldPlates = [];
const manifoldBaseY = [];
const manifoldBaseRotZ = [];

for (let i = 0; i < FOLD_COUNT; i++) {
  const plateGeo = new THREE.PlaneGeometry(3.4, 2.2, 12, 8);
  const posAttr = plateGeo.attributes.position;
  for (let j = 0; j < posAttr.count; j++) {
    const vx = posAttr.getX(j);
    const vy = posAttr.getY(j);
    posAttr.setZ(j, Math.sin(vx * 0.85 + i) * 0.22 + Math.cos(vy * 0.85) * 0.16);
  }
  plateGeo.computeVertexNormals();

  const isBestFold = i === 2;
  const plateMat = new THREE.MeshStandardMaterial({
    color: isBestFold ? new THREE.Color(pal.pen) : new THREE.Color(pal.accent),
    roughness: 0.2,
    metalness: 0.6,
    transparent: true,
    opacity: isBestFold ? 0.88 : 0.60,
    side: THREE.DoubleSide,
  });

  const plate = new THREE.Mesh(plateGeo, plateMat);
  const baseY = (i - 2) * 0.28;
  const baseRotZ = 0.15;
  plate.position.set((i - 2) * 0.42, baseY, (i - 2) * 0.48);
  plate.rotation.set(-0.35 + i * 0.08, 0.45 - i * 0.06, baseRotZ);
  pipelineGroup.add(plate);

  const wireframe = new THREE.LineSegments(
    new THREE.WireframeGeometry(plateGeo),
    new THREE.LineBasicMaterial({
      color: isBestFold ? new THREE.Color(pal.pen) : new THREE.Color(pal.graphite),
      transparent: true,
      opacity: 0.65,
    })
  );
  plate.add(wireframe);

  manifoldPlates.push(plate);
  manifoldBaseY.push(baseY);
  manifoldBaseRotZ.push(baseRotZ);
}

// Stage 4: Anti-Overfit Guard Group (Dual Octahedron & Guard Ring)
const diagnosticsGroup = new THREE.Group();
parallaxGroup.add(diagnosticsGroup);

const outerGeo = new THREE.OctahedronGeometry(2.1, 0);
const outerMat = new THREE.MeshStandardMaterial({
  color: new THREE.Color(pal.positive),
  roughness: 0.3,
  metalness: 0.6,
  transparent: true,
  opacity: 0.65,
  side: THREE.DoubleSide,
});
const radarOuter = new THREE.Mesh(outerGeo, outerMat);
const radarOuterEdges = new THREE.LineSegments(
  new THREE.EdgesGeometry(outerGeo),
  new THREE.LineBasicMaterial({ color: new THREE.Color(pal.positive), transparent: true, opacity: 0.95 })
);
radarOuter.add(radarOuterEdges);
diagnosticsGroup.add(radarOuter);

const innerGeo = new THREE.OctahedronGeometry(1.65, 0);
const innerMat = new THREE.MeshStandardMaterial({
  color: new THREE.Color(pal.positive),
  roughness: 0.25,
  metalness: 0.8,
  transparent: true,
  opacity: 0.75,
});
const radarInner = new THREE.Mesh(innerGeo, innerMat);
diagnosticsGroup.add(radarInner);

const guardRingGeo = new THREE.TorusGeometry(2.45, 0.065, 16, 80);
const guardRingMat = new THREE.MeshBasicMaterial({
  color: new THREE.Color(pal.positive),
  transparent: true,
  opacity: 0.85,
});
const guardRing = new THREE.Mesh(guardRingGeo, guardRingMat);
guardRing.rotation.x = Math.PI / 2;
diagnosticsGroup.add(guardRing);

// Stage 5: Executive Dossier Group (Frosted Tablet & Golden Wax Seal)
const reportGroup = new THREE.Group();
parallaxGroup.add(reportGroup);

const easelGeo = new THREE.BoxGeometry(3.4, 2.4, 0.08);
const easelMat = new THREE.MeshStandardMaterial({
  color: new THREE.Color(pal.sheet),
  roughness: 0.35,
  metalness: 0.25,
  transparent: true,
  opacity: 0.95,
});
const easel = new THREE.Mesh(easelGeo, easelMat);
reportGroup.add(easel);

const easelEdgesMat = new THREE.LineBasicMaterial({ color: new THREE.Color(pal.pen), transparent: true, opacity: 0.95 });
const easelEdges = new THREE.LineSegments(new THREE.EdgesGeometry(easelGeo), easelEdgesMat);
easel.add(easelEdges);

const sealGeo = new THREE.CylinderGeometry(0.42, 0.42, 0.10, 32);
const sealMat = new THREE.MeshStandardMaterial({
  color: new THREE.Color(pal.pen),
  emissive: new THREE.Color(pal.pen),
  emissiveIntensity: 0.4,
  roughness: 0.2,
  metalness: 0.9,
  transparent: true,
  opacity: 0.95,
});
const seal = new THREE.Mesh(sealGeo, sealMat);
seal.position.set(1.2, 0.75, 0.08);
seal.rotation.x = Math.PI / 2;
reportGroup.add(seal);

// Dossier Binding Spine Rings
const spineMat = new THREE.MeshStandardMaterial({ color: new THREE.Color(pal.graphite), metalness: 0.8, transparent: true, opacity: 0.85 });
for (let i = -0.9; i <= 0.9; i += 0.6) {
  const spineGeo = new THREE.TorusGeometry(0.16, 0.032, 12, 24);
  const spine = new THREE.Mesh(spineGeo, spineMat);
  spine.position.set(-1.7, i, 0);
  spine.rotation.y = Math.PI / 2;
  reportGroup.add(spine);
}

// Stage Groups Registry for Continuous Opacity Cross-Fading
const stageHeroGroups = [
  rlmGroup,
  ingestionGroup,
  statsGroup,
  pipelineGroup,
  diagnosticsGroup,
  reportGroup,
];

// Snapshot base opacity for each material to prevent over-saturation during fade
stageHeroGroups.forEach((grp) => {
  grp.traverse((child) => {
    if (child.material) {
      child.userData.baseOpacity = child.material.opacity !== undefined ? child.material.opacity : 0.85;
    }
  });
});

/* -------------------------------------------------------------------------- */
/* SECTION 4.2: CONTINUOUS CAMERA SPLINE & 6-STAGE NARRATIVE WAYPOINTS       */
/* -------------------------------------------------------------------------- */
export const WAYPOINTS = [
  // 0: Overview & RLM Core Engine: High-angle view centered on right-pane focal subject
  {
    camera: new THREE.Vector3(2.2, 2.6, 10.5),
    lookAt: new THREE.Vector3(0.0, 0.0, 0.0),
    hudTag: 'RLM CORE ENGINE // RECURSIVE DECOMPOSITION',
    hudTagClass: 'hud-tag',
    anchorObj: rlmCore,
  },
  // 1: Ingestion & Profiling: Pushing into the 3D voxel grid at raking laser angle
  {
    camera: new THREE.Vector3(1.2, 1.3, 7.8),
    lookAt: new THREE.Vector3(0.0, 0.1, 0.0),
    hudTag: 'INGESTION // REAL-TIME LASER PROFILING',
    hudTagClass: 'hud-tag cyan',
    anchorObj: laserPlane,
  },
  // 2: Statistical Testing: Low-angle dramatic upward look into constellation
  {
    camera: new THREE.Vector3(1.8, -1.6, 8.8),
    lookAt: new THREE.Vector3(0.0, -0.2, 0.0),
    hudTag: 'HYPOTHESIS DECISION // 3D CORRELATION WEB',
    hudTagClass: 'hud-tag',
    anchorObj: statsGroup,
  },
  // 3: ML Pipeline: Oblique raking slide along stacked decision manifolds
  {
    camera: new THREE.Vector3(2.2, 1.8, 8.2),
    lookAt: new THREE.Vector3(0.0, 0.15, 0.0),
    hudTag: 'ML PIPELINE // 5-FOLD STRATIFIED MANIFOLDS',
    hudTagClass: 'hud-tag positive',
    anchorObj: manifoldPlates[2],
  },
  // 4: Anti-Overfit Guard: Elevated orbit surveying generalization envelope
  {
    camera: new THREE.Vector3(1.4, 2.2, 8.6),
    lookAt: new THREE.Vector3(0.0, 0.2, 0.0),
    hudTag: 'OVERFIT GUARD // GENERALIZATION ENVELOPE',
    hudTagClass: 'hud-tag positive',
    anchorObj: radarOuter,
  },
  // 5: Executive Ledger: Square-on desk elevation focusing on sealed dossier
  {
    camera: new THREE.Vector3(1.2, 0.9, 9.4),
    lookAt: new THREE.Vector3(0.0, 0.0, 0.0),
    hudTag: 'VERIFIED SYNTHESIS // SEALED LEDGER',
    hudTagClass: 'hud-tag',
    anchorObj: seal,
  },
];

// Precompute Catmull-Rom Smooth Motion Curves
const camCurve = new THREE.CatmullRomCurve3(
  WAYPOINTS.map((w) => w.camera),
  false,
  'catmullrom',
  0.5
);

const lookAtCurve = new THREE.CatmullRomCurve3(
  WAYPOINTS.map((w) => w.lookAt),
  false,
  'catmullrom',
  0.5
);

/* -------------------------------------------------------------------------- */
/* SECTION 4.1: THE SINGLE JOURNEY SCALAR s in [0, 5]                         */
/* -------------------------------------------------------------------------- */
let s = 0.0;
let sTarget = 0.0;
let sVelocity = 0.0;
let gestureNudge = 0.0;
let scrollVelocity = 0.0;
let lastS = 0.0;
let activeStageIndex = 0;
// Step 0 weight: 1 = loose cloud (compact mode opens here), 0 = the journey formation
let introTarget = HAS_INTRO ? 1.0 : 0.0;
let introWeight = introTarget;
let introEaseNow = introWeight;
const INTRO_HUD_TAG = 'RAW ROWS // NOT YET ANALYZED';
// Step 0 lifts the cloud by this fraction of the viewport so the title can sit beneath it
const INTRO_VIEW_LIFT = 0.12;
let viewOffsetBaseX = 0;

// Reusable calculation vectors to guarantee ZERO per-frame heap allocations (Pillar 7)
const vCamPos = new THREE.Vector3();
const vLookAt = new THREE.Vector3();
const dummyObj = new THREE.Object3D();
const tempInstColor = new THREE.Color();

// Wheel & Touch In-Flight Gesture Nudge (§4.1)
window.addEventListener('wheel', (e) => {
  if (prefersReducedMotion) return;
  const delta = Math.max(-1, Math.min(1, e.deltaY / 100)) * 0.15;
  gestureNudge = Math.max(-0.45, Math.min(0.45, gestureNudge + delta));
}, { passive: true });

let touchStartY = 0;
window.addEventListener('touchstart', (e) => {
  if (e.touches.length > 0) touchStartY = e.touches[0].clientY;
}, { passive: true });

window.addEventListener('touchmove', (e) => {
  if (prefersReducedMotion || e.touches.length === 0) return;
  const delta = (touchStartY - e.touches[0].clientY) * 0.002;
  gestureNudge = Math.max(-0.45, Math.min(0.45, gestureNudge + delta));
  touchStartY = e.touches[0].clientY;
}, { passive: true });

/* -------------------------------------------------------------------------- */
/* SECTION 4.4 & PILLAR 4: METAMASK-STYLE CURSOR PHYSICS (Frame-rate indep)   */
/* -------------------------------------------------------------------------- */
class MouseTrackingPhysics {
  constructor() {
    this.target = { x: 0, y: 0 };
    this.current = { x: 0, y: 0 };
    this.bounds = { yaw: 0.28, pitch: 0.20 }; // Strict angular bounds
    this.mouseNDC = new THREE.Vector2(0, 0);

    window.addEventListener('mousemove', (e) => {
      const rawX = (e.clientX / window.innerWidth) * 2 - 1;
      const rawY = -(e.clientY / window.innerHeight) * 2 + 1;
      this.mouseNDC.set(rawX, rawY);
      this.target.x = Math.max(-1, Math.min(1, rawX)) * this.bounds.yaw;
      this.target.y = Math.max(-1, Math.min(1, rawY)) * this.bounds.pitch;
    });

    document.addEventListener('mouseleave', () => {
      this.target.x = 0;
      this.target.y = 0;
      this.mouseNDC.set(0, 0);
    });
  }

  update(dt) {
    // Frame-rate independent exponential damping: 1 - exp(-lambda * dt) (Defect 6 Fix)
    const k = 1.0 - Math.exp(-MOTION.damp.cursor * dt);
    this.current.x += (this.target.x - this.current.x) * k;
    this.current.y += (this.target.y - this.current.y) * k;
    return this.current;
  }
}

const mousePhysics = new MouseTrackingPhysics();

/* -------------------------------------------------------------------------- */
/* PARTICLE INSPECTION: hover lens, left-click ripple, right-click gather     */
/* -------------------------------------------------------------------------- */
// Hover: particles near the cursor's line of sight swell and brighten in place (a loupe over
// the data, so the formation stays readable). Left-click: a shockwave ring travels outward
// through the formation. Right-click: particles within reach are pulled into a knot at the
// cursor, held, then released back into formation. All effects are in parallaxGroup space.
const LENS = { radius: 0.85, swell: 1.6, glow: 0.6, damp: 8.0 };
const RIPPLE = { speed: 3.6, width: 0.42, amp: 0.38, pop: 0.6, life: 1.5, max: 4 };
const GATHER = { radius: 1.7, rise: 0.35, hold: 0.45, release: 0.9, knot: 0.16 };

const cursorNDC = new THREE.Vector2();
const cursorRaycaster = new THREE.Raycaster();
const cursorPlane = new THREE.Plane();
const cursorWorld = new THREE.Vector3();
const cursorLocal = new THREE.Vector3();
const lensOrigin = new THREE.Vector3();
const lensDir = new THREE.Vector3();
const camForward = new THREE.Vector3();
const sceneOrigin = new THREE.Vector3(0, 0, 0);
const cLensGlow = new THREE.Color(isNight ? 0xffe2b8 : pal.pen);
let cursorInside = false;
let lensStrength = 0;
const ripples = [];
const gatherState = { active: false, x: 0, y: 0, z: 0, t0: 0 };

function isUiTarget(target) {
  return !!(target && target.closest &&
    target.closest('button, a, input, .glass-card, #global-header, #fp-nav, .datum-strip, .scroll-hint'));
}

// Intersect the cursor ray with the camera-facing plane through the subject; result in local space
function resolveCursorLocal() {
  cursorRaycaster.setFromCamera(cursorNDC, camera);
  camera.getWorldDirection(camForward);
  cursorPlane.setFromNormalAndCoplanarPoint(camForward, sceneOrigin);
  if (!cursorRaycaster.ray.intersectPlane(cursorPlane, cursorWorld)) return false;
  cursorLocal.copy(cursorWorld);
  parallaxGroup.worldToLocal(cursorLocal);
  return true;
}

window.addEventListener('pointermove', (e) => {
  cursorNDC.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  cursorInside = e.pointerType === 'mouse' && !isUiTarget(e.target);
}, { passive: true });

document.addEventListener('mouseleave', () => {
  cursorInside = false;
});

window.addEventListener('pointerdown', (e) => {
  if (isUiTarget(e.target) || prefersReducedMotion) return;
  cursorNDC.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  if (!resolveCursorLocal()) return;
  const now = performance.now() * 0.001;
  if (e.button === 0) {
    if (ripples.length >= RIPPLE.max) ripples.shift();
    ripples.push({ x: cursorLocal.x, y: cursorLocal.y, z: cursorLocal.z, t0: now });
  } else if (e.button === 2) {
    gatherState.active = true;
    gatherState.x = cursorLocal.x;
    gatherState.y = cursorLocal.y;
    gatherState.z = cursorLocal.z;
    gatherState.t0 = now;
  }
});

// Right-click belongs to the gather gesture over the scene; menus still work on UI
window.addEventListener('contextmenu', (e) => {
  if (!isUiTarget(e.target)) e.preventDefault();
});

// 0..1 envelope: rise, hold, release
function gatherEnvelope(now) {
  if (!gatherState.active) return 0;
  const age = now - gatherState.t0;
  if (age < GATHER.rise) return smoothstep(age / GATHER.rise);
  if (age < GATHER.rise + GATHER.hold) return 1;
  const rel = (age - GATHER.rise - GATHER.hold) / GATHER.release;
  if (rel >= 1) {
    gatherState.active = false;
    return 0;
  }
  return 1 - smoothstep(rel);
}

/* -------------------------------------------------------------------------- */
/* PILLAR 5: DYNAMIC 3D-TO-2D VECTOR LEADER LINES & PROJECTED HUD PINS       */
/* -------------------------------------------------------------------------- */
const hudPinActive = document.getElementById('hud-pin-active');
const hudTagActive = document.getElementById('hud-tag-active');
const hudTextActive = document.getElementById('hud-text-active');
const leaderCanvas = document.getElementById('leader-lines-canvas');
const leaderCtx = leaderCanvas ? leaderCanvas.getContext('2d') : null;
const tempWorldVec = new THREE.Vector3();

// Cached active card bounding box (Defect 9 Fix: Dynamic elbow target)
let cachedCardRight = 540;
function updateCachedCardBounds() {
  const activeSection = document.querySelector('.section.active') || document.querySelectorAll('.section')[activeStageIndex + SECTION_OFFSET];
  if (activeSection) {
    const wrapper = activeSection.querySelector('.section-content-wrapper');
    if (wrapper) {
      const r = wrapper.getBoundingClientRect();
      if (r && r.right > 0) {
        cachedCardRight = r.right;
        return;
      }
    }
  }
  cachedCardRight = Math.max(540, window.innerWidth * 0.44);
}

function resizeLeaderCanvas() {
  if (!leaderCanvas || !leaderCtx) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2.0);
  leaderCanvas.width = window.innerWidth * dpr;
  leaderCanvas.height = window.innerHeight * dpr;
  leaderCtx.scale(dpr, dpr);
  updateCachedCardBounds();
}
resizeLeaderCanvas();
window.addEventListener('resize', resizeLeaderCanvas);

function updateProjectedHudAndLeaderLines() {
  if (!hudPinActive) return;

  // Step 0 keeps the stage clear: no floating tag or leader line over the unformed cloud
  if (introEaseNow > 0.35) {
    hudPinActive.classList.remove('visible');
    if (leaderCtx) leaderCtx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    return;
  }
  const currentWp = WAYPOINTS[activeStageIndex];
  const targetObj = currentWp?.anchorObj || rlmCore;
  if (!targetObj) return;

  targetObj.getWorldPosition(tempWorldVec);
  tempWorldVec.y += 1.35;
  tempWorldVec.project(camera);

  // Behind camera check
  if (tempWorldVec.z > 1.0) {
    hudPinActive.classList.remove('visible');
    if (leaderCtx) leaderCtx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    return;
  }

  const screenX = (tempWorldVec.x * 0.5 + 0.5) * window.innerWidth;
  const screenY = (-(tempWorldVec.y * 0.5) + 0.5) * window.innerHeight;

  // Strict right-pane clamping: Keep pin badge strictly to the right of text cards
  const minPinX = Math.max(cachedCardRight + 120, window.innerWidth * 0.52);
  const pinnedX = Math.min(Math.max(screenX, minPinX), window.innerWidth - 180);
  const pinnedY = Math.max(screenY, 86);

  // STRICT COMPOSITOR TRANSFORM (Pillar 7: Zero Reflow, never touch top/left)
  hudPinActive.style.transform = `translate3d(${pinnedX.toFixed(1)}px, ${pinnedY.toFixed(1)}px, 0) translate(-50%, -100%)`;
  hudPinActive.classList.add('visible');

  // Draw Angled Elbow Leader Lines linking 3D Anchor to HUD Badge
  if (leaderCtx) {
    leaderCtx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    leaderCtx.lineWidth = 1.2;
    leaderCtx.strokeStyle = isNight ? 'rgba(240, 162, 74, 0.45)' : 'rgba(163, 79, 32, 0.35)';
    leaderCtx.fillStyle = isNight ? '#f0a24a' : '#a34f20';

    // Origin Dot at 3D anchor projection
    leaderCtx.beginPath();
    leaderCtx.arc(screenX, screenY, 3.5, 0, Math.PI * 2);
    leaderCtx.fill();

    // Clean leader bracket to HUD pin badge
    const badgeBottomY = pinnedY;
    const elbowY = (screenY + badgeBottomY) * 0.5;
    leaderCtx.beginPath();
    leaderCtx.moveTo(screenX, screenY);
    leaderCtx.lineTo(pinnedX, elbowY);
    leaderCtx.lineTo(pinnedX, badgeBottomY + 4);
    leaderCtx.stroke();

    // Termination dot under badge
    leaderCtx.beginPath();
    leaderCtx.arc(pinnedX, badgeBottomY + 4, 2.5, 0, Math.PI * 2);
    leaderCtx.fill();
  }
}

/* -------------------------------------------------------------------------- */
/* SECTION 5: ANIME.JS V4 DOM CHOREOGRAPHY & HUD TRANSITIONS                 */
/* -------------------------------------------------------------------------- */
let initialEntryAnimated = false;
function animateSectionEntry(stageIdx) {
  // Only animate on initial page reveal (stage 0) to avoid jarring double-render / flicker on scroll
  if (initialEntryAnimated && stageIdx !== 0) return;
  initialEntryAnimated = true;

  const sections = document.querySelectorAll('.section');
  const targetSection = sections[stageIdx + SECTION_OFFSET];
  if (!targetSection) return;

  // Animate the glass cards as cohesive units (never animate children separately to avoid double-stagger jitter)
  const animTargets = targetSection.querySelectorAll('.glass-card');

  if (animTargets.length > 0 && typeof animate === 'function') {
    animate(animTargets, {
      opacity: [0, 1],
      translateY: [16, 0],
      delay: stagger(60, { start: 40 }),
      duration: MOTION.dur.ui,
      ease: 'outCubic',
    });
  }

  // Animated Number Counters on Initial Section Entry
  animateNumberCounters(stageIdx);
}

let countersAnimated = false;
function animateNumberCounters(stageIdx) {
  if (typeof animate !== 'function' || countersAnimated) return;
  countersAnimated = true;

  if (stageIdx === 0) {
    const el = document.getElementById('stat-row-count');
    const targetVal = RAW_STATE.dataset?.row_count || 1420;
    if (el) {
      const obj = { count: 0 };
      animate(obj, {
        count: targetVal,
        duration: MOTION.dur.stage,
        ease: 'outCubic',
        onUpdate: () => {
          el.textContent = Math.round(obj.count).toLocaleString();
        },
      });
    }
  }
}

function transitionToSection(targetIndex) {
  if (targetIndex < 0 || targetIndex >= WAYPOINTS.length) return;
  activeStageIndex = targetIndex;
  sTarget = targetIndex;
  introTarget = 0.0;
  updateCachedCardBounds();

  const wp = WAYPOINTS[targetIndex];
  setHudTag(wp.hudTag, wp.hudTagClass);
  syncQuickPills(targetIndex);
}

// Step 0 (compact mode): the cloud before stage 0 assembles; no quick pill is active
function enterIntro() {
  activeStageIndex = 0;
  sTarget = 0;
  introTarget = 1.0;
  updateCachedCardBounds();
  setHudTag(INTRO_HUD_TAG, 'hud-tag');
  syncQuickPills(-1);
}

function setHudTag(text, className) {
  // Animated HUD Tag text transition
  if (hudTextActive && hudTextActive.textContent !== text) {
    // Flash the inner tag, not the pin: a tween leaves inline opacity behind, and on the pin
    // that would override the `.visible` class that hides it (step 0, behind-camera)
    if (typeof animate === 'function' && hudTagActive) {
      animate(hudTagActive, {
        opacity: [1, 0.3, 1],
        scale: [1, 0.96, 1],
        duration: MOTION.dur.micro,
        ease: 'outQuad',
      });
    }
    hudTextActive.textContent = text;
  }
  if (hudTagActive) hudTagActive.className = className;
}

function syncQuickPills(stageIdx) {
  document.querySelectorAll('.pill-btn').forEach((btn, idx) => {
    btn.classList.toggle('active', idx === stageIdx);
  });
}

/* -------------------------------------------------------------------------- */
/* FULLPAGE.JS SNAP SCROLLING                                                 */
/* -------------------------------------------------------------------------- */
try {
  if (window.fullpage) {
    new fullpage('#fullpage', {
      licenseKey: 'gplv3-license',
      css3: true,
      scrollingSpeed: MOTION.dur.stage, // Unified duration token (Defect 12 Fix)
      easingcss3: 'cubic-bezier(0.22, 1, 0.36, 1)',
      autoScrolling: true,
      fitToSection: true,
      fitToSectionDelay: 100,
      scrollBar: false,
      navigation: true,
      navigationPosition: 'right',
      navigationTooltips: (HAS_INTRO ? ['00 Start'] : []).concat([
        '01 Overview & RLM',
        '02 Ingestion & Profiling',
        '03 Statistical Testing',
        '04 ML Pipeline & Folds',
        '05 Overfit Guard',
        '06 Executive Ledger',
      ]),
      // A permanently shown tooltip collides with the HUD tag in the small hero box
      showActiveTooltip: !HAS_INTRO,
      anchors: (HAS_INTRO ? ['intro'] : []).concat(['overview', 'ingestion', 'stats', 'pipeline', 'diagnostics', 'report']),
      touchSensitivity: 8,
      normalScrollElements: '.mini-table, .glass-card, #correlations-list',
      onLeave: (origin, destination) => {
        const stageIdx = destination.index - SECTION_OFFSET;
        if (stageIdx < 0) {
          enterIntro();
        } else {
          transitionToSection(stageIdx);
        }
      },
      afterLoad: (origin, destination) => {
        // Section landing: update cached card bounds for 3D parallax & HUD projection
        // Do NOT re-animate already-slid content on every arrival (prevents double refresh)
        updateCachedCardBounds();
      },
    });
  }
} catch (err) {
  console.warn('fullpage.js initialized with fallback:', err);
}

/* -------------------------------------------------------------------------- */
/* LIFECYCLE MANAGEMENT (Locked 60fps across Sandboxed Iframes & Standalone)  */
/* -------------------------------------------------------------------------- */
let isAppVisible = true;
document.addEventListener('visibilitychange', () => {
  isAppVisible = !document.hidden;
  if (!isAppVisible && tourInterval) {
    clearInterval(tourInterval);
    tourInterval = null;
    isTourActive = false;
    if (btnCameraView) {
      btnCameraView.style.borderColor = 'var(--card-border)';
      btnCameraView.style.boxShadow = 'none';
    }
  }
});

/* -------------------------------------------------------------------------- */
/* PILLAR 1: SINGLE UNIFIED RENDER LOOP (Decoupled Engine Stepper)            */
/* -------------------------------------------------------------------------- */
let lastFrameTime = performance.now();
let laserDir = 1;
let laserY = 0;

renderer.setAnimationLoop((timestamp) => {
  if (!isAppVisible) return; // Halt loop when tab is hidden or minimized

  const delta = Math.min((timestamp - lastFrameTime) * 0.001, 0.05);
  lastFrameTime = timestamp;
  const elapsed = timestamp * 0.001;

  // 1. Advance Anime.js v4 engine synchronously (Pillar 1: Single Source of Time)
  if (engine && typeof engine.update === 'function') {
    engine.update(timestamp);
  }

  // 2. Critically Damped Spring Integration on s (§4.1)
  gestureNudge *= Math.exp(-MOTION.damp.velocity * delta);

  if (prefersReducedMotion) {
    s = sTarget;
    sVelocity = 0.0;
  } else {
    const targetWithNudge = Math.max(0, Math.min(5, sTarget + gestureNudge));
    const deltaS = targetWithNudge - s;
    const omega = MOTION.damp.journey;
    const springForce = omega * omega * deltaS - 2.0 * omega * sVelocity;
    sVelocity += springForce * delta;
    s += sVelocity * delta;
  }
  s = Math.max(0, Math.min(5, s));

  // Step 0 cloud weight (compact mode): assembles as the viewer scrolls into stage 0
  introWeight = prefersReducedMotion
    ? introTarget
    : introWeight + (introTarget - introWeight) * (1.0 - Math.exp(-3.2 * delta));
  const introEase = smoothstep(introWeight);
  introEaseNow = introEase;

  // Scroll Velocity Dynamics (§4.4)
  const instantVel = Math.abs(s - lastS) / Math.max(delta, 0.001);
  lastS = s;
  scrollVelocity += (instantVel - scrollVelocity) * (1.0 - Math.exp(-MOTION.damp.velocity * delta));

  // Dynamic Camera FOV Widening during fast scroll
  const fovWiden = prefersReducedMotion ? 0 : Math.min(scrollVelocity * 0.65, 3.0);
  camera.fov = BASE_FOV + fovWiden;
  if (HAS_INTRO && camera.view) {
    // Step 0 frames the cloud centered and lifted above the stacked title; it glides back
    // to the right-pane framing as the cloud assembles into stage 0
    camera.view.offsetX = viewOffsetBaseX * (1.0 - introEase);
    camera.view.offsetY = window.innerHeight * INTRO_VIEW_LIFT * introEase;
  }
  camera.updateProjectionMatrix();

  // Derive Journey Stage Indices & Fractional Progress
  const stageA = Math.min(Math.max(Math.floor(s), 0), 4);
  const stageB = stageA + 1;
  const localProg = Math.min(Math.max(s - stageA, 0), 1);
  const t = smoothstep(localProg);

  // 3. Evaluate Continuous Camera Spline (§4.2)
  const normS = Math.min(Math.max(s / 5.0, 0), 1);
  camCurve.getPoint(normS, vCamPos);
  lookAtCurve.getPoint(normS, vLookAt);

  // Apply Cursor-Driven Truck/Dolly Parallax Offset after Spline
  const mouseRot = mousePhysics.update(delta);
  vCamPos.x += mouseRot.x * 0.35;
  vCamPos.y += mouseRot.y * 0.25;
  // Step 0 frames the whole cloud from a little further back
  vCamPos.z += 2.2 * introEase;
  vCamPos.y += 0.3 * introEase;

  camera.position.copy(vCamPos);
  camera.lookAt(vLookAt);

  // Tilt dedicated ParallaxGroup for model perspective
  parallaxGroup.rotation.y = mouseRot.x * 0.65;
  parallaxGroup.rotation.x = -mouseRot.y * 0.65;

  // Inspection lens ray in parallaxGroup space (matrices refreshed so the ray matches this frame)
  lensStrength += ((cursorInside ? 1 : 0) - lensStrength) * (1.0 - Math.exp(-LENS.damp * delta));
  let lensActive = false;
  if (lensStrength > 0.001) {
    camera.updateMatrixWorld();
    parallaxGroup.updateMatrixWorld();
    if (resolveCursorLocal()) {
      lensOrigin.copy(camera.position);
      parallaxGroup.worldToLocal(lensOrigin);
      lensDir.copy(cursorLocal).sub(lensOrigin).normalize();
      lensActive = true;
    }
  }
  const nowSec = timestamp * 0.001;
  const gatherEnv = gatherEnvelope(nowSec);
  while (ripples.length && nowSec - ripples[0].t0 > RIPPLE.life) ripples.shift();
  const lensRadiusSq = LENS.radius * LENS.radius;

  // 4. Update Continuous Matter Field Morph (§4.3)
  const bufA = stageBuffers[stageA];
  const bufB = stageBuffers[stageB];
  const colA = stageColors[stageA];
  const colB = stageColors[stageB];
  const arcBoost = prefersReducedMotion ? 0 : 1.0 + Math.min(scrollVelocity * 0.4, 1.2);

  for (let i = 0; i < MATTER_COUNT; i++) {
    const i3 = i * 3;

    // Per-instance stagger
    const staggerFactor = (i % 48) / 48.0 * 0.15;
    const instT = smoothstep(Math.min(Math.max((t - staggerFactor * 0.5) / (1.0 - staggerFactor * 0.5), 0), 1));

    // Base linear lerp
    let x = bufA[i3] + (bufB[i3] - bufA[i3]) * instT;
    let y = bufA[i3 + 1] + (bufB[i3 + 1] - bufA[i3 + 1]) * instT;
    let z = bufA[i3 + 2] + (bufB[i3 + 2] - bufA[i3 + 2]) * instT;

    // Living Arc Lift: Parabolic swing peaking at midpoint
    const arc = 4.0 * instT * (1.0 - instT) * arcBoost;
    if (arc > 0.001) {
      const radialSeed = i * 1.37;
      y += (Math.sin(radialSeed) * 0.35 + 0.45) * arc;
      z += Math.cos(radialSeed) * 0.35 * arc;
    }

    // Micro-scale breathing: Unpack & repack
    let scale = 1.0 - 0.32 * arc;

    // Color Interpolation
    let cr = colA[i3] + (colB[i3] - colA[i3]) * instT;
    let cg = colA[i3 + 1] + (colB[i3 + 1] - colA[i3 + 1]) * instT;
    let cb = colA[i3 + 2] + (colB[i3 + 2] - colA[i3 + 2]) * instT;

    // Step 0: blend toward the slowly drifting raw cloud
    if (introEase > 0.001) {
      const drift = prefersReducedMotion ? 0 : 0.14;
      const cx = cloudBuffer[i3] + Math.sin(elapsed * 0.31 + i * 0.71) * drift;
      const cy = cloudBuffer[i3 + 1] + Math.sin(elapsed * 0.27 + i * 1.13) * drift;
      const cz = cloudBuffer[i3 + 2] + Math.cos(elapsed * 0.23 + i * 0.53) * drift;
      x += (cx - x) * introEase;
      y += (cy - y) * introEase;
      z += (cz - z) * introEase;
      scale += (0.85 - scale) * introEase;
      cr += (cloudColors[i3] - cr) * introEase;
      cg += (cloudColors[i3 + 1] - cg) * introEase;
      cb += (cloudColors[i3 + 2] - cb) * introEase;
    }

    // Right-click gather: pull particles within reach into a small knot at the cursor
    if (gatherEnv > 0.001) {
      const gx = x - gatherState.x;
      const gy = y - gatherState.y;
      const gz = z - gatherState.z;
      const gd = Math.sqrt(gx * gx + gy * gy + gz * gz);
      if (gd < GATHER.radius) {
        const f = smoothstep(1.0 - gd / GATHER.radius) * gatherEnv;
        x += (gatherState.x + Math.sin(i * 12.9898) * GATHER.knot - x) * f;
        y += (gatherState.y + Math.sin(i * 78.233) * GATHER.knot - y) * f;
        z += (gatherState.z + Math.sin(i * 37.719) * GATHER.knot - z) * f;
      }
    }

    // Left-click ripples: a shockwave band pushes particles outward as it passes, then fades
    for (let rIdx = 0; rIdx < ripples.length; rIdx++) {
      const rp = ripples[rIdx];
      const age = nowSec - rp.t0;
      const rx = x - rp.x;
      const ry = y - rp.y;
      const rz = z - rp.z;
      const rd = Math.sqrt(rx * rx + ry * ry + rz * rz);
      const offset = (rd - age * RIPPLE.speed) / RIPPLE.width;
      if (rd > 1e-4 && offset > -3 && offset < 3) {
        const band = Math.exp(-offset * offset) * (1.0 - age / RIPPLE.life);
        const push = RIPPLE.amp * band / rd;
        x += rx * push;
        y += ry * push;
        z += rz * push;
        scale *= 1.0 + RIPPLE.pop * band;
      }
    }

    // Hover lens: swell and brighten by distance from the cursor's line of sight (no displacement)
    if (lensActive) {
      const px = x - lensOrigin.x;
      const py = y - lensOrigin.y;
      const pz = z - lensOrigin.z;
      const qx = py * lensDir.z - pz * lensDir.y;
      const qy = pz * lensDir.x - px * lensDir.z;
      const qz = px * lensDir.y - py * lensDir.x;
      const distSq = qx * qx + qy * qy + qz * qz;
      if (distSq < lensRadiusSq) {
        const f = 1.0 - Math.sqrt(distSq) / LENS.radius;
        const w = f * f * lensStrength;
        scale *= 1.0 + LENS.swell * w;
        const g = w * LENS.glow;
        cr += (cLensGlow.r - cr) * g;
        cg += (cLensGlow.g - cg) * g;
        cb += (cLensGlow.b - cb) * g;
      }
    }

    dummyObj.position.set(x, y, z);
    dummyObj.scale.setScalar(scale);
    dummyObj.updateMatrix();
    matterMesh.setMatrixAt(i, dummyObj.matrix);

    tempInstColor.setRGB(cr, cg, cb);
    matterMesh.setColorAt(i, tempInstColor);
  }
  matterMesh.instanceMatrix.needsUpdate = true;
  if (matterMesh.instanceColor) matterMesh.instanceColor.needsUpdate = true;

  // 5. Update Continuous Link Field Morph (§4.3)
  const linkPosAttr = linkGeo.attributes.position;
  const linkColAttr = linkGeo.attributes.color;
  const pairsA = linkPairsMap[stageA];
  const pairsB = linkPairsMap[stageB];

  for (let k = 0; k < LINK_PAIRS; k++) {
    const k6 = k * 6;
    const [pA1, pB1] = pairsA[k];
    const [pA2, pB2] = pairsB[k];

    // Lerp start point
    const s1A = pA1 * 3;
    const s2A = pA2 * 3;
    const ax = (bufA[s1A] + (bufB[s1A] - bufA[s1A]) * t) * (1 - t) + (bufA[s2A] + (bufB[s2A] - bufA[s2A]) * t) * t;
    const ay = (bufA[s1A + 1] + (bufB[s1A + 1] - bufA[s1A + 1]) * t) * (1 - t) + (bufA[s2A + 1] + (bufB[s2A + 1] - bufA[s2A + 1]) * t) * t;
    const az = (bufA[s1A + 2] + (bufB[s1A + 2] - bufA[s1A + 2]) * t) * (1 - t) + (bufA[s2A + 2] + (bufB[s2A + 2] - bufA[s2A + 2]) * t) * t;

    // Lerp end point
    const s1B = pB1 * 3;
    const s2B = pB2 * 3;
    const bx = (bufA[s1B] + (bufB[s1B] - bufA[s1B]) * t) * (1 - t) + (bufA[s2B] + (bufB[s2B] - bufA[s2B]) * t) * t;
    const by = (bufA[s1B + 1] + (bufB[s1B + 1] - bufA[s1B + 1]) * t) * (1 - t) + (bufA[s2B + 1] + (bufB[s2B + 1] - bufA[s2B + 1]) * t) * t;
    const bz = (bufA[s1B + 2] + (bufB[s1B + 2] - bufA[s1B + 2]) * t) * (1 - t) + (bufA[s2B + 2] + (bufB[s2B + 2] - bufA[s2B + 2]) * t) * t;

    linkPositions[k6] = ax;
    linkPositions[k6 + 1] = ay;
    linkPositions[k6 + 2] = az;
    linkPositions[k6 + 3] = bx;
    linkPositions[k6 + 4] = by;
    linkPositions[k6 + 5] = bz;

    const linkAlpha = 0.65 * (1.0 - Math.abs(t - 0.5) * 0.4);
    linkColors[k6] = cPen.r * linkAlpha;
    linkColors[k6 + 1] = cPen.g * linkAlpha;
    linkColors[k6 + 2] = cPen.b * linkAlpha;
    linkColors[k6 + 3] = cAccent.r * linkAlpha;
    linkColors[k6 + 4] = cAccent.g * linkAlpha;
    linkColors[k6 + 5] = cAccent.b * linkAlpha;
  }
  linkPosAttr.needsUpdate = true;
  linkColAttr.needsUpdate = true;
  // Links describe a formation; the raw step 0 cloud has none
  linkMat.opacity = 0.72 * (1.0 - introEase);
  linkSegments.visible = introEase < 0.99;

  // 6. Solid Hero Meshes Continuous Cross-Fading (§4.3)
  stageHeroGroups.forEach((grp, idx) => {
    const dist = Math.abs(s - idx);
    // Smooth transition: fully solid within 0.35 of target stage, smoothly fading between 0.35 and 0.90
    const heroAlpha = (dist <= 0.35 ? 1.0 : (dist >= 0.90 ? 0.0 : smoothstep(1.0 - (dist - 0.35) / 0.55)))
      * (1.0 - introEase); // solid meshes appear only once the step 0 cloud has assembled

    if (heroAlpha <= 0.001) {
      grp.visible = false;
    } else {
      grp.visible = true;
      grp.scale.setScalar(0.92 + 0.08 * heroAlpha);
      grp.traverse((child) => {
        if (child.material) {
          child.material.transparent = true;
          const base = child.userData.baseOpacity !== undefined ? child.userData.baseOpacity : 0.85;
          child.material.opacity = Math.max(0.18, Math.min(1.0, base * heroAlpha));
        }
      });
    }
  });

  // 7. Kinetic Mechanics for Active Components
  if (rlmGroup.visible && !prefersReducedMotion) {
    rlmCore.rotation.y += delta * 0.45;
    rlmCore.rotation.x += delta * 0.22;
    ring1.rotation.z += delta * 0.55;
    ring2.rotation.x += delta * 0.35;
    ring3.rotation.y += delta * 0.25;
  }

  if (ingestionGroup.visible && !prefersReducedMotion) {
    laserY += delta * 2.2 * laserDir;
    if (laserY > 2.5) { laserY = 2.5; laserDir = -1; }
    if (laserY < -2.5) { laserY = -2.5; laserDir = 1; }
    laserPlane.position.y = laserY;
  }

  if (pipelineGroup.visible && !prefersReducedMotion) {
    // Manifold oscillation from base positions (Defect 7 Fix: No integration runaway)
    manifoldPlates.forEach((plate, i) => {
      plate.position.y = manifoldBaseY[i] + Math.sin(elapsed * 1.8 + i) * 0.06;
      plate.rotation.z = manifoldBaseRotZ[i] + Math.cos(elapsed * 1.2 + i) * 0.04;
    });
  }

  if (diagnosticsGroup.visible && !prefersReducedMotion) {
    radarOuter.rotation.y += delta * 0.45;
    radarOuter.rotation.x += delta * 0.22;
    radarInner.rotation.y -= delta * 0.35;
    guardRing.rotation.z += delta * 0.3;
  }

  if (reportGroup.visible && !prefersReducedMotion) {
    easel.position.y = Math.sin(elapsed * 1.6) * 0.06;
    seal.rotation.z += delta * 0.35;
  }

  // 8. Projected 3D-to-2D HUD Pin & Dynamic Leader Lines (Pillar 5)
  updateProjectedHudAndLeaderLines();

  // 9. Render Scene to WebGL Canvas
  renderer.render(scene, camera);
});

/* -------------------------------------------------------------------------- */
/* WINDOW RESIZE & CAMERA VIEWPORT PROJECTION                                 */
/* -------------------------------------------------------------------------- */
function updateCameraProjection() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  // Shift optical center so the 3D focal subject appears centered in the right pane (X ~ 70%).
  // The compact hero box is narrower than 768px but still has a text column, so keep the shift there.
  viewOffsetBaseX = w > 768 || HAS_INTRO ? -w * 0.22 : 0;
  camera.setViewOffset(w, h, viewOffsetBaseX, 0, w, h);
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2.0));
  resizeLeaderCanvas();
}
window.addEventListener('resize', updateCameraProjection);
updateCameraProjection();

/* -------------------------------------------------------------------------- */
/* PILLAR 6: MANSORY ZERO-FREEZE PRELOADER & SHADER WARM-UP                  */
/* -------------------------------------------------------------------------- */
function warmUpShadersAndReveal() {
  const preloaderBar = document.getElementById('preloader-bar');
  const preloaderStatus = document.getElementById('preloader-status');
  const preloaderCurtain = document.getElementById('preloader-curtain');

  // Step 1: Pre-compile all scene shaders with renderer.compile
  try {
    renderer.compile(scene, camera);
    if (preloaderBar) preloaderBar.style.width = '65%';
    if (preloaderStatus) preloaderStatus.textContent = 'Pre-Warming Shaders… 65%';
  } catch (err) {
    console.warn('Shader pre-compilation warning:', err);
  }

  // Step 2: Ensure all web fonts are loaded
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => {
      finishPreloader();
    }).catch(() => finishPreloader());
  } else {
    setTimeout(finishPreloader, 350);
  }

  function finishPreloader() {
    if (preloaderBar) {
      if (typeof animate === 'function') {
        animate(preloaderBar, {
          width: '100%',
          duration: 380,
          ease: MOTION.ease.out,
        });
      } else {
        preloaderBar.style.width = '100%';
      }
    }
    if (preloaderStatus) preloaderStatus.textContent = 'Engine Ready // 100%';

    setTimeout(() => {
      if (preloaderCurtain) {
        preloaderCurtain.classList.add('loaded');
      }
      animateSectionEntry(0);
    }, 450);
  }
}

warmUpShadersAndReveal();

/* -------------------------------------------------------------------------- */
/* CENTRALIZED PALETTE APPLICATION & THEME TOGGLE (Defect 8 Fix)              */
/* -------------------------------------------------------------------------- */
export function applyThemePalette(themeName) {
  isNight = themeName === 'night';
  THEME = isNight ? 'night' : 'day';
  const p = getActivePalette();
  document.documentElement.classList.toggle('theme-day', !isNight);
  document.body.classList.toggle('theme-day', !isNight);

  document.documentElement.style.setProperty('--stock', p.stock);
  document.documentElement.style.setProperty('--sheet', p.sheet);
  document.documentElement.style.setProperty('--sheet-alt', isNight ? '#282017' : '#f1e4cb');
  document.documentElement.style.setProperty('--ink', isNight ? '#f6eedf' : '#3a2b1e');
  document.documentElement.style.setProperty('--graphite', p.graphite);
  document.documentElement.style.setProperty('--pen', p.pen);
  document.documentElement.style.setProperty('--accent', p.accent);
  document.documentElement.style.setProperty('--risk', p.risk);
  document.documentElement.style.setProperty('--positive', p.positive);
  document.documentElement.style.setProperty('--grid', p.grid);
  document.documentElement.style.setProperty('--card-bg', isNight ? 'rgba(28, 22, 16, 0.85)' : 'rgba(255, 251, 242, 0.94)');
  document.documentElement.style.setProperty('--card-border', isNight ? 'rgba(240, 162, 74, 0.22)' : 'rgba(138, 118, 96, 0.28)');

  const stockC = new THREE.Color(p.stock);
  renderer.setClearColor(stockC, 1);
  scene.background = stockC;
  scene.fog.color.copy(stockC);
  scene.fog.density = p.fogDensity;
  renderer.toneMappingExposure = isNight ? 1.15 : 1.0;

  ambientLight.color.setHex(p.ambient);
  ambientLight.intensity = p.ambientInt;
  keyLight.color.setHex(p.key);
  keyLight.intensity = p.keyInt;
  fillLight.color.setHex(p.fill);
  fillLight.intensity = p.fillInt;
  rimLight.color.setHex(p.rim);
  rimLight.intensity = p.rimInt;

  // Re-tint all hero mesh materials
  dodecaMat.color.set(p.pen);
  dodecaMat.emissive.setHex(p.coreEmissive);
  dodecaEdgesMat.color.setHex(isNight ? 0xffe2b8 : 0x5a2d0d);
  ring1Mat.color.set(p.pen);
  ring2Mat.color.set(p.accent);
  ring3Mat.color.set(p.graphite);
  laserMat.color.set(p.accent);
  laserMat.emissive.set(p.accent);
  laserFrameMat.color.set(p.accent);
  manifoldPlates.forEach((pl, i) => {
    pl.material.color.set(i === 2 ? p.pen : p.accent);
  });
  outerMat.color.set(p.positive);
  innerMat.color.set(p.positive);
  guardRingMat.color.set(p.positive);
  easelMat.color.set(isNight ? p.sheet : '#fff5e6');
  easelEdgesMat.color.set(p.pen);
  sealMat.color.set(p.pen);
  sealMat.emissive.set(p.pen);
  spineMat.color.set(p.graphite);

  gridHelper.material.color.set(p.grid);
  gridHelper.material.opacity = p.gridOpacity;

  // Re-tint Matter & Link instance buffers
  cPen.set(p.pen);
  cAccent.set(p.accent);
  cPositive.set(p.positive);
  cGraphite.set(p.graphite);
  cLensGlow.set(isNight ? 0xffe2b8 : p.pen);
}

const btnTheme = document.getElementById('btn-theme-toggle');
if (btnTheme) {
  btnTheme.addEventListener('click', () => {
    applyThemePalette(isNight ? 'day' : 'night');
  });
}

const btnFullscreen = document.getElementById('btn-fullscreen');
if (btnFullscreen) {
  btnFullscreen.addEventListener('click', () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  });
}

const btnCameraView = document.getElementById('btn-camera-view');
let isTourActive = false;
let tourInterval = null;
if (btnCameraView) {
  btnCameraView.addEventListener('click', () => {
    isTourActive = !isTourActive;
    btnCameraView.style.borderColor = isTourActive ? 'var(--pen)' : 'var(--card-border)';
    btnCameraView.style.boxShadow = isTourActive ? '0 0 12px var(--pen-glow)' : 'none';

    if (isTourActive) {
      tourInterval = setInterval(() => {
        if (!isAppVisible) return;
        const nextIdx = (activeStageIndex + 1) % WAYPOINTS.length;
        if (window.fullpage_api) {
          fullpage_api.moveTo(nextIdx + 1 + SECTION_OFFSET);
        } else {
          transitionToSection(nextIdx);
        }
      }, 5000);
    } else if (tourInterval) {
      clearInterval(tourInterval);
      tourInterval = null;
    }
  });
}

/* -------------------------------------------------------------------------- */
/* POPULATE DYNAMIC DATA FROM INJECTED STATE                                  */
/* -------------------------------------------------------------------------- */
function populateDataFromState() {
  if (!RAW_STATE.dataset) return;

  const d = RAW_STATE.dataset;
  const s = RAW_STATE.statistics || {};
  const m = RAW_STATE.ml || {};
  const syn = RAW_STATE.synthesis || {};

  // Overview
  const elRowCount = document.getElementById('stat-row-count');
  if (elRowCount) elRowCount.textContent = (d.row_count || 0).toLocaleString();

  // Ingestion
  const elDsName = document.getElementById('profile-dataset-name');
  if (elDsName) elDsName.textContent = d.name || 'dataset.csv';
  const elSummary = document.getElementById('profile-summary-text');
  if (elSummary) {
    elSummary.textContent = `${(d.row_count || 0).toLocaleString()} rows × ${d.col_count || 0} columns. Target: ${d.target_col || 'N/A'}. Data quality audited at ${d.quality_score || 94}%.`;
  }
  const elQS = document.getElementById('profile-quality-score');
  if (elQS) elQS.textContent = d.quality_score || 94;

  const countNum = document.getElementById('count-numeric');
  if (countNum && d.column_types) countNum.textContent = d.column_types.numeric || 0;
  const countCat = document.getElementById('count-categorical');
  if (countCat && d.column_types) countCat.textContent = d.column_types.categorical || 0;
  const countMiss = document.getElementById('count-missing');
  if (countMiss) countMiss.textContent = d.missing_cells || 0;
  const valTask = document.getElementById('val-task-type');
  if (valTask) valTask.textContent = (d.task_type || 'Classification').toUpperCase();

  // Statistics
  const elTestName = document.getElementById('stat-test-name');
  if (elTestName && s.test_name) elTestName.textContent = s.test_name;
  const elPVal = document.getElementById('stat-p-value');
  if (elPVal && s.p_value !== undefined) {
    elPVal.textContent = typeof s.p_value === 'number' ? (s.p_value < 0.001 ? '< 0.001' : s.p_value.toFixed(4)) : s.p_value;
  }
  const elOutlierTag = document.getElementById('stat-outlier-tag');
  if (elOutlierTag && s.outlier_pct !== undefined) {
    elOutlierTag.textContent = `${s.outlier_pct}% Outliers Handled`;
  }

  // Correlations
  const corrList = document.getElementById('correlations-list');
  if (corrList && Array.isArray(s.top_correlations) && s.top_correlations.length > 0) {
    corrList.innerHTML = s.top_correlations.map((c) => `
      <div style="display: flex; justify-content: space-between; font-size: 13px; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
        <span style="font-family: var(--mono); color: var(--ink);">${c.pair || ''}</span>
        <span style="font-weight: 700; color: ${c.val >= 0 ? 'var(--pen)' : 'var(--risk)'};">${c.val >= 0 ? '+' : ''}${c.val}</span>
      </div>
    `).join('');
  }

  // ML Pipeline
  const elBestModel = document.getElementById('ml-best-name');
  if (elBestModel) elBestModel.textContent = m.best_model || 'Best Model';
  const elBestCv = document.getElementById('ml-best-cv');
  if (elBestCv) elBestCv.textContent = `${m.best_cv || 92.4}%`;

  const tbody = document.getElementById('models-tbody');
  if (tbody && Array.isArray(m.models) && m.models.length > 0) {
    tbody.innerHTML = m.models.map((mod) => {
      const isBest = mod.is_best || mod.name === m.best_model;
      const isWarn = mod.gap > 10;
      return `
        <tr class="${isBest ? 'best-row' : ''}">
          <td>${mod.name} ${isBest ? '★' : ''}</td>
          <td>${mod.cv_mean}%</td>
          <td>±${mod.cv_std || 1.5}%</td>
          <td style="${isWarn ? 'color: var(--risk); font-weight: 700;' : ''}">${mod.gap}% ${isWarn ? '⚠️' : ''}</td>
        </tr>
      `;
    }).join('');
  }

  // Overfit Guard
  const elGapVal = document.getElementById('guard-gap-val');
  if (elGapVal) elGapVal.textContent = `${m.best_gap || 3.2}%`;
  const elGapStatus = document.getElementById('guard-gap-status');
  const isOverfit = (m.best_gap || 0) > 10;
  if (elGapStatus) {
    elGapStatus.textContent = isOverfit ? 'RISK' : 'SAFE';
    elGapStatus.style.color = isOverfit ? 'var(--risk)' : 'var(--positive)';
  }
  const elSafeTag = document.getElementById('guard-safe-tag');
  if (elSafeTag) {
    if (isOverfit) {
      elSafeTag.className = 'pill-tag warn';
      elSafeTag.textContent = '⚠️ GAP > 10% WARNING';
    } else {
      elSafeTag.className = 'pill-tag safe';
      elSafeTag.textContent = '✓ GAP < 10% CERTIFIED';
    }
  }

  // Executive Synthesis
  const elReasoning = document.getElementById('exec-reasoning-text');
  if (elReasoning && syn.reasoning) elReasoning.textContent = syn.reasoning;

  const findingsList = document.getElementById('exec-findings-list');
  if (findingsList && Array.isArray(syn.findings) && syn.findings.length > 0) {
    findingsList.innerHTML = syn.findings.map((f) => `<li>${f}</li>`).join('');
  }

  // Datum Bar
  const datumFile = document.getElementById('datum-file');
  if (datumFile) datumFile.textContent = d.name || 'sample_dataset.csv';
  const datumModel = document.getElementById('datum-model');
  if (datumModel) datumModel.textContent = m.best_model || 'GradientBoosting';
  const datumScore = document.getElementById('datum-score');
  if (datumScore) datumScore.textContent = `${m.best_cv || 92.4}%`;
  const datumGap = document.getElementById('datum-gap');
  if (datumGap) datumGap.textContent = `${m.best_gap || 3.2}%`;
}

populateDataFromState();

// Initialize the opening state: step 0 cloud in compact mode, otherwise section 0
if (HAS_INTRO) {
  enterIntro();
} else {
  transitionToSection(0);
}
