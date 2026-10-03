/**
 * The plate — a warm, friendly 3D view of the seven-step pipeline.
 *
 * - Soft directional lighting and gentle contact shadows for depth.
 * - Fluid data particle stream flowing through active pipeline stages and refinement arcs.
 * - Morphing kinetic modules with stage-specific geometries and mechanical animations.
 * - 3D world-space projected HUD pin callouts.
 * - Interactive perspective toggles (ISO, PLAN, FRONT, RESET).
 * - Full performance gating (dirty flag, visibility check, DPR clamp, teardown).
 */
// Three.js is loaded below (a local copy first, then the CDN), so a failed load cannot kill this script.

const STATE = window.__PIPELINE_STATE__ || { stages: [], palette: {} };
const C = STATE.palette || {};
const STAGES = STATE.stages || [];
const THEME = STATE.theme || "day";
const isNight = THEME === "night";

const host = document.getElementById("scene");
const elTitle = document.getElementById("readout-title");
const elBody = document.getElementById("readout-body");
const hudPin = document.getElementById("hud-pin");
const hudStep = document.getElementById("hud-step");
const hudText = document.getElementById("hud-text");
const hudBadge = document.getElementById("hud-badge");

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ------------------------------------------------------------------ *
 * MOTION CONTRACT (Shared with cinematic_3d)
 * ------------------------------------------------------------------ */
export const MOTION = {
  dur: { micro: 180, ui: 320, stage: 950, cinematic: 1400 },
  ease: { out: 'cubicBezier(0.22, 1, 0.36, 1)', inOut: 'easeInOutCubic', spring: 'spring(1, 80, 10, 0)' },
  damp: { cursor: 3.5, journey: 6.0, velocity: 4.0 },
  stagger: 40,
};

/* Streamlit reruns this iframe's document on *any* widget interaction
 * anywhere in the app, not just after an analysis run advances a stage —
 * moving an unrelated sidebar slider triggers the same full page rerun.
 * build_document() is deterministic for unchanged (stages, theme), so
 * replaying the entrance animation only makes sense when that pair has
 * actually changed since we last played it. sessionStorage survives an
 * iframe document reload within the same tab, so it's the signal: skip
 * the animation and snap straight to final state on a rerun that changed
 * nothing (IMPROVEMENTS.md #10); play it in full when the run progressed.
 */
const RUN_SIGNATURE = JSON.stringify({
  theme: THEME,
  stages: STAGES.map((s) => [s.num, s.status, s.detail]),
});
function hasPlayedEntranceFor(signature) {
  try {
    return sessionStorage.getItem("pipeline3d:lastEntrance") === signature;
  } catch {
    return false; // storage blocked (sandboxed iframe) — always animate
  }
}
function markEntrancePlayed(signature) {
  try {
    sessionStorage.setItem("pipeline3d:lastEntrance", signature);
  } catch {
    /* storage blocked — nothing to persist, next load just animates again */
  }
}

/* ------------------------------------------------------------------ *
 * Drawing geometry & layout constants
 * ------------------------------------------------------------------ */
const GAP = 2.1;               // Spacing between modules along rail
const CAGE = 0.95;              // Wireframe module bounding size
const REFINE_FROM = 4;          // Stage 5 (0-indexed)
const REFINE_TO = 2;            // Loops back to Stage 3
const RLM_INDEX = 5;            // Stage 6 recursive decomposition
const STROKE_SAMPLES = 28;      // Line sample points for pen reveal

const xFor = (i) => (i - (STAGES.length - 1) / 2) * GAP;
const isReached = (s) => s === "done" || s === "active" || s === "error";

function inkFor(status) {
  if (status === "error") return C.risk || "#a33526";
  if (status === "done" || status === "active") return C.pen || "#a34f20";
  return C.graphite || "#8a7660";
}

const CAGE_OPACITY = {
  pending: 0.35,
  skipped: 0.18,
  done: 0.85,
  active: 1.0,
  error: 0.95,
};

const FILL_OPACITY = {
  pending: 0.15,
  skipped: 0.05,
  done: 0.92,
  active: 1.0,
  error: 0.65,
};

function fallback(message) {
  if (!host) return;
  host.innerHTML = "";
  const p = document.createElement("p");
  p.className = "fallback";
  p.textContent = message;
  host.appendChild(p);
  // The same seven stages, as text, so the plate is never an empty box.
  const list = document.createElement("ol");
  list.className = "fallback-steps";
  for (const s of STAGES) {
    const li = document.createElement("li");
    li.className = s.status || "pending";
    li.textContent = `${s.name} (${s.status || "pending"})${s.detail ? ": " + s.detail : ""}`;
    list.appendChild(li);
  }
  host.appendChild(list);
}

let THREE = null;
for (const url of window.__THREE_URLS__ || []) {
  try {
    THREE = await import(url);
    break;
  } catch {
    /* try the next source */
  }
}
if (!THREE) {
  fallback("The 3D view could not load here. The steps are listed below instead.");
  throw new Error("Three.js unavailable");
}

/* ------------------------------------------------------------------ *
 * Renderer, Scene, Camera & Lighting
 * ------------------------------------------------------------------ */
let renderer;
try {
  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
  });
} catch (err) {
  fallback("This 3D view isn't available here. See the steps list below instead.");
  throw err;
}

renderer.setClearColor(0x000000, 0);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = isNight ? 1.15 : 1.0;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);

// Warm ambient + directional lighting, like a reading lamp over the desk
const ambientLight = new THREE.AmbientLight(
  isNight ? 0x3a2c1c : 0xfaf1de,
  isNight ? 1.1 : 0.85
);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(isNight ? 0xffe0b0 : 0xffffff, isNight ? 1.0 : 0.65);
dirLight.position.set(6, 12, 8);
scene.add(dirLight);

const fillLight = new THREE.DirectionalLight(
  isNight ? 0xf0a24a : 0xecdfc4,
  0.35
);
fillLight.position.set(-6, -4, -4);
scene.add(fillLight);

const drawing = new THREE.Group();
scene.add(drawing);

/* ------------------------------------------------------------------ *
 * Shared Resources & Disposal Tracking
 * ------------------------------------------------------------------ */
const disposables = [];
const keep = (res) => {
  if (res) disposables.push(res);
  return res;
};

const cageGeo = keep(new THREE.EdgesGeometry(new THREE.BoxGeometry(CAGE, CAGE, CAGE)));
const pickGeo = keep(new THREE.BoxGeometry(CAGE * 1.4, CAGE * 1.4, CAGE * 1.4));
const pickMat = keep(
  new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0,
    depthWrite: false,
    colorWrite: false,
  })
);

/* ------------------------------------------------------------------ *
 * Core Stage Geometries
 * ------------------------------------------------------------------ */
function coreGeometries(index) {
  switch (index) {
    case 0: // Ingestion: layered raw data sheet
      return [
        [new THREE.BoxGeometry(0.66, 0.06, 0.66), 0, -0.04, 0],
        [new THREE.BoxGeometry(0.52, 0.05, 0.52), 0, 0.06, 0],
      ];
    case 1: // Reasoning: prismatic crystal of thought
      return [
        [new THREE.OctahedronGeometry(0.38, 0), 0, 0, 0],
      ];
    case 2: // Execution: solid machine caliper block with bore
      return [
        [new THREE.BoxGeometry(0.48, 0.48, 0.48), 0, 0, 0],
        [new THREE.CylinderGeometry(0.12, 0.12, 0.52, 16), 0, 0, 0],
      ];
    case 3: // Interpretation: faceted geodesic sphere
      return [
        [new THREE.IcosahedronGeometry(0.38, 1), 0, 0, 0],
      ];
    case 4: // Refinement: gyroscopic dual-ring torus
      return [
        [new THREE.TorusGeometry(0.30, 0.07, 12, 28), 0, 0, 0],
      ];
    case 5: // RLM Decomposition: 4 recursive sub-task cubes
      return [-0.16, 0.16].flatMap((x) =>
        [-0.16, 0.16].map((z) => [new THREE.BoxGeometry(0.22, 0.22, 0.22), x, 0, z])
      );
    default: // Report: final bound specification sheets
      return [
        [new THREE.BoxGeometry(0.62, 0.05, 0.52), 0, 0.08, 0],
        [new THREE.BoxGeometry(0.62, 0.05, 0.52), 0, -0.06, 0],
      ];
  }
}

function buildCore(index, fillMat, edgeMat) {
  const group = new THREE.Group();
  for (const [geometry, x, y, z] of coreGeometries(index)) {
    keep(geometry);
    const fill = new THREE.Mesh(geometry, fillMat);
    fill.position.set(x, y, z);
    group.add(fill);

    const edges = new THREE.LineSegments(keep(new THREE.EdgesGeometry(geometry)), edgeMat);
    edges.position.set(x, y, z);
    group.add(edges);
  }
  return group;
}

/* ------------------------------------------------------------------ *
 * Modules & Bench Shadows
 * ------------------------------------------------------------------ */
const shadowGeo = keep(new THREE.PlaneGeometry(1.2, 1.2));
const shadowMat = keep(
  new THREE.MeshBasicMaterial({
    color: new THREE.Color(isNight ? 0x120d08 : 0x3a2b1e),
    transparent: true,
    opacity: isNight ? 0.22 : 0.09,
    depthWrite: false,
  })
);

const nodes = STAGES.map((stage, i) => {
  const group = new THREE.Group();
  group.position.x = xFor(i);

  // Ground contact shadow disc
  const shadow = new THREE.Mesh(shadowGeo, shadowMat);
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = -1.33;
  group.add(shadow);

  // Wireframe cage
  const cageMat = keep(
    new THREE.LineBasicMaterial({
      color: new THREE.Color(C.graphite || "#8a7660"),
      transparent: true,
      opacity: CAGE_OPACITY.pending,
    })
  );
  const cage = new THREE.LineSegments(cageGeo, cageMat);
  group.add(cage);

  // Volumetric material for core solid
  const fillMat = keep(
    new THREE.MeshStandardMaterial({
      color: new THREE.Color(C.sheet || "#fffbf2"),
      roughness: 0.25,
      metalness: 0.35,
      transparent: true,
      opacity: FILL_OPACITY.pending,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    })
  );

  const edgeMat = keep(
    new THREE.LineBasicMaterial({
      color: new THREE.Color(C.graphite || "#8a7660"),
      transparent: true,
      opacity: 0.55,
    })
  );

  const core = buildCore(i, fillMat, edgeMat);
  group.add(core);

  // Raycasting hit target
  const pick = new THREE.Mesh(pickGeo, pickMat);
  pick.userData.index = i;
  group.add(pick);

  drawing.add(group);
  return { stage, group, cage, cageMat, core, fillMat, edgeMat, pick };
});

/* ------------------------------------------------------------------ *
 * Connectors, Refinement Arc, Satellites
 * ------------------------------------------------------------------ */
function makeStroke(points, color, opacity) {
  const material = keep(
    new THREE.LineBasicMaterial({
      color: new THREE.Color(color),
      transparent: true,
      opacity,
    })
  );
  const geometry = keep(new THREE.BufferGeometry().setFromPoints(points));
  const line = new THREE.Line(geometry, material);
  drawing.add(line);
  return { material, geometry, count: points.length };
}

function straight(from, to) {
  return new THREE.LineCurve3(from, to).getPoints(STROKE_SAMPLES);
}

const links = nodes.slice(0, -1).map((_, i) =>
  makeStroke(
    straight(
      new THREE.Vector3(xFor(i) + CAGE / 2, 0, 0),
      new THREE.Vector3(xFor(i + 1) - CAGE / 2, 0, 0)
    ),
    C.graphite || "#8a7660",
    0.42
  )
);

// Stage 5 -> Stage 3 refinement feedback arc
const refineArcCurve = new THREE.QuadraticBezierCurve3(
  new THREE.Vector3(xFor(REFINE_FROM), CAGE / 2, 0),
  new THREE.Vector3((xFor(REFINE_FROM) + xFor(REFINE_TO)) / 2, 2.6, 0),
  new THREE.Vector3(xFor(REFINE_TO), CAGE / 2, 0)
);
const refineArc = makeStroke(refineArcCurve.getPoints(64), C.graphite || "#8a7660", 0.35);

// Stage 6 RLM recursive satellites
const satellites = [-1, 1].map((side) => {
  const pos = new THREE.Vector3(xFor(RLM_INDEX), 0.15, side * 1.35);
  const geo = keep(new THREE.BoxGeometry(0.24, 0.24, 0.24));

  const fillMat = keep(
    new THREE.MeshStandardMaterial({
      color: new THREE.Color(C.sheet || "#fffbf2"),
      roughness: 0.28,
      metalness: 0.35,
      transparent: true,
      opacity: 0.1,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    })
  );
  const edgeMat = keep(
    new THREE.LineBasicMaterial({
      color: new THREE.Color(C.graphite || "#8a7660"),
      transparent: true,
      opacity: 0.5,
    })
  );

  const fill = new THREE.Mesh(geo, fillMat);
  fill.position.copy(pos);
  drawing.add(fill);

  const edges = new THREE.LineSegments(keep(new THREE.EdgesGeometry(geo)), edgeMat);
  edges.position.copy(pos);
  drawing.add(edges);

  const tether = makeStroke(
    straight(new THREE.Vector3(xFor(RLM_INDEX), 0, side * (CAGE / 2)), pos.clone()),
    C.graphite || "#8a7660",
    0.3
  );

  return { fillMat, edgeMat, fill, edges, tether, position: pos };
});

// Bench grid plane
const gridColor = new THREE.Color(C.grid || C.ink || "#3a2b1e");
const grid = new THREE.GridHelper(22, 22, gridColor, gridColor);
grid.position.y = -1.35;
grid.material.transparent = true;
grid.material.opacity = isNight ? 0.12 : 0.05;
keep(grid.geometry);
keep(grid.material);
drawing.add(grid);

/* ------------------------------------------------------------------ *
 * Dynamic Fluid Ink Particle Stream
 * ------------------------------------------------------------------ */
const PARTICLE_COUNT = 84;
const pGeo = keep(new THREE.OctahedronGeometry(0.042, 0));
const pMat = keep(
  new THREE.MeshBasicMaterial({
    color: new THREE.Color(C.pen || "#a34f20"),
    transparent: true,
    opacity: 0.85,
  })
);
const particleMesh = new THREE.InstancedMesh(pGeo, pMat, PARTICLE_COUNT);
particleMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
particleMesh.visible = false;
drawing.add(particleMesh);

// Pre-initialize particle offsets and paths
const particleData = Array.from({ length: PARTICLE_COUNT }, (_, i) => ({
  t: i / PARTICLE_COUNT,
  speed: 0.25 + (i % 5) * 0.05,
  spread: (Math.random() - 0.5) * 0.12,
  isArc: i % 4 === 0, // portion of particles travel overhead refinement arc
}));
const pMatrix = new THREE.Matrix4();
const pPos = new THREE.Vector3();

function updateParticles(delta) {
  const reachedCount = STAGES.filter((s) => isReached(s.status)).length;
  if (reachedCount <= 0) {
    particleMesh.visible = false;
    return;
  }
  particleMesh.visible = true;

  const maxStage = Math.max(0, reachedCount - 1);
  const startX = xFor(0) - 0.8;
  const endX = xFor(maxStage);
  const totalDist = Math.max(endX - startX, 0.1);
  const hasRefined = isReached(STAGES[REFINE_FROM]?.status);

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const p = particleData[i];
    p.t = (p.t + delta * p.speed) % 1;

    if (p.isArc && hasRefined && maxStage >= REFINE_FROM) {
      // Traverse overhead refinement loop
      refineArcCurve.getPoint(p.t, pPos);
      pPos.z += p.spread;
    } else {
      // Traverse main rail
      pPos.x = startX + p.t * totalDist;
      pPos.y = Math.sin(p.t * Math.PI * (maxStage + 1)) * 0.06;
      pPos.z = p.spread;
    }

    const scale = 0.8 + Math.sin(p.t * Math.PI * 4) * 0.25;
    pMatrix.makeScale(scale, scale, scale);
    pMatrix.setPosition(pPos.x, pPos.y, pPos.z);
    particleMesh.setMatrixAt(i, pMatrix);
  }
  particleMesh.instanceMatrix.needsUpdate = true;
}

/* ------------------------------------------------------------------ *
 * Camera Framing & CAD Projections
 * ------------------------------------------------------------------ */
const fitPoints = [];
STAGES.forEach((_, i) => {
  for (const dx of [-CAGE / 2, CAGE / 2]) {
    for (const dy of [-CAGE / 2, CAGE / 2]) {
      for (const dz of [-CAGE / 2, CAGE / 2]) {
        fitPoints.push(new THREE.Vector3(xFor(i) + dx, dy, dz));
      }
    }
  }
});
fitPoints.push(new THREE.Vector3((xFor(REFINE_FROM) + xFor(REFINE_TO)) / 2, 2.7, 0));
for (const side of [-1, 1]) {
  fitPoints.push(new THREE.Vector3(xFor(RLM_INDEX), 0.3, side * 1.5));
}

const FIT_MARGIN = 0.88;
const orbit = { theta: 0.42, phi: 1.16, radius: 12.5 };
const drag = { theta: 0, phi: 0 };
const parallax = { x: 0, y: 0 };
const smooth = { theta: 0.42, phi: 1.16 };
const probe = new THREE.PerspectiveCamera();
const ndc = new THREE.Vector3();
const camRight = new THREE.Vector3();
const camUp = new THREE.Vector3();
let framed = false;

function placeCamera(cam, radius, theta, phi) {
  cam.position.set(
    radius * Math.sin(phi) * Math.sin(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.cos(theta)
  );
  cam.lookAt(0, 0, 0);
}

function projectedBounds() {
  probe.updateMatrixWorld(true);
  const b = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
  for (const point of fitPoints) {
    ndc.copy(point).project(probe);
    b.minX = Math.min(b.minX, ndc.x);
    b.maxX = Math.max(b.maxX, ndc.x);
    b.minY = Math.min(b.minY, ndc.y);
    b.maxY = Math.max(b.maxY, ndc.y);
  }
  return b;
}

function fitCamera(width, height) {
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();

  orbit.theta = THREE.MathUtils.clamp(
    THREE.MathUtils.mapLinear(camera.aspect, 1.1, 2.6, 0.88, 0.32),
    0.32,
    0.88
  );

  probe.copy(camera);
  drawing.position.set(0, 0, 0);

  let radius = 14;
  let bounds = null;
  for (let pass = 0; pass < 6; pass++) {
    placeCamera(probe, radius, orbit.theta, orbit.phi);
    bounds = projectedBounds();
    const spanX = (bounds.maxX - bounds.minX) / 2;
    const spanY = (bounds.maxY - bounds.minY) / 2;
    radius *= Math.max(spanX, spanY) / FIT_MARGIN;
  }
  orbit.radius = THREE.MathUtils.clamp(radius, 7, 36);

  placeCamera(probe, orbit.radius, orbit.theta, orbit.phi);
  bounds = projectedBounds();
  const halfH = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * orbit.radius;
  const halfW = halfH * camera.aspect;
  camRight.setFromMatrixColumn(probe.matrixWorld, 0);
  camUp.setFromMatrixColumn(probe.matrixWorld, 1);
  drawing.position
    .copy(camRight)
    .multiplyScalar((-(bounds.minX + bounds.maxX) / 2) * halfW)
    .addScaledVector(camUp, (-(bounds.minY + bounds.maxY) / 2) * halfH);

  if (!framed) {
    smooth.theta = orbit.theta;
    smooth.phi = orbit.phi;
    framed = true;
  }
}

function updateCamera(delta = 0.016) {
  const targetTheta = orbit.theta + drag.theta + parallax.x * 0.12;
  const targetPhi = THREE.MathUtils.clamp(orbit.phi + drag.phi + parallax.y * 0.08, 0.1, 1.65);
  const moving =
    Math.abs(targetTheta - smooth.theta) > 1e-4 || Math.abs(targetPhi - smooth.phi) > 1e-4;

  // Frame-rate independent exponential damping: 1 - exp(-lambda * dt)
  const k = 1.0 - Math.exp(-MOTION.damp.cursor * delta * 2.5);
  smooth.theta += (targetTheta - smooth.theta) * k;
  smooth.phi += (targetPhi - smooth.phi) * k;

  placeCamera(camera, orbit.radius, smooth.theta, smooth.phi);
  return moving;
}

/* ------------------------------------------------------------------ *
 * Readout & 3D Projected HUD Callout Pins
 * ------------------------------------------------------------------ */
const STATUS_WORD = {
  done: "Completed",
  active: "Executing",
  error: "Inspection Flag",
  skipped: "Omitted",
  pending: "Pending",
};

function defaultReadout() {
  const errored = STAGES.find((s) => s.status === "error");
  const running = STAGES.find((s) => s.status === "active");
  const finished = STAGES.filter((s) => s.status === "done").length;

  if (errored) return ["Something Needs Checking", `Step ${errored.num} flagged: ${errored.name}`];
  if (running) return ["Working On It", `Step ${running.num} of ${STAGES.length} — ${running.name}`];
  if (finished === STAGES.length) return ["All Done!", `All ${STAGES.length} steps finished.`];
  if (finished > 0) return ["Stopped Partway", `${finished} of ${STAGES.length} steps finished.`];
  return ["Ready When You Are", "Upload your file and tell us what you'd like to know."];
}

function setReadout(title, body) {
  if (elTitle) elTitle.textContent = title;
  if (elBody) elBody.textContent = body;
}
setReadout(...defaultReadout());

const hudPos = new THREE.Vector3();
function updateHUD(hoveredIndex, activeIndex) {
  if (!hudPin || !hudBadge) return;
  const targetIdx = hoveredIndex >= 0 ? hoveredIndex : activeIndex;
  if (targetIdx < 0 || targetIdx >= nodes.length) {
    hudPin.classList.remove("active");
    return;
  }

  const node = nodes[targetIdx];
  node.group.getWorldPosition(hudPos);
  hudPos.y += CAGE * 0.75;
  hudPos.project(camera);

  const rect = renderer.domElement.getBoundingClientRect();
  const x = (hudPos.x * 0.5 + 0.5) * rect.width;
  const y = (-hudPos.y * 0.5 + 0.5) * rect.height;

  hudPin.style.left = `${x}px`;
  hudPin.style.top = `${y}px`;

  const s = node.stage;
  if (hudStep) hudStep.textContent = s.num.padStart(2, "0");
  if (hudText) hudText.textContent = `${s.name} [${STATUS_WORD[s.status] || "Ready"}]`;

  if (s.status === "error") {
    hudBadge.classList.add("err");
  } else {
    hudBadge.classList.remove("err");
  }

  hudPin.classList.add("active");
}

/* ------------------------------------------------------------------ *
 * Inking State Transitions
 * ------------------------------------------------------------------ */
function ink(i, animated) {
  const { stage, group, cageMat, fillMat, edgeMat } = nodes[i];
  const color = new THREE.Color(inkFor(stage.status));
  const cageOpacity = CAGE_OPACITY[stage.status] ?? CAGE_OPACITY.pending;
  const fillOpacity = FILL_OPACITY[stage.status] ?? 0.15;
  const edgeOpacity = isReached(stage.status) ? 1.0 : 0.5;

  if (fillMat && fillMat.emissive) {
    if (stage.status === "active") {
      fillMat.emissive.copy(new THREE.Color(C.pen || "#a34f20"));
      fillMat.emissiveIntensity = 0.32;
    } else if (stage.status === "done") {
      fillMat.emissive.copy(new THREE.Color(C.pen || "#a34f20"));
      fillMat.emissiveIntensity = 0.12;
    } else if (stage.status === "error") {
      fillMat.emissive.copy(new THREE.Color(C.risk || "#a33526"));
      fillMat.emissiveIntensity = 0.42;
    } else {
      fillMat.emissive.setHex(0x000000);
      fillMat.emissiveIntensity = 0;
    }
  }

  if (!animated || REDUCED_MOTION) {
    cageMat.color.copy(color);
    cageMat.opacity = cageOpacity;
    edgeMat.color.copy(color);
    edgeMat.opacity = edgeOpacity;
    fillMat.opacity = fillOpacity;
    return;
  }

  gsap.timeline({ onUpdate: invalidate })
    .to([cageMat.color, edgeMat.color], {
      r: color.r,
      g: color.g,
      b: color.b,
      duration: 0.35,
      ease: "power2.out",
    }, 0)
    .to(cageMat, { opacity: cageOpacity, duration: 0.35 }, 0)
    .to(edgeMat, { opacity: edgeOpacity, duration: 0.35 }, 0)
    .to(fillMat, { opacity: fillOpacity, duration: 0.4, ease: "power2.out" }, 0)
    .to(group.scale, { x: 1.12, y: 1.12, z: 1.12, duration: 0.15, ease: "back.out(2)" }, 0)
    .to(group.scale, { x: 1, y: 1, z: 1, duration: 0.3, ease: "power2.out" }, 0.15);
}

function inkStructure(animated) {
  const reached = STAGES.map((s) => isReached(s.status));
  const paint = (mat, col, op) => {
    const target = new THREE.Color(col);
    if (!animated || REDUCED_MOTION) {
      mat.color.copy(target);
      mat.opacity = op;
      return;
    }
    gsap.to(mat.color, { r: target.r, g: target.g, b: target.b, duration: 0.45, onUpdate: invalidate });
    gsap.to(mat, { opacity: op, duration: 0.45, onUpdate: invalidate });
  };

  links.forEach((link, i) => {
    const crossed = reached[i] && reached[i + 1];
    paint(link.material, crossed ? (C.pen || "#a34f20") : (C.graphite || "#8a7660"), crossed ? 0.9 : 0.42);
  });

  const refined = reached[REFINE_FROM];
  paint(refineArc.material, refined ? (C.pen || "#a34f20") : (C.graphite || "#8a7660"), refined ? 0.8 : 0.35);

  const recursed = reached[RLM_INDEX];
  for (const sat of satellites) {
    paint(sat.edgeMat, recursed ? (C.pen || "#a34f20") : (C.graphite || "#8a7660"), recursed ? 1 : 0.5);
    paint(sat.fillMat, C.sheet || "#fffbf2", recursed ? 0.92 : 0.1);
    paint(sat.tether.material, recursed ? (C.pen || "#a34f20") : (C.graphite || "#8a7660"), recursed ? 0.75 : 0.3);
  }
}

/* ------------------------------------------------------------------ *
 * Entrance Animation Sequence
 * ------------------------------------------------------------------ */
const strokes = [...links, refineArc, ...satellites.map((s) => s.tether)];

function drawStroke(tl, stroke, at, dur) {
  const walk = { n: 0 };
  stroke.geometry.setDrawRange(0, 0);
  tl.to(
    walk,
    {
      n: stroke.count,
      duration: dur,
      ease: "none",
      onUpdate: () => {
        stroke.geometry.setDrawRange(0, Math.ceil(walk.n));
        invalidate();
      },
    },
    at
  );
}

function snapToFinalState() {
  for (const stroke of strokes) stroke.geometry.setDrawRange(0, stroke.count);
  nodes.forEach((_, i) => ink(i, false));
  inkStructure(false);
  invalidate();
}

function playEntrance() {
  if (REDUCED_MOTION) {
    snapToFinalState();
    return;
  }
  if (hasPlayedEntranceFor(RUN_SIGNATURE)) {
    snapToFinalState();
    return;
  }
  markEntrancePlayed(RUN_SIGNATURE);

  inkStructure(true);
  const tl = gsap.timeline({ onUpdate: invalidate });

  // 1. Grid bench ruling in and scene rotation
  tl.from(grid.material, { opacity: 0, duration: 0.8, ease: "power2.out" }, 0)
    .from(drawing.rotation, { y: -0.4, duration: 1.2, ease: "power3.out" }, 0);

  // 2. Rail stroke reveal
  links.forEach((link, i) => drawStroke(tl, link, 0.12 + i * 0.06, 0.22));

  // 3. Modules stamp down
  tl.from(
    nodes.map((n) => n.group.scale),
    { x: 0.01, y: 0.01, z: 0.01, duration: 0.45, stagger: 0.05, ease: "back.out(2)" },
    0.2
  );

  // 4. Overhead arcs & RLM tethers
  drawStroke(tl, refineArc, 0.55, 0.4);
  satellites.forEach((sat, i) => drawStroke(tl, sat.tether, 0.65 + i * 0.05, 0.2));

  // 5. Inking visited stages
  const traversal = STAGES.reduce((acc, s, i) => (isReached(s.status) ? [...acc, i] : acc), []);
  let at = 0.8;
  for (const idx of traversal) {
    tl.call(() => ink(idx, true), undefined, at);
    at += 0.18;
  }
}

/* ------------------------------------------------------------------ *
 * Interaction & Pointer Handlers
 * ------------------------------------------------------------------ */
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const pickTargets = nodes.map((n) => n.pick);
const canvas = renderer.domElement;

let hovered = -1;
let dragging = false;
let lastPointer = { x: 0, y: 0 };
const activeIndex = STAGES.findIndex((s) => s.status === "active");

function setHover(i) {
  if (i === hovered) return;
  hovered = i;

  if (i < 0) {
    setReadout(...defaultReadout());
    canvas.style.cursor = "grab";
  } else {
    const s = STAGES[i];
    const det = s.detail ? ` — ${s.detail}` : "";
    setReadout(`${s.num}. ${s.name}`, `${STATUS_WORD[s.status] || "Not run"}${det}`);
    canvas.style.cursor = "pointer";
  }
  updateHUD(hovered, activeIndex);
  invalidate();
}

canvas.addEventListener("pointermove", (e) => {
  const rect = canvas.getBoundingClientRect();
  const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const ny = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  if (dragging) {
    drag.theta = THREE.MathUtils.clamp(drag.theta - (e.clientX - lastPointer.x) * 0.005, -0.85, 0.85);
    drag.phi = THREE.MathUtils.clamp(drag.phi - (e.clientY - lastPointer.y) * 0.004, -0.4, 0.4);
    lastPointer = { x: e.clientX, y: e.clientY };
    invalidate();
    return;
  }

  parallax.x = nx;
  parallax.y = ny;
  pointer.set(nx, ny);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(pickTargets, false)[0];
  setHover(hit ? hit.object.userData.index : -1);
});

canvas.addEventListener("pointerdown", (e) => {
  dragging = true;
  lastPointer = { x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
  canvas.style.cursor = "grabbing";
});

function endDrag(e) {
  if (!dragging) return;
  dragging = false;
  if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  canvas.style.cursor = hovered >= 0 ? "pointer" : "grab";
}
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

canvas.addEventListener("pointerleave", () => {
  parallax.x = 0;
  parallax.y = 0;
  setHover(-1);
});

/* Keyboard Navigation */
canvas.tabIndex = 0;
canvas.setAttribute("role", "img");
canvas.setAttribute("aria-label", "3D view of the analysis steps and their progress");

const KEY_STEP = 0.14;
canvas.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") drag.theta = THREE.MathUtils.clamp(drag.theta + KEY_STEP, -0.85, 0.85);
  else if (e.key === "ArrowRight") drag.theta = THREE.MathUtils.clamp(drag.theta - KEY_STEP, -0.85, 0.85);
  else if (e.key === "ArrowUp") drag.phi = THREE.MathUtils.clamp(drag.phi + KEY_STEP * 0.6, -0.4, 0.4);
  else if (e.key === "ArrowDown") drag.phi = THREE.MathUtils.clamp(drag.phi - KEY_STEP * 0.6, -0.4, 0.4);
  else if (e.key === "Home") {
    drag.theta = 0;
    drag.phi = 0;
    setHover(-1);
  } else if (e.key === "]" || e.key === "[") {
    const next = e.key === "]" ? hovered + 1 : hovered - 1;
    setHover(next < 0 || next >= STAGES.length ? -1 : next);
  } else return;

  e.preventDefault();
  invalidate();
});

/* CAD Perspective Controls */
function bindCadButtons() {
  const btnIso = document.getElementById("btn-iso");
  const btnPlan = document.getElementById("btn-plan");
  const btnFront = document.getElementById("btn-front");
  const btnReset = document.getElementById("btn-reset");

  const setCadActive = (btn) => {
    [btnIso, btnPlan, btnFront].forEach((b) => b && b.classList.remove("active"));
    if (btn) btn.classList.add("active");
  };

  const tweenCamera = (theta, phi) => {
    gsap.to(orbit, {
      theta,
      phi,
      duration: 0.65,
      ease: "power2.inOut",
      onUpdate: () => {
        drag.theta = 0;
        drag.phi = 0;
        invalidate();
      },
    });
  };

  if (btnIso) {
    btnIso.addEventListener("click", () => {
      setCadActive(btnIso);
      tweenCamera(0.42, 1.16);
    });
  }
  if (btnPlan) {
    btnPlan.addEventListener("click", () => {
      setCadActive(btnPlan);
      tweenCamera(0.001, 0.15); // Top-down architectural plan
    });
  }
  if (btnFront) {
    btnFront.addEventListener("click", () => {
      setCadActive(btnFront);
      tweenCamera(0.001, Math.PI / 2 - 0.05); // Elevation front
    });
  }
  const btnCinema = document.getElementById("btn-cinema");
  if (btnCinema) {
    btnCinema.addEventListener("click", () => {
      setCadActive(btnCinema);
      // One animation engine for the scene: a GSAP timeline through three keyframes, about seven seconds.
      const sweep = { onUpdate: () => { drag.theta = 0; drag.phi = 0; invalidate(); } };
      gsap.timeline({ onComplete: () => { if (btnIso) setCadActive(btnIso); } })
        .to(orbit, { theta: 0.95, phi: 0.75, duration: 2.33, ease: "sine.inOut", ...sweep })
        .to(orbit, { theta: -0.45, phi: 1.25, duration: 2.33, ease: "sine.inOut", ...sweep })
        .to(orbit, { theta: 0.42, phi: 1.16, duration: 2.34, ease: "sine.inOut", ...sweep });
    });
  }
  if (btnReset) {
    btnReset.addEventListener("click", () => {
      setCadActive(btnIso);
      drag.theta = 0;
      drag.phi = 0;
      parallax.x = 0;
      parallax.y = 0;
      tweenCamera(0.42, 1.16);
    });
  }
}
bindCadButtons();

/* ------------------------------------------------------------------ *
 * Render Loop & Animation Dynamics
 * ------------------------------------------------------------------ */
let onScreen = true;
let pageVisible = !document.hidden;
let dirty = true;

function invalidate() {
  dirty = true;
}

function resize() {
  const w = host.clientWidth;
  const h = host.clientHeight;
  if (!w || !h) return;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h, false);
  fitCamera(w, h);
  updateHUD(hovered, activeIndex);
  invalidate();
}

new ResizeObserver(resize).observe(host);
resize();

new IntersectionObserver(([e]) => {
  onScreen = e.isIntersecting;
  invalidate();
}, { threshold: 0 }).observe(host);

document.addEventListener("visibilitychange", () => {
  pageVisible = !document.hidden;
  invalidate();
});

const clock = new THREE.Clock();

renderer.setAnimationLoop(() => {
  if (!onScreen || !pageVisible) return;

  const delta = clock.getDelta();
  const t = clock.getElapsedTime();
  const cameraMoving = updateCamera(delta);

  // Active Stage Kinetic Animations
  if (activeIndex >= 0 && nodes[activeIndex]) {
    const core = nodes[activeIndex].core;
    switch (activeIndex) {
      case 0: // Ingestion: scan translation
        core.position.y = Math.sin(t * 3) * 0.04;
        break;
      case 1: // Reasoning: dual crystal rotation
        core.rotation.y = t * 0.7;
        core.rotation.x = Math.sin(t * 0.5) * 0.25;
        break;
      case 2: // Execution: mechanical stepped spin
        core.rotation.y = Math.floor(t * 1.5) * (Math.PI / 2);
        core.rotation.z = Math.sin(t * 4) * 0.08;
        break;
      case 3: // Interpretation: geodesic precession
        core.rotation.y = t * 0.5;
        core.rotation.z = t * 0.3;
        break;
      case 4: // Refinement: gyro precession
        core.rotation.x = t * 1.2;
        core.rotation.y = t * 0.8;
        break;
      case 5: // RLM Decomposition: satellite breathing
        satellites.forEach((sat, si) => {
          sat.fill.position.y = sat.position.y + Math.sin(t * 3 + si) * 0.08;
          sat.edges.position.y = sat.fill.position.y;
        });
        break;
      case 6: // Report: gentle float
        core.position.y = Math.sin(t * 2) * 0.03;
        break;
    }
    dirty = true;
  }

  // Fluid particle stream
  updateParticles(delta);

  // Floating HUD coordinates
  if (hovered >= 0 || activeIndex >= 0) {
    updateHUD(hovered, activeIndex);
  }

  if (!dirty && !cameraMoving && activeIndex < 0 && !particleMesh.visible) return;
  dirty = false;

  renderer.render(scene, camera);
});

playEntrance();

/* Teardown */
window.addEventListener("pagehide", () => {
  renderer.setAnimationLoop(null);
  gsap.globalTimeline.clear();
  for (const r of disposables) r.dispose?.();
  renderer.dispose();
});
