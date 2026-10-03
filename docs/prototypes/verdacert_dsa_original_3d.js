(function initPipelineTrack() {
  'use strict';

  /* ------------------------------------------------------------------
     How the motion stays smooth
     - The scroll handler only records a target (no DOM writes).
     - One animation loop eases toward that target and writes only
       compositor-friendly properties (clip-path, transform, opacity).
     - The card is always full size; the collapsed "window" is a clip-path
       inset, so the canvas is never resized while scrolling.
     - Layout is measured on resize (ResizeObserver), never per frame.
     - Rendering pauses when the track is off screen or the tab is hidden.
     ------------------------------------------------------------------ */

  const THREE = window.THREE;
  const canvas = document.getElementById('solare-webgl-canvas');
  const leaderCanvas = document.getElementById('solare-leader-canvas');
  const leaderContext = leaderCanvas ? leaderCanvas.getContext('2d') : null;
  const canvasWrap = document.querySelector('.solare-canvas-wrap');
  const track = document.getElementById('pipeline-scrolly-section');
  const card = document.getElementById('solare-card');
  const hudPin = document.getElementById('solare-hud-pin');
  const hudText = document.getElementById('solare-hud-text');
  const panelWrap = document.querySelector('.solare-stage-panel-wrap');
  const prelude = document.getElementById('solare-prelude');
  const outro = document.getElementById('solare-outro');
  const preludeFill = document.getElementById('prelude-fill');
  const crop = document.getElementById('solare-crop');
  const cropMarks = crop ? Array.from(crop.querySelectorAll('i')) : [];
  const nav = document.querySelector('.nav');
  const telemetryTag = document.getElementById('telemetry-stage-tag');
  const telemetrySub = document.getElementById('telemetry-sub-tag');
  const telemetryFill = document.getElementById('telemetry-fill');
  const fallback = document.getElementById('solare-webgl-fallback');
  const landingSurface = document.getElementById('surface-landing');

  if (!THREE || !canvas || !track || !card || !canvasWrap) {
    if (fallback) fallback.hidden = false;
    return;
  }

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
  } catch (error) {
    canvas.hidden = true;
    if (fallback) fallback.hidden = false;
    return;
  }

  const reduceMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  let reduceMotion = reduceMotionQuery.matches;

  const clamp01 = (value) => Math.max(0, Math.min(1, value));
  const smoothstep = (start, end, value) => {
    const amount = clamp01((value - start) / (end - start));
    return amount * amount * (3 - 2 * amount);
  };
  // Smootherstep: leaves slowly, moves fast through the middle, lands softly.
  const settle = (amount) => amount * amount * amount * (amount * (amount * 6 - 15) + 10);
  // Where on the scroll track the card opens and closes. `expansion` is 0 (the
  // closed window) to 1 (the full viewport); `exit` is the closing share.
  function trackPhase(normalized) {
    if (normalized < ENTER_THRESHOLD) {
      return { expansion: settle(clamp01((normalized - HOLD_IN) / (ENTER_THRESHOLD - HOLD_IN))), exit: 0 };
    }
    if (normalized > EXIT_THRESHOLD) {
      const exit = settle(clamp01((normalized - EXIT_THRESHOLD) / (1 - HOLD_OUT - EXIT_THRESHOLD)));
      return { expansion: 1 - exit, exit };
    }
    return { expansion: 1, exit: 0 };
  }

  /* The 3D inks are the page's own --stage-* tokens (see the CSS), read again on
     every theme change so the scene always matches the card it sits on. The
     fallbacks are the night values. */
  const readToken = (name, fallback) => {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  };
  function readPalette() {
    return {
      paper: readToken('--stage-token-paper', '#f4efe1'),
      pen: readToken('--stage-token-pen', '#8fd1a8'),
      accent: readToken('--stage-token-accent', '#e0a458'),
      positive: readToken('--stage-token-positive', '#5fbf85'),
      graphite: readToken('--stage-token-graphite', '#8fa396'),
      risk: readToken('--stage-token-risk', '#ee8f78'),
      grid: readToken('--stage-grid', '#5b8a70'),
      gridOpacity: Number.isFinite(parseFloat(readToken('--stage-grid-opacity', '0.16')))
        ? parseFloat(readToken('--stage-grid-opacity', '0.16'))
        : 0.16
    };
  }
  let PALETTE = readPalette();
  const hexToRgba = (hex, alpha) => {
    const n = parseInt(hex.replace('#', ''), 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
  };
  // r128 does no colour management: convert so sRGB hexes render as written.
  const linear = (hex) => new THREE.Color(hex).convertSRGBToLinear();

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(46, 1, 0.1, 1000);
  camera.position.set(0, 0, 30);

  const worldGroup = new THREE.Group();
  const parallaxGroup = new THREE.Group();
  worldGroup.add(parallaxGroup);
  scene.add(worldGroup);

  const dock = { stageX: 0, stageY: 0 };
  function updateDocking() {
    const desktop = window.innerWidth >= 960;
    dock.stageX = desktop ? 9.2 : 0;
    dock.stageY = desktop ? -0.2 : -2.2;
  }
  updateDocking();

  // The coordinate-paper backdrop. It sits behind every token (so no line ever
  // crosses a cube), is wider than any viewport (so it never shows an edge), and
  // stays put while the scene slides and scales in front of it.
  // GridHelper bakes its colour into the geometry, so a theme change rebuilds it.
  const GRID_SIZE = 140;
  const GRID_DIVISIONS = 108;
  const GRID_DEPTH = -8;
  let gridHelper = null;
  function buildGrid() {
    if (gridHelper) {
      scene.remove(gridHelper);
      gridHelper.geometry.dispose();
      gridHelper.material.dispose();
      gridHelper = null;
    }
    if (PALETTE.gridOpacity <= 0) return; // a theme with no grid builds none
    gridHelper = new THREE.GridHelper(GRID_SIZE, GRID_DIVISIONS, linear(PALETTE.grid), linear(PALETTE.grid));
    gridHelper.position.z = GRID_DEPTH;
    gridHelper.rotation.x = Math.PI / 2;
    gridHelper.material.transparent = true;
    gridHelper.material.opacity = PALETTE.gridOpacity;
    gridHelper.material.depthWrite = false;
    gridHelper.renderOrder = -1;
    scene.add(gridHelper);
  }
  buildGrid();

  // Front faces sum to about 1.0, so a token shows the colour it was given.
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const keyLight = new THREE.DirectionalLight(0xffffff, 0.62);
  keyLight.position.set(-8, 12, 18);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.2);
  fillLight.position.set(12, -4, 10);
  scene.add(fillLight);

  const TOKEN_COUNT = 720;
  const FORMATION_COUNT = 4; // one per stage; the last one holds while the card closes
  const tokenGeometry = new THREE.BoxGeometry(0.42, 0.42, 0.16);
  // No vertexColors: the box has no colour attribute, so it would read as black.
  const tokenMaterial = new THREE.MeshLambertMaterial({ color: 0xffffff });
  const tokens = new THREE.InstancedMesh(tokenGeometry, tokenMaterial, TOKEN_COUNT);
  tokens.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  tokens.frustumCulled = false;
  parallaxGroup.add(tokens);

  const stageTargets = Array.from({ length: FORMATION_COUNT }, () => new Float32Array(TOKEN_COUNT * 3));
  const stageColors = Array.from({ length: FORMATION_COUNT }, () => new Float32Array(TOKEN_COUNT * 3));
  const currentPositions = new Float32Array(TOKEN_COUNT * 3);
  const currentRotations = new Float32Array(TOKEN_COUNT * 3);
  const mouseDisplaceX = new Float32Array(TOKEN_COUNT);
  const mouseDisplaceY = new Float32Array(TOKEN_COUNT);
  const mouseDisplaceZ = new Float32Array(TOKEN_COUNT);
  const mouseSpinVelocity = new Float32Array(TOKEN_COUNT);

  const colorPen = linear(PALETTE.pen);
  const colorAccent = linear(PALETTE.accent);
  const colorPositive = linear(PALETTE.positive);
  const colorGraphite = linear(PALETTE.graphite);
  const colorRisk = linear(PALETTE.risk);
  const colorSheet = linear(PALETTE.paper);

  function assignColor(target, index, color) {
    target[index] = color.r;
    target[index + 1] = color.g;
    target[index + 2] = color.b;
  }

  function computeFormations() {
    for (let i = 0; i < TOKEN_COUNT; i += 1) {
      const i3 = i * 3;

      // Stage 0: the uploaded table as a gently rippling matrix.
      const col0 = i % 24;
      const row0 = Math.floor(i / 24);
      stageTargets[0][i3] = (col0 - 11.5) * 0.62;
      stageTargets[0][i3 + 1] = (row0 - 14.5) * 0.56;
      stageTargets[0][i3 + 2] = Math.sin(col0 * 0.32 + row0 * 0.22) * 1.5 + ((((i * 17) % 100) / 100) - 0.5) * 0.5;
      assignColor(stageColors[0], i3, i % 5 === 0 ? colorPen : i % 2 === 0 ? colorGraphite : colorSheet);

      // Stage 1: five column profiles with unusual values outside their bounds.
      const barIndex = i % 5;
      const tokenInBar = Math.floor(i / 5);
      const barX = (barIndex - 2) * 3.4;
      let barY = 0;
      let isOutlier = false;
      if (barIndex === 0) {
        barY = -6.5 + (tokenInBar / 144) * 11.5;
      } else if (barIndex === 1) {
        if (tokenInBar >= 126) {
          isOutlier = true;
          barY = 6.4 + (tokenInBar - 126) * 0.38;
        } else {
          barY = -6.8 + Math.pow(tokenInBar / 126, 2) * 12;
        }
      } else if (barIndex === 2) {
        const secondCluster = tokenInBar >= 72;
        const clusterIndex = secondCluster ? tokenInBar - 72 : tokenInBar;
        const center = secondCluster ? 2.5 : -3.5;
        barY = center + ((clusterIndex - 36) / 36) * 2.6;
      } else if (barIndex === 3) {
        barY = -6.5 + (tokenInBar / 144) * 12;
      } else if (tokenInBar >= 128) {
        isOutlier = true;
        barY = 6.6 + (tokenInBar - 128) * 0.36;
      } else if (tokenInBar <= 8) {
        isOutlier = true;
        barY = -7.6 - (8 - tokenInBar) * 0.32;
      } else {
        barY = -6.2 + ((tokenInBar - 8) / 120) * 11.8;
      }
      stageTargets[1][i3] = barX + ((((i * 13) % 100) / 100) - 0.5) * 1.4;
      stageTargets[1][i3 + 1] = barY;
      stageTargets[1][i3 + 2] = ((((i * 19) % 100) / 100) - 0.5) * 1.2;
      assignColor(stageColors[1], i3, isOutlier ? colorRisk : tokenInBar % 3 === 0 ? colorPen : colorGraphite);

      // Stage 2: a central planner linked to four tool clusters.
      let hubX = 0;
      let hubY = 0;
      let hubZ = 0;
      let clusterColor = colorPen;
      if (i >= 80 && i < 240) {
        hubX = -5.6; hubY = 4.4; hubZ = 1.2; clusterColor = colorAccent;
      } else if (i >= 240 && i < 400) {
        hubX = 5.6; hubY = 4.4; hubZ = -1.2; clusterColor = colorPositive;
      } else if (i >= 400 && i < 560) {
        hubX = -5.2; hubY = -4.6; hubZ = -1.5; clusterColor = colorGraphite;
      } else if (i >= 560) {
        hubX = 5.2; hubY = -4.6; hubZ = 1.5; clusterColor = colorAccent;
      }
      const angle = i * 0.45;
      const orbitRadius = 1 + (((i * 23) % 100) / 100) * 2.2;
      stageTargets[2][i3] = hubX + Math.cos(angle) * orbitRadius;
      stageTargets[2][i3 + 1] = hubY + Math.sin(angle) * orbitRadius * 0.8;
      stageTargets[2][i3 + 2] = hubZ + Math.sin(angle * 1.5) * 1.2;
      assignColor(stageColors[2], i3, clusterColor);

      // Stage 3: model core, training fold, and held-out guard ring.
      if (i < 100) {
        const phi = Math.acos(1 - (2 * (i + 0.5)) / 100);
        const theta = Math.PI * (1 + Math.sqrt(5)) * i;
        const radius = 2.4 + (((i * 11) % 100) / 100) * 0.5;
        stageTargets[3][i3] = radius * Math.sin(phi) * Math.cos(theta);
        stageTargets[3][i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
        stageTargets[3][i3 + 2] = radius * Math.cos(phi) * 0.8;
        assignColor(stageColors[3], i3, colorPen);
      } else if (i < 410) {
        const trainAngle = ((i - 100) / 310) * Math.PI * 2;
        const trainRadius = 5.8 + ((((i * 13) % 100) / 100) - 0.5) * 0.6;
        stageTargets[3][i3] = trainRadius * Math.cos(trainAngle);
        stageTargets[3][i3 + 1] = trainRadius * Math.sin(trainAngle);
        stageTargets[3][i3 + 2] = ((((i * 17) % 100) / 100) - 0.5) * 0.8;
        assignColor(stageColors[3], i3, colorPositive);
      } else {
        const testAngle = ((i - 410) / 310) * Math.PI * 2;
        const testRadius = 8.2 + ((((i * 19) % 100) / 100) - 0.5) * 0.7;
        stageTargets[3][i3] = testRadius * Math.cos(testAngle);
        stageTargets[3][i3 + 1] = testRadius * Math.sin(testAngle);
        stageTargets[3][i3 + 2] = ((((i * 23) % 100) / 100) - 0.5) * 0.8;
        assignColor(stageColors[3], i3, i % 2 === 0 ? colorPen : colorGraphite);
      }
    }
  }
  computeFormations();

  // Step 0: before the table forms, the cubes lie scattered. Fixed, so the
  // scene looks the same every visit. Sized to fill the closed window.
  const scatterTargets = new Float32Array(TOKEN_COUNT * 3);
  const unit = (index, salt) => {
    const value = Math.sin(index * 12.9898 + salt * 78.233) * 43758.5453;
    return value - Math.floor(value);
  };
  for (let i = 0; i < TOKEN_COUNT; i += 1) {
    scatterTargets[i * 3] = (unit(i, 1) - 0.5) * 34;
    scatterTargets[i * 3 + 1] = (unit(i, 2) - 0.5) * 17;
    scatterTargets[i * 3 + 2] = (unit(i, 3) - 0.5) * 12;
  }

  const dummy = new THREE.Object3D();
  const tempColor = new THREE.Color();
  for (let i = 0; i < TOKEN_COUNT; i += 1) {
    const i3 = i * 3;
    // Start where step 0 starts (scattered), so the first visible frame is the
    // scatter and the table forms from it. Reduced motion starts as the table.
    const start = reduceMotion ? stageTargets[0] : scatterTargets;
    currentPositions[i3] = start[i3];
    currentPositions[i3 + 1] = start[i3 + 1];
    currentPositions[i3 + 2] = start[i3 + 2];
    tempColor.setRGB(stageColors[0][i3], stageColors[0][i3 + 1], stageColors[0][i3 + 2]);
    tokens.setColorAt(i, tempColor);
    dummy.position.set(currentPositions[i3], currentPositions[i3 + 1], currentPositions[i3 + 2]);
    dummy.rotation.set(0, 0, 0);
    dummy.updateMatrix();
    tokens.setMatrixAt(i, dummy.matrix);
  }
  tokens.instanceColor.needsUpdate = true;
  tokens.instanceMatrix.needsUpdate = true;

  const thresholdMaterial = new THREE.LineDashedMaterial({
    color: linear(PALETTE.risk), dashSize: 0.6, gapSize: 0.4, transparent: true, opacity: 0
  });
  const thresholdLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-8.5, 5.8, 0), new THREE.Vector3(8.5, 5.8, 0)]),
    thresholdMaterial
  );
  thresholdLine.computeLineDistances();
  parallaxGroup.add(thresholdLine);

  const hubPositions = [
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(-5.6, 4.4, 1.2),
    new THREE.Vector3(5.6, 4.4, -1.2),
    new THREE.Vector3(-5.2, -4.6, -1.5),
    new THREE.Vector3(5.2, -4.6, 1.5)
  ];
  const synapsePoints = [];
  for (let hub = 1; hub <= 4; hub += 1) {
    synapsePoints.push(hubPositions[0], hubPositions[hub]);
    synapsePoints.push(hubPositions[hub], hubPositions[(hub % 4) + 1]);
  }
  const synapseMaterial = new THREE.LineBasicMaterial({ color: linear(PALETTE.accent), transparent: true, opacity: 0 });
  const synapses = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(synapsePoints), synapseMaterial);
  parallaxGroup.add(synapses);

  const guardMaterial = new THREE.MeshBasicMaterial({
    color: linear(PALETTE.positive), side: THREE.DoubleSide, transparent: true, opacity: 0
  });
  const guardRing = new THREE.Mesh(new THREE.RingGeometry(8.3, 8.4, 64), guardMaterial);
  parallaxGroup.add(guardRing);

  // Re-colour everything the scene owns after the page theme changes.
  function applyPalette() {
    PALETTE = readPalette();
    colorPen.copy(linear(PALETTE.pen));
    colorAccent.copy(linear(PALETTE.accent));
    colorPositive.copy(linear(PALETTE.positive));
    colorGraphite.copy(linear(PALETTE.graphite));
    colorRisk.copy(linear(PALETTE.risk));
    colorSheet.copy(linear(PALETTE.paper));
    thresholdMaterial.color.copy(linear(PALETTE.risk));
    synapseMaterial.color.copy(linear(PALETTE.accent));
    guardMaterial.color.copy(linear(PALETTE.positive));
    buildGrid();
    computeFormations();
    colorScalar = -1; // forces the next frame to rewrite every instance colour
  }

  const HUD_TITLES = [
    'Your file as rows of a table',
    'Feature distributions and unusual values',
    'A plan routed to Python tools',
    'Checks and the report'
  ];
  const STAGE_SUMMARIES = [
    { tag: 'Stage 1 of 4', sub: 'Each token stands for a row of your table' },
    { tag: 'Stage 2 of 4', sub: 'Distributions, with unusual values marked' },
    { tag: 'Stage 3 of 4', sub: 'One bounded plan routed across Python tools' },
    { tag: 'Stage 4 of 4', sub: 'Held-out checks, then the report' }
  ];

  // ---- sizes: measured on resize only, never per frame ------------------
  const size = { width: 1, height: 1 };
  const cardSize = { width: 1, height: 1 };
  const metrics = { start: 0, length: 1 };
  let navHeight = 0;

  function resizeRenderer() {
    const width = Math.max(1, canvasWrap.clientWidth);
    const height = Math.max(1, canvasWrap.clientHeight);
    if (width === size.width && height === size.height) return;
    size.width = width;
    size.height = height;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    if (leaderCanvas && leaderContext) {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      leaderCanvas.width = Math.round(width * dpr);
      leaderCanvas.height = Math.round(height * dpr);
      leaderContext.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }

  function measure() {
    cardSize.width = Math.max(1, card.offsetWidth);
    cardSize.height = Math.max(1, card.offsetHeight);
    navHeight = nav ? nav.offsetHeight : 0;
    const narrow = window.innerWidth < 760;
    const maxWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--maxw')) || 1220;
    // The closed window must clear the nav, so its vertical inset has a floor.
    collapsed.y = Math.max(cardSize.height * COLLAPSED_INSET_Y, navHeight + 16);
    collapsed.enterX = cardSize.width * (narrow ? ENTER_INSET_X_NARROW : ENTER_INSET_X);
    collapsed.exitX = Math.max(0, (cardSize.width - Math.min(maxWidth, cardSize.width - EXIT_SIDE_GUTTER * 2)) / 2);
    const wrapper = card.parentElement;
    const stickyTop = parseFloat(getComputedStyle(wrapper).top) || 0;
    metrics.start = track.getBoundingClientRect().top + window.scrollY - stickyTop;
    metrics.length = Math.max(1, track.offsetHeight - wrapper.offsetHeight);
    resizeRenderer();
    onScroll();
    appliedNorm = -1; // card size changed: recompute the clip-path next frame
  }

  // ---- scroll model ------------------------------------------------------
  const STAGE_COUNT = 4;
  const HOLD_IN = 0.11;          // the closed window holds this long: the cubes settle into the table (step 0)
  const ENTER_THRESHOLD = 0.26;  // fully open from here...
  const EXIT_THRESHOLD = 0.8;    // ...until here, then it closes
  const HOLD_OUT = 0.05;         // and sits still again before the track scrolls away
  const MORPH_HALF = 0.3;
  const STAGE_HYSTERESIS = 0.05;
  // Closed window, before opening: a centred window about 60% of the card wide.
  const ENTER_INSET_X = 0.2;
  const ENTER_INSET_X_NARROW = 0.07;   // phones
  const COLLAPSED_INSET_Y = 0.2;
  const COLLAPSED_RADIUS = 8;
  // Closed window, after closing: the width of the page's content column,
  // like the audited-entry panel that follows.
  const EXIT_SIDE_GUTTER = 24;
  const CROP_GAP = 12;                 // px between the window and its corner marks
  const SCENE_CLOSED_SCALE = 0.86;     // the scene is a little smaller in the closed window
  const collapsed = { enterX: 0, exitX: 0, y: 0 };

  let targetNorm = 0;        // where the scrollbar is
  let displayNorm = 0;       // what the page shows (eased toward targetNorm)
  let appliedNorm = -1;
  let targetScalar = 0;
  let currentScalar = 0;
  let currentPanelIndex = 0;
  let pendingStageIndex = null;
  let scrollTween = null;
  let expanded = false;
  let pastTrack = false;
  let lastClip = '';
  let lastShift = '';
  let lastFade = '';
  let lastMarkOpacity = '';
  let lastPrelude = '';
  let lastOutroShift = '';
  let lastFill = -1;
  let lastNavShift = '';

  function setNavShift(value) {
    if (!nav || value === lastNavShift) return;
    nav.style.setProperty('--nav-shift', value);
    lastNavShift = value;
  }

  function onScroll() {
    targetNorm = clamp01((window.scrollY - metrics.start) / metrics.length);
  }

  function scalarFromProgress(progress) {
    for (let boundary = 1; boundary < STAGE_COUNT; boundary += 1) {
      if (progress < boundary + MORPH_HALF) {
        const t = clamp01((progress - (boundary - MORPH_HALF)) / (2 * MORPH_HALF));
        return (boundary - 1) + t * t * (3 - 2 * t);
      }
    }
    return STAGE_COUNT - 1;
  }

  function stageIndexWithHysteresis(progress) {
    let index = Math.min(STAGE_COUNT - 1, Math.floor(progress));
    if (index > currentPanelIndex && progress < index + STAGE_HYSTERESIS) index -= 1;
    if (index < currentPanelIndex && progress > index + 1 - STAGE_HYSTERESIS) index += 1;
    return index;
  }

  // ---- panels and stage chrome -------------------------------------------
  const stagePanels = Array.from(document.querySelectorAll('.solare-stage-panel'))
    .sort((a, b) => Number(a.dataset.stage) - Number(b.dataset.stage));
  const stagePills = Array.from(document.querySelectorAll('.stage-nav-pill'));
  const panelPartSelector = '.panel-step-badge, .panel-title, .panel-desc, .panel-facts';
  const panelParts = (panel) => panel.querySelectorAll(panelPartSelector);

  function settlePanel(panel, visible) {
    panel.classList.toggle('active', visible);
    panel.classList.remove('leaving');
    panel.style.opacity = '';
    panel.style.transform = '';
    panelParts(panel).forEach((element) => {
      element.style.opacity = '';
      element.style.transform = '';
    });
  }

  function transitionPanels(fromIndex, toIndex) {
    const anime = window.anime;
    const nextPanel = stagePanels[toIndex];
    const previousPanel = stagePanels[fromIndex];
    if (!nextPanel) return;
    if (anime) {
      stagePanels.forEach((panel) => {
        anime.remove(panel);
        anime.remove(panelParts(panel));
      });
    }
    if (!anime || reduceMotion) {
      stagePanels.forEach((panel, index) => settlePanel(panel, index === toIndex));
      return;
    }

    const direction = toIndex > fromIndex ? 1 : -1;
    const resuming = nextPanel.classList.contains('leaving');
    stagePanels.forEach((panel) => {
      if (panel !== nextPanel && panel !== previousPanel) settlePanel(panel, false);
    });
    if (previousPanel && previousPanel !== nextPanel) {
      previousPanel.classList.remove('active');
      previousPanel.classList.add('leaving');
      anime({
        targets: previousPanel,
        opacity: 0,
        translateY: -direction * 16,
        duration: 220,
        easing: 'easeOutQuad',
        complete: () => settlePanel(previousPanel, false)
      });
    }
    nextPanel.classList.remove('leaving');
    nextPanel.classList.add('active');
    if (resuming) {
      anime({ targets: nextPanel, opacity: 1, translateY: 0, duration: 240, easing: 'easeOutCubic' });
      anime({ targets: panelParts(nextPanel), opacity: 1, translateY: 0, duration: 240, easing: 'easeOutCubic' });
      return;
    }
    const parts = panelParts(nextPanel);
    parts.forEach((element) => {
      element.style.opacity = '0';
      element.style.transform = `translateY(${direction * 16}px)`;
    });
    anime({
      targets: parts,
      opacity: [0, 1],
      translateY: [direction * 16, 0],
      delay: anime.stagger(45, { start: 120 }),
      duration: 520,
      easing: 'cubicBezier(0.16, 1, 0.3, 1)'
    });
  }

  function updateStageUI(stageIndex) {
    const index = Math.max(0, Math.min(STAGE_COUNT - 1, stageIndex));
    if (index === currentPanelIndex) return;
    const previousIndex = currentPanelIndex;
    currentPanelIndex = index;
    stagePills.forEach((pill) => pill.classList.toggle('active', Number(pill.dataset.stage) === index));
    transitionPanels(previousIndex, index);
    const summary = STAGE_SUMMARIES[index];
    if (telemetryTag) telemetryTag.textContent = summary.tag;
    if (telemetrySub) telemetrySub.textContent = summary.sub;
  }

  // ---- card geometry: clip-path + transform only -------------------------
  let lastMarks = '';
  let sceneOpen = 0; // 0 closed window .. 1 full viewport; read by the render loop
  let assembleAmount = 0; // step 0: 0 scattered .. 1 settled into the table
  let lastCueFill = '';
  function setCardGeometry(expansion, exit, closing) {
    const open = 1 - expansion;
    const insetX = open * (closing ? collapsed.exitX : collapsed.enterX);
    const insetY = open * collapsed.y;
    const clip = expansion > 0.9995
      ? 'none'
      : `inset(${Math.round(insetY)}px ${Math.round(insetX)}px round ${(open * COLLAPSED_RADIUS).toFixed(1)}px)`;
    // Full size means the nav is out of the way; it slides back as the card closes.
    setNavShift(`${(-navHeight * smoothstep(0.55, 0.95, expansion)).toFixed(1)}px`);
    sceneOpen = expansion;

    // The closing card lifts a little as it settles into its closed window.
    const lift = exit > 0 ? -36 * exit : 0;
    const shift = lift !== 0 ? `translate3d(0, ${lift.toFixed(1)}px, 0)` : '';
    if (clip !== lastClip) {
      card.style.clipPath = clip;
      lastClip = clip;
    }
    if (shift !== lastShift) {
      card.style.transform = shift;
      lastShift = shift;
    }

    // Viewfinder corners hug the window and spread outward as it opens. They
    // lift with the card, so they stay on its corners while it closes.
    if (crop && cropMarks.length === 4) {
      const x = Math.max(0, insetX - CROP_GAP).toFixed(1);
      const yTop = Math.max(0, insetY - CROP_GAP + lift).toFixed(1);
      const yBottom = Math.max(0, insetY - CROP_GAP - lift).toFixed(1);
      const marks = `${x}|${yTop}|${yBottom}`;
      if (marks !== lastMarks) {
        cropMarks[0].style.transform = `translate3d(${x}px, ${yTop}px, 0)`;
        cropMarks[1].style.transform = `translate3d(-${x}px, ${yTop}px, 0)`;
        cropMarks[2].style.transform = `translate3d(${x}px, -${yBottom}px, 0)`;
        cropMarks[3].style.transform = `translate3d(-${x}px, -${yBottom}px, 0)`;
        lastMarks = marks;
      }
      const markOpacity = (1 - smoothstep(0.3, 0.85, expansion)).toFixed(3);
      if (markOpacity !== lastMarkOpacity) {
        crop.style.opacity = markOpacity;
        lastMarkOpacity = markOpacity;
      }
    }

    // Captions for the closed window. The opening one (the raw table comes in)
    // shows before the card opens; the closing one (the checked report comes out)
    // shows after stage 4. Each fades as the window opens.
    if (prelude && outro) {
      const closedness = 1 - smoothstep(0, 0.3, expansion);
      const text = `${closing ? 0 : closedness.toFixed(3)}|${closing ? closedness.toFixed(3) : 0}`;
      if (text !== lastPrelude) {
        prelude.style.opacity = closing ? '0' : closedness.toFixed(3);
        outro.style.opacity = closing ? closedness.toFixed(3) : '0';
        lastPrelude = text;
      }
      // The closing caption rides with the lifted card, so the gap stays even.
      if (shift !== lastOutroShift) {
        outro.style.transform = shift;
        lastOutroShift = shift;
      }
    }

    // The narrative panel fades in once the card is nearly open and out as it
    // closes, so a sliced panel never shows in the small window.
    const fade = smoothstep(0.6, 0.95, expansion).toFixed(3);
    if (panelWrap && fade !== lastFade) {
      panelWrap.style.opacity = fade;
      lastFade = fade;
    }
  }

  function applyProgress(normalized) {
    if (normalized === appliedNorm) return;
    appliedNorm = normalized;

    const { expansion, exit } = trackPhase(normalized);
    // Step 0 plays out over the opening hold; reduced motion shows the table at once.
    assembleAmount = reduceMotion ? 1 : settle(clamp01(normalized / HOLD_IN));
    if (preludeFill) {
      const fill = assembleAmount.toFixed(3);
      if (fill !== lastCueFill) {
        preludeFill.style.transform = `scaleX(${fill})`;
        lastCueFill = fill;
      }
    }
    const activeProgress = clamp01((normalized - ENTER_THRESHOLD) / (EXIT_THRESHOLD - ENTER_THRESHOLD));
    const progress = activeProgress * STAGE_COUNT;

    setCardGeometry(expansion, exit, normalized > 0.5);
    expanded = expansion > 0.95;
    pastTrack = normalized > EXIT_THRESHOLD;

    if (normalized < ENTER_THRESHOLD) {
      targetScalar = 0;
    } else if (pastTrack) {
      targetScalar = STAGE_COUNT - 1; // the last stage holds while the card closes
    } else {
      targetScalar = scalarFromProgress(progress);
    }

    updateStageUI(stageIndexWithHysteresis(progress));
    if (telemetryFill && Math.abs(activeProgress - lastFill) > 0.002) {
      telemetryFill.style.transform = `scaleX(${activeProgress.toFixed(3)})`;
      lastFill = activeProgress;
    }
  }

  // ---- HUD pin ----------------------------------------------------------
  const hudAnchor = new THREE.Vector3();
  let hudVisible = false;
  function updateHudPin(scalar) {
    if (!hudPin || !hudText || !leaderContext) return;
    const show = expanded && !pastTrack && size.width >= 760;
    if (!show) {
      if (hudVisible) {
        hudPin.style.opacity = '0';
        leaderContext.clearRect(0, 0, size.width, size.height);
        hudVisible = false;
      }
      return;
    }
    hudAnchor.set(0, 9, 0);
    parallaxGroup.localToWorld(hudAnchor);
    hudAnchor.project(camera);
    if (hudAnchor.z > 1) return;
    const screenX = (hudAnchor.x * 0.5 + 0.5) * size.width;
    const screenY = (-hudAnchor.y * 0.5 + 0.5) * size.height;
    const minimumX = Math.max(size.width * 0.52, Math.min(480, size.width * 0.6));
    const pinX = Math.min(Math.max(screenX, minimumX), size.width - 180);
    const pinY = Math.max(screenY - 45, 40);
    hudPin.style.transform = `translate3d(${pinX.toFixed(1)}px, ${pinY.toFixed(1)}px, 0) translate(-50%, -100%)`;
    if (!hudVisible) {
      hudPin.style.opacity = '1';
      hudVisible = true;
    }
    const label = HUD_TITLES[Math.min(HUD_TITLES.length - 1, Math.round(scalar))];
    if (hudText.textContent !== label) hudText.textContent = label;
    leaderContext.clearRect(0, 0, size.width, size.height);
    leaderContext.lineWidth = 1.2;
    leaderContext.strokeStyle = hexToRgba(PALETTE.pen, 0.5);
    leaderContext.fillStyle = PALETTE.pen;
    leaderContext.beginPath();
    leaderContext.arc(screenX, screenY, 3.5, 0, Math.PI * 2);
    leaderContext.fill();
    const elbowY = (screenY + pinY) * 0.5;
    leaderContext.beginPath();
    leaderContext.moveTo(screenX, screenY);
    leaderContext.lineTo(pinX, elbowY);
    leaderContext.lineTo(pinX, pinY);
    leaderContext.stroke();
  }

  // ---- pointer: sampled once per frame, relative to the canvas -----------
  const mouseNormal = new THREE.Vector2(-10, -10);
  const mouseWorld = new THREE.Vector3(-999, -999, 0);
  const localMouse = new THREE.Vector3(-999, -999, 0);
  const raycaster = new THREE.Raycaster();
  const interactionPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  const pointer = { x: 0, y: 0, active: false };
  const targetParallax = { x: 0, y: 0 };
  const currentParallax = { x: 0, y: 0 };

  window.addEventListener('mousemove', (event) => {
    if (reduceMotion) return;
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    pointer.active = true;
    targetParallax.x = (event.clientX / window.innerWidth - 0.5) * 2;
    targetParallax.y = (event.clientY / window.innerHeight - 0.5) * 2;
  }, { passive: true });
  document.addEventListener('mouseleave', () => {
    pointer.active = false;
    targetParallax.x = 0;
    targetParallax.y = 0;
  });

  // ---- navigation --------------------------------------------------------
  function stopScrollTween() {
    if (scrollTween) {
      scrollTween.pause();
      scrollTween = null;
    }
    pendingStageIndex = null;
  }

  function scrollToStage(stageIndex) {
    const index = Math.max(0, Math.min(STAGE_COUNT - 1, stageIndex));
    const stagePosition = ENTER_THRESHOLD + ((index + 0.5) / STAGE_COUNT) * (EXIT_THRESHOLD - ENTER_THRESHOLD);
    const targetY = metrics.start + metrics.length * stagePosition;
    stopScrollTween();
    pendingStageIndex = index;
    if (window.anime && !reduceMotion) {
      const scrollPosition = { y: window.scrollY };
      scrollTween = window.anime({
        targets: scrollPosition,
        y: targetY,
        duration: 800,
        easing: 'cubicBezier(0.25, 0.1, 0.25, 1)',
        update: () => window.scrollTo(0, scrollPosition.y),
        complete: () => {
          scrollTween = null;
          pendingStageIndex = null;
        }
      });
    } else {
      window.scrollTo({ top: targetY, behavior: 'auto' });
      pendingStageIndex = null;
    }
  }
  window.jumpToStage = scrollToStage;

  ['wheel', 'touchstart'].forEach((eventName) => {
    window.addEventListener(eventName, stopScrollTween, { passive: true });
  });
  window.addEventListener('scroll', onScroll, { passive: true });

  const previousButton = document.getElementById('btn-prev-stage');
  const nextButton = document.getElementById('btn-next-stage');
  if (previousButton) previousButton.addEventListener('click', () => scrollToStage(currentPanelIndex - 1));
  if (nextButton) {
    nextButton.addEventListener('click', () => {
      if (currentPanelIndex < STAGE_COUNT - 1) {
        scrollToStage(currentPanelIndex + 1);
      } else {
        const nextSection = document.querySelector('.showcase-sec');
        if (nextSection) {
          const navHeight = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--nav-h')) || 0;
          window.scrollTo({
            top: nextSection.getBoundingClientRect().top + window.scrollY - navHeight,
            behavior: reduceMotion ? 'auto' : 'smooth'
          });
        }
      }
    });
  }

  // Arrow keys step between stages only inside the pinned range, so the keys
  // still scroll normally above, below, and past the first and last stage.
  window.addEventListener('keydown', (event) => {
    const activeTag = document.activeElement ? document.activeElement.tagName : '';
    if (activeTag === 'INPUT' || activeTag === 'TEXTAREA' || activeTag === 'SELECT' || activeTag === 'BUTTON') return;
    const pinned = targetNorm > ENTER_THRESHOLD && targetNorm < EXIT_THRESHOLD;
    if (!pinned) return;
    const baseIndex = pendingStageIndex === null ? currentPanelIndex : pendingStageIndex;
    if ((event.code === 'ArrowDown' || event.code === 'PageDown') && baseIndex < STAGE_COUNT - 1) {
      event.preventDefault();
      scrollToStage(baseIndex + 1);
    } else if ((event.code === 'ArrowUp' || event.code === 'PageUp') && baseIndex > 0) {
      event.preventDefault();
      scrollToStage(baseIndex - 1);
    }
  });

  // ---- render loop -------------------------------------------------------
  const clock = new THREE.Clock();
  let tabVisible = !document.hidden;
  let trackInView = true;
  let colorScalar = -1;

  renderer.setAnimationLoop(() => {
    if (landingSurface && landingSurface.style.display === 'none') {
      setNavShift('0px'); // the workspace surface always shows the nav
      return;
    }
    if (!tabVisible) return;
    const delta = Math.min(clock.getDelta(), 0.04);
    const elapsed = clock.getElapsedTime();

    // Ease the page toward the scrollbar. Frame-rate independent, and the one
    // place scroll becomes motion, so wheel notches never look stepped.
    const follow = reduceMotion ? 1 : 1 - Math.exp(-9 * delta);
    displayNorm += (targetNorm - displayNorm) * follow;
    if (Math.abs(targetNorm - displayNorm) < 0.0002) displayNorm = targetNorm;
    applyProgress(displayNorm);

    if (!trackInView) return;

    currentScalar = targetScalar;
    currentParallax.x += (targetParallax.x - currentParallax.x) * (1 - Math.exp(-4 * delta));
    currentParallax.y += (targetParallax.y - currentParallax.y) * (1 - Math.exp(-4 * delta));

    parallaxGroup.rotation.y = currentParallax.x * 0.035;
    parallaxGroup.rotation.x = -currentParallax.y * 0.035;
    // Closed: the scene sits centred and a little smaller inside the window.
    // Opening: it slides to its dock beside the panel as the window grows.
    const open = settle(sceneOpen);
    parallaxGroup.position.x = dock.stageX * open;
    parallaxGroup.position.y = dock.stageY * open;
    parallaxGroup.scale.setScalar(SCENE_CLOSED_SCALE + (1 - SCENE_CLOSED_SCALE) * open);

    const lastFormation = FORMATION_COUNT - 1;
    const sourceStage = Math.min(lastFormation, Math.floor(currentScalar));
    const targetStage = Math.min(lastFormation, sourceStage + 1);
    const morph = currentScalar - sourceStage;
    const sourcePositions = stageTargets[sourceStage];
    const targetPositions = stageTargets[targetStage];
    const sourceColors = stageColors[sourceStage];
    const targetColors = stageColors[targetStage];

    let mouseOver = false;
    if (!reduceMotion && pointer.active) {
      const rect = canvasWrap.getBoundingClientRect();
      mouseOver = pointer.x >= rect.left && pointer.x <= rect.right && pointer.y >= rect.top && pointer.y <= rect.bottom;
      if (mouseOver) {
        mouseNormal.x = ((pointer.x - rect.left) / rect.width) * 2 - 1;
        mouseNormal.y = -((pointer.y - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouseNormal, camera);
        raycaster.ray.intersectPlane(interactionPlane, mouseWorld);
      }
    }
    localMouse.copy(mouseWorld);
    parallaxGroup.worldToLocal(localMouse);
    const morphDamping = reduceMotion ? 1 : 1 - Math.exp(-4.8 * delta);
    const rotationRelax = Math.exp(-3.2 * delta);
    // Stage 3: the tool network sways about its centre. The same angle turns the
    // lines and every token's target, so the clusters stay on the line ends.
    const nearTools = 1 - Math.min(1, Math.abs(currentScalar - 2));
    const swayAngle = reduceMotion ? 0 : Math.sin(elapsed * 0.45) * 0.34 * nearTools;
    const swayCos = Math.cos(swayAngle);
    const swaySin = Math.sin(swayAngle);
    const influenceRadius = 7.5;
    const influenceRadiusSquared = influenceRadius * influenceRadius;
    const updateColors = Math.abs(currentScalar - colorScalar) > 0.0005;

    for (let i = 0; i < TOKEN_COUNT; i += 1) {
      const i3 = i * 3;
      let targetX = sourcePositions[i3] + (targetPositions[i3] - sourcePositions[i3]) * morph;
      let targetY = sourcePositions[i3 + 1] + (targetPositions[i3 + 1] - sourcePositions[i3 + 1]) * morph;
      let targetZ = sourcePositions[i3 + 2] + (targetPositions[i3 + 2] - sourcePositions[i3 + 2]) * morph;
      if (assembleAmount < 0.999) {
        targetX = scatterTargets[i3] + (targetX - scatterTargets[i3]) * assembleAmount;
        targetY = scatterTargets[i3 + 1] + (targetY - scatterTargets[i3 + 1]) * assembleAmount;
        targetZ = scatterTargets[i3 + 2] + (targetZ - scatterTargets[i3 + 2]) * assembleAmount;
      }
      if (swayAngle !== 0) {
        const swungX = targetX * swayCos + targetZ * swaySin;
        targetZ = targetZ * swayCos - targetX * swaySin;
        targetX = swungX;
      }
      const dx = currentPositions[i3] - localMouse.x;
      const dy = currentPositions[i3 + 1] - localMouse.y;
      const dz = currentPositions[i3 + 2] - localMouse.z;
      const distanceSquared = dx * dx + dy * dy + dz * dz;

      if (mouseOver && distanceSquared < influenceRadiusSquared && distanceSquared > 0.001) {
        const distance = Math.sqrt(distanceSquared);
        const factor = 1 - distance / influenceRadius;
        const force = factor * factor * 4.4;
        mouseDisplaceX[i] = (dx / distance) * force;
        mouseDisplaceY[i] = (dy / distance) * force;
        mouseDisplaceZ[i] = (dz / distance) * force + Math.sin(elapsed * 4 + i * 0.25) * 0.7;
        mouseSpinVelocity[i] = factor * 0.12;
      } else {
        mouseDisplaceX[i] *= 0.88;
        mouseDisplaceY[i] *= 0.88;
        mouseDisplaceZ[i] *= 0.88;
        mouseSpinVelocity[i] *= 0.9;
      }

      currentPositions[i3] += (targetX + mouseDisplaceX[i] - currentPositions[i3]) * morphDamping;
      currentPositions[i3 + 1] += (targetY + mouseDisplaceY[i] - currentPositions[i3 + 1]) * morphDamping;
      currentPositions[i3 + 2] += (targetZ + mouseDisplaceZ[i] - currentPositions[i3 + 2]) * morphDamping;
      // Tokens stay square to the page, like cells of a table. Only the pointer
      // turns them, and they ease back to flat afterwards (no constant tumbling).
      if (!reduceMotion) {
        currentRotations[i3] = (currentRotations[i3] + mouseSpinVelocity[i] * 0.6) * rotationRelax;
        currentRotations[i3 + 1] = (currentRotations[i3 + 1] + mouseSpinVelocity[i] * 1.2) * rotationRelax;
      }

      dummy.position.set(currentPositions[i3], currentPositions[i3 + 1], currentPositions[i3 + 2]);
      dummy.rotation.set(currentRotations[i3], currentRotations[i3 + 1], 0);
      const popScale = 1 + Math.min(Math.abs(mouseSpinVelocity[i]) * 3.5, 0.4);
      dummy.scale.setScalar(popScale);
      dummy.updateMatrix();
      tokens.setMatrixAt(i, dummy.matrix);

      if (updateColors) {
        tempColor.setRGB(
          sourceColors[i3] + (targetColors[i3] - sourceColors[i3]) * morph,
          sourceColors[i3 + 1] + (targetColors[i3 + 1] - sourceColors[i3 + 1]) * morph,
          sourceColors[i3 + 2] + (targetColors[i3 + 2] - sourceColors[i3 + 2]) * morph
        );
        tokens.setColorAt(i, tempColor);
      }
    }
    tokens.instanceMatrix.needsUpdate = true;
    if (updateColors) {
      tokens.instanceColor.needsUpdate = true;
      colorScalar = currentScalar;
    }

    const nearProfile = 1 - Math.min(1, Math.abs(currentScalar - 1));
    const nearChecks = 1 - Math.min(1, Math.abs(currentScalar - 3));
    thresholdMaterial.opacity = nearProfile * 0.5;
    synapseMaterial.opacity = nearTools * 0.45;
    guardMaterial.opacity = nearChecks * 0.45;
    synapses.rotation.y = swayAngle;
    if (!reduceMotion && nearChecks > 0.05) guardRing.rotation.z += 0.0035;

    updateHudPin(currentScalar);
    renderer.render(scene, camera);
  });

  // ---- observers and lifecycle ------------------------------------------
  // Fires on first layout and again when the landing surface is shown after
  // being display:none, which is when the metrics must be re-read.
  if (typeof ResizeObserver === 'function') {
    new ResizeObserver(measure).observe(track);
    new ResizeObserver(measure).observe(card);
  }
  if (typeof IntersectionObserver === 'function') {
    new IntersectionObserver((entries) => {
      trackInView = entries[0].isIntersecting;
    }, { rootMargin: '200px 0px' }).observe(track);
  }
  document.addEventListener('visibilitychange', () => {
    tabVisible = !document.hidden;
    if (tabVisible) clock.getDelta();
  });
  const onMotionPreferenceChange = (event) => {
    reduceMotion = event.matches;
    if (reduceMotion) {
      pointer.active = false;
      targetParallax.x = 0;
      targetParallax.y = 0;
    }
  };
  if (typeof reduceMotionQuery.addEventListener === 'function') {
    reduceMotionQuery.addEventListener('change', onMotionPreferenceChange);
  } else if (typeof reduceMotionQuery.addListener === 'function') {
    reduceMotionQuery.addListener(onMotionPreferenceChange);
  }
  window.addEventListener('resize', () => {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    updateDocking();
    measure();
  }, { passive: true });

  // toggleTheme() sets data-theme first, so the tokens already hold the new values.
  window.__solareThree = { updateTheme: applyPalette, remeasure: measure };

  measure();
  displayNorm = targetNorm;
  applyProgress(displayNorm);
})();
