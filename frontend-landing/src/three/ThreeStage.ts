/**
 * ThreeStage.ts — Modular 3D Scene Controller
 * Implements the 12 Invariants from 3d-ui-skill with native continuous scroll tracking.
 * 
 * Features:
 * - Single clock managed via renderer.setAnimationLoop()
 * - Exponential frame-rate independent dt-damping
 * - Native scroll progress scalar s in [0, 3] with continuous interpolation
 * - Tangible data tokens (BoxGeometry InstancedMesh, TOKEN_COUNT = 720)
 * - 4 Continuous Formations:
 *     0: Raw Tabular Matrix (CSV Undulation)
 *     1: Statistical Histograms with IQR 1.5x Outlier Bounds (Flagged in Risk Red)
 *     2: Recursive Multi-Agent Reasoning Hub & Swarm Synapses
 *     3: Certified Model Core & Generalization Guard Envelope (<10% Gap)
 * - Interactive 3D cursor deflection & spin impulse via raycasting
 * - Leader lines and projected HUD callout pin
 * - Zero per-frame memory allocation
 * - Day / Night Warm Ledger Theme switching
 */

import * as THREE from 'three';
import type { ThemeMode, ThemePalette, ThreeMotionTokens, ThreeStageMetrics } from '../types/landing';

export const MOTION_TOKENS: ThreeMotionTokens = {
  dur: {
    micro: 180,
    ui: 320,
    stage: 750,
    cinematic: 1400,
  },
  damp: {
    cursor: 28.0,
    journey: 6.0,
    velocity: 4.0,
    parallax: 4.0,
  },
  bounds: {
    maxParallaxDeg: 4.5,
    maxCursorImpulse: 4.4,
  },
};

export const LEDGER_PALETTES: Record<ThemeMode, ThemePalette> = {
  night: {
    stock: '#241c14',
    sheet: '#1c1610',
    sheetAlt: '#282017',
    ink: '#f6eedf',
    graphite: '#bdae97',
    pen: '#f0a24a',
    penGlow: 'rgba(240, 162, 74, 0.45)',
    risk: '#e2685a',
    positive: '#7fb77e',
    accent: '#4fc3f7',
    grid: '#4a3c28',
    cardBg: 'rgba(28, 22, 16, 0.88)',
    cardBorder: 'rgba(240, 162, 74, 0.24)',
    fogDensity: 0.016,
    exposure: 1.15,
    gridOpacity: 0.12,
  },
  day: {
    stock: '#f7eedd',
    sheet: '#fffbf2',
    sheetAlt: '#f1e4cb',
    ink: '#3a2b1e',
    graphite: '#6a5948',
    pen: '#a34f20',
    penGlow: 'rgba(163, 79, 32, 0.25)',
    risk: '#a33526',
    positive: '#3d753c',
    accent: '#d47a28',
    grid: '#e4d4bc',
    cardBg: 'rgba(255, 251, 242, 0.95)',
    cardBorder: 'rgba(138, 118, 96, 0.28)',
    fogDensity: 0.014,
    exposure: 1.0,
    gridOpacity: 0.10,
  },
};

export class ThreeStageController {
  private canvas: HTMLCanvasElement;
  private leaderCanvas: HTMLCanvasElement | null;
  private leaderCtx: CanvasRenderingContext2D | null;
  private hudPinEl: HTMLElement | null;
  private hudTagEl: HTMLElement | null;

  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private clock: THREE.Clock;

  private worldGroup: THREE.Group;
  private parallaxGroup: THREE.Group;
  private gridHelper: THREE.GridHelper;

  private ambientLight: THREE.AmbientLight;
  private keyLight: THREE.DirectionalLight;
  private fillLight: THREE.DirectionalLight;

  private tokenMesh: THREE.InstancedMesh;
  private thresholdLine: THREE.Line;
  private thresholdMat: THREE.LineDashedMaterial;
  private synapseMesh: THREE.LineSegments;
  private synapseMat: THREE.LineBasicMaterial;
  private guardMesh: THREE.Mesh;
  private guardMat: THREE.MeshBasicMaterial;

  public static readonly TOKEN_COUNT = 720;
  private currentPositions = new Float32Array(ThreeStageController.TOKEN_COUNT * 3);
  private targetPositions = new Float32Array(ThreeStageController.TOKEN_COUNT * 3);
  private currentRotations = new Float32Array(ThreeStageController.TOKEN_COUNT * 3);

  private mouseDisplaceX = new Float32Array(ThreeStageController.TOKEN_COUNT);
  private mouseDisplaceY = new Float32Array(ThreeStageController.TOKEN_COUNT);
  private mouseDisplaceZ = new Float32Array(ThreeStageController.TOKEN_COUNT);
  private mouseSpinVel = new Float32Array(ThreeStageController.TOKEN_COUNT);

  private stageTargets: Float32Array[] = [];
  private stageColors: Float32Array[] = [];

  // Scratch objects for zero per-frame allocation
  private dummy = new THREE.Object3D();
  private tempVec = new THREE.Vector3();
  private tempColor = new THREE.Color();
  private mouseWorld = new THREE.Vector3(-999, -999, 0);
  private localMouse = new THREE.Vector3(-999, -999, 0);
  private mouseNorm = new THREE.Vector2(-10, -10);
  private raycaster = new THREE.Raycaster();
  private interactionPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);

  // State
  private theme: ThemeMode = 'night';
  private scalarProgress = 0;
  private targetScalarProgress = 0;
  private isVisible = true;
  private prefersReducedMotion = false;

  private curParallax = { x: 0, y: 0 };
  private targetParallax = { x: 0, y: 0 };
  private isMouseActive = false;

  private readonly hudTitles = [
    'RAW DATASET // UNSTRUCTURED TABULAR STREAM',
    'FEATURE DISTRIBUTIONS // PROFILE & OUTLIER BOUNDS',
    'MULTI-AGENT GRAPH // RECURSIVE REASONING NETWORK',
    'CERTIFIED LEDGER // SEALED GENERALIZATION BOUND',
  ];

  constructor(
    canvas: HTMLCanvasElement,
    leaderCanvas: HTMLCanvasElement | null,
    hudPinEl: HTMLElement | null,
    hudTagEl: HTMLElement | null,
    initialTheme: ThemeMode = 'night'
  ) {
    this.canvas = canvas;
    this.leaderCanvas = leaderCanvas;
    this.leaderCtx = leaderCanvas ? leaderCanvas.getContext('2d') : null;
    this.hudPinEl = hudPinEl;
    this.hudTagEl = hudTagEl;
    this.theme = initialTheme;

    const pal = LEDGER_PALETTES[this.theme];
    const initialBgColor = new THREE.Color(pal.stock);

    // Invariant 6: Color management declared once
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setClearColor(initialBgColor, 1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = pal.exposure;

    this.scene = new THREE.Scene();
    this.scene.background = initialBgColor;
    this.scene.fog = new THREE.FogExp2(initialBgColor.getHex(), pal.fogDensity);

    this.camera = new THREE.PerspectiveCamera(46, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.camera.position.set(0, 0, 42);

    this.clock = new THREE.Clock();

    this.worldGroup = new THREE.Group();
    this.scene.add(this.worldGroup);

    this.parallaxGroup = new THREE.Group();
    this.worldGroup.add(this.parallaxGroup);

    // Responsive 3D placement: docked right on desktop, centered on mobile
    this.updateStagePlacement();

    // Floor Grid
    this.gridHelper = new THREE.GridHelper(56, 44, new THREE.Color(pal.grid), new THREE.Color(pal.grid));
    this.gridHelper.position.set(0, -14, 0);
    this.gridHelper.material.transparent = true;
    this.gridHelper.material.opacity = pal.gridOpacity;
    this.worldGroup.add(this.gridHelper);

    // Lights
    this.ambientLight = new THREE.AmbientLight(pal.stock, 1.4);
    this.scene.add(this.ambientLight);

    this.keyLight = new THREE.DirectionalLight(0xfffaed, 2.0);
    this.keyLight.position.set(15, 25, 20);
    this.scene.add(this.keyLight);

    this.fillLight = new THREE.DirectionalLight(pal.accent, 0.9);
    this.fillLight.position.set(-20, -10, -10);
    this.scene.add(this.fillLight);

    // Geometry: BoxGeometry for tangible data tokens
    const tokenGeo = new THREE.BoxGeometry(0.42, 0.42, 0.16);
    const tokenMat = new THREE.MeshStandardMaterial({
      roughness: 0.35,
      metalness: 0.18,
      flatShading: true,
    });
    this.tokenMesh = new THREE.InstancedMesh(tokenGeo, tokenMat, ThreeStageController.TOKEN_COUNT);
    this.tokenMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.parallaxGroup.add(this.tokenMesh);

    // Auxiliary geometry: Threshold line, Synapses, Guard ring
    const thresholdPoints = [new THREE.Vector3(-8.5, 5.8, 0), new THREE.Vector3(8.5, 5.8, 0)];
    const thresholdGeo = new THREE.BufferGeometry().setFromPoints(thresholdPoints);
    this.thresholdMat = new THREE.LineDashedMaterial({
      color: new THREE.Color(pal.risk),
      dashSize: 0.6,
      gapSize: 0.4,
      transparent: true,
      opacity: 0,
    });
    this.thresholdLine = new THREE.Line(thresholdGeo, this.thresholdMat);
    this.thresholdLine.computeLineDistances();
    this.parallaxGroup.add(this.thresholdLine);

    const hubPositions = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(-5.6, 4.4, 1.2),
      new THREE.Vector3(5.6, 4.4, -1.2),
      new THREE.Vector3(-5.2, -4.6, -1.5),
      new THREE.Vector3(5.2, -4.6, 1.5),
    ];
    const synapsePoints: THREE.Vector3[] = [];
    for (let h = 1; h <= 4; h++) {
      synapsePoints.push(hubPositions[0], hubPositions[h]);
      synapsePoints.push(hubPositions[h], hubPositions[(h % 4) + 1]);
    }
    const synapseGeo = new THREE.BufferGeometry().setFromPoints(synapsePoints);
    this.synapseMat = new THREE.LineBasicMaterial({
      color: new THREE.Color(pal.accent),
      transparent: true,
      opacity: 0,
    });
    this.synapseMesh = new THREE.LineSegments(synapseGeo, this.synapseMat);
    this.parallaxGroup.add(this.synapseMesh);

    const guardGeo = new THREE.RingGeometry(8.3, 8.4, 64);
    this.guardMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(pal.positive),
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0,
    });
    this.guardMesh = new THREE.Mesh(guardGeo, this.guardMat);
    this.parallaxGroup.add(this.guardMesh);

    // Compute Formations
    this.initFormations();

    // Check reduced motion
    const motionMedia = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.prefersReducedMotion = motionMedia.matches;
    motionMedia.addEventListener('change', (e) => {
      this.prefersReducedMotion = e.matches;
    });

    // Lifecycle events
    document.addEventListener('visibilitychange', () => {
      this.isVisible = !document.hidden;
    });

    window.addEventListener('resize', () => this.handleResize());
    window.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    window.addEventListener('mouseleave', () => this.handleMouseLeave());

    this.resizeLeaderCanvas();
    this.startLoop();
  }

  private updateStagePlacement(): void {
    const isWide = window.innerWidth >= 900;
    this.parallaxGroup.position.x = isWide ? 10.8 : 0;
    this.parallaxGroup.position.y = isWide ? 0.2 : -2.4;
  }

  private initFormations(): void {
    const pal = LEDGER_PALETTES[this.theme];
    const cPen = new THREE.Color(pal.pen);
    const cAccent = new THREE.Color(pal.accent);
    const cPositive = new THREE.Color(pal.positive);
    const cGraphite = new THREE.Color(pal.graphite);
    const cRisk = new THREE.Color(pal.risk);
    const cSheet = new THREE.Color(pal.sheetAlt);

    for (let s = 0; s < 4; s++) {
      this.stageTargets[s] = new Float32Array(ThreeStageController.TOKEN_COUNT * 3);
      this.stageColors[s] = new Float32Array(ThreeStageController.TOKEN_COUNT * 3);
    }

    for (let i = 0; i < ThreeStageController.TOKEN_COUNT; i++) {
      const i3 = i * 3;

      // Stage 0: Unparsed Tabular CSV Matrix Grid
      const col0 = i % 24;
      const row0 = Math.floor(i / 24);
      const gridX = (col0 - 11.5) * 0.62;
      const gridY = (row0 - 14.5) * 0.56;
      const waveZ = Math.sin(col0 * 0.32 + row0 * 0.22) * 1.5 + (((i * 17) % 100) / 100 - 0.5) * 0.5;

      this.stageTargets[0][i3] = gridX;
      this.stageTargets[0][i3 + 1] = gridY;
      this.stageTargets[0][i3 + 2] = waveZ;

      const colS0 = i % 5 === 0 ? cPen : i % 2 === 0 ? cGraphite : cSheet;
      this.stageColors[0][i3] = colS0.r;
      this.stageColors[0][i3 + 1] = colS0.g;
      this.stageColors[0][i3 + 2] = colS0.b;

      // Stage 1: Feature Column Profiles & Outlier Bounds
      const barIdx = i % 5;
      const tokenInBar = Math.floor(i / 5);
      const barX = (barIdx - 2) * 3.4;
      let barY = 0;
      let isOutlier = false;

      if (barIdx === 0) {
        barY = -6.5 + (tokenInBar / 144.0) * 11.5;
      } else if (barIdx === 1) {
        if (tokenInBar >= 126) {
          isOutlier = true;
          barY = 6.4 + (tokenInBar - 126) * 0.38;
        } else {
          const ratio = tokenInBar / 126.0;
          barY = -6.8 + Math.pow(ratio, 2.0) * 12.0;
        }
      } else if (barIdx === 2) {
        const isClusterB = tokenInBar >= 72;
        const subIdx = isClusterB ? tokenInBar - 72 : tokenInBar;
        const center = isClusterB ? 2.5 : -3.5;
        barY = center + ((subIdx - 36) / 36.0) * 2.6;
      } else if (barIdx === 3) {
        barY = -6.5 + (tokenInBar / 144.0) * 12.0;
      } else {
        if (tokenInBar >= 128) {
          isOutlier = true;
          barY = 6.6 + (tokenInBar - 128) * 0.36;
        } else if (tokenInBar <= 8) {
          isOutlier = true;
          barY = -7.6 - (8 - tokenInBar) * 0.32;
        } else {
          barY = -6.2 + ((tokenInBar - 8) / 120.0) * 11.8;
        }
      }

      const jitX = (((i * 13) % 100) / 100 - 0.5) * 1.4;
      const jitZ = (((i * 19) % 100) / 100 - 0.5) * 1.2;

      this.stageTargets[1][i3] = barX + jitX;
      this.stageTargets[1][i3 + 1] = barY;
      this.stageTargets[1][i3 + 2] = jitZ;

      const colS1 = isOutlier ? cRisk : tokenInBar % 3 === 0 ? cPen : cGraphite;
      this.stageColors[1][i3] = colS1.r;
      this.stageColors[1][i3 + 1] = colS1.g;
      this.stageColors[1][i3 + 2] = colS1.b;

      // Stage 2: Recursive Multi-Agent Reasoning Graph
      const hubCenter = new THREE.Vector3(0, 0, 0);
      let clusterColor = cPen;
      if (i < 80) {
        hubCenter.set(0, 0, 0);
        clusterColor = cPen;
      } else if (i < 240) {
        hubCenter.set(-5.6, 4.4, 1.2);
        clusterColor = cAccent;
      } else if (i < 400) {
        hubCenter.set(5.6, 4.4, -1.2);
        clusterColor = cPositive;
      } else if (i < 560) {
        hubCenter.set(-5.2, -4.6, -1.5);
        clusterColor = cPen;
      } else {
        hubCenter.set(5.2, -4.6, 1.5);
        clusterColor = cAccent;
      }

      const angle = i * 0.45;
      const orbitR = 1.0 + (((i * 23) % 100) / 100) * 2.2;
      this.stageTargets[2][i3] = hubCenter.x + Math.cos(angle) * orbitR;
      this.stageTargets[2][i3 + 1] = hubCenter.y + Math.sin(angle) * orbitR * 0.8;
      this.stageTargets[2][i3 + 2] = hubCenter.z + Math.sin(angle * 1.5) * 1.2;

      this.stageColors[2][i3] = clusterColor.r;
      this.stageColors[2][i3 + 1] = clusterColor.g;
      this.stageColors[2][i3 + 2] = clusterColor.b;

      // Stage 3: Generalization Shield & Certified Model
      if (i < 100) {
        const phiC = Math.acos(1 - (2 * (i + 0.5)) / 100);
        const thetaC = Math.PI * (1 + Math.sqrt(5)) * i;
        const rC = 2.4 + (((i * 11) % 100) / 100) * 0.5;
        this.stageTargets[3][i3] = rC * Math.sin(phiC) * Math.cos(thetaC);
        this.stageTargets[3][i3 + 1] = rC * Math.sin(phiC) * Math.sin(thetaC);
        this.stageTargets[3][i3 + 2] = rC * Math.cos(phiC) * 0.8;

        this.stageColors[3][i3] = cPen.r;
        this.stageColors[3][i3 + 1] = cPen.g;
        this.stageColors[3][i3 + 2] = cPen.b;
      } else if (i < 410) {
        const tRingIdx = i - 100;
        const angTrain = (tRingIdx / 310.0) * Math.PI * 2.0;
        const rTrain = 5.8 + (((i * 13) % 100) / 100 - 0.5) * 0.6;
        this.stageTargets[3][i3] = rTrain * Math.cos(angTrain);
        this.stageTargets[3][i3 + 1] = rTrain * Math.sin(angTrain);
        this.stageTargets[3][i3 + 2] = (((i * 17) % 100) / 100 - 0.5) * 0.8;

        this.stageColors[3][i3] = cPositive.r;
        this.stageColors[3][i3 + 1] = cPositive.g;
        this.stageColors[3][i3 + 2] = cPositive.b;
      } else {
        const testIdx = i - 410;
        const angTest = (testIdx / 310.0) * Math.PI * 2.0;
        const rTest = 8.2 + (((i * 19) % 100) / 100 - 0.5) * 0.7;
        this.stageTargets[3][i3] = rTest * Math.cos(angTest);
        this.stageTargets[3][i3 + 1] = rTest * Math.sin(angTest);
        this.stageTargets[3][i3 + 2] = (((i * 23) % 100) / 100 - 0.5) * 0.8;

        const col = i % 2 === 0 ? cPen : cGraphite;
        this.stageColors[3][i3] = col.r;
        this.stageColors[3][i3 + 1] = col.g;
        this.stageColors[3][i3 + 2] = col.b;
      }

      // Initial positions
      this.currentPositions[i3] = this.stageTargets[0][i3];
      this.currentPositions[i3 + 1] = this.stageTargets[0][i3 + 1];
      this.currentPositions[i3 + 2] = this.stageTargets[0][i3 + 2];

      this.tempColor.setRGB(this.stageColors[0][i3], this.stageColors[0][i3 + 1], this.stageColors[0][i3 + 2]);
      this.tokenMesh.setColorAt(i, this.tempColor);

      this.dummy.position.set(this.currentPositions[i3], this.currentPositions[i3 + 1], this.currentPositions[i3 + 2]);
      this.dummy.rotation.set(0.1, 0.2, 0);
      this.dummy.scale.set(1, 1, 1);
      this.dummy.updateMatrix();
      this.tokenMesh.setMatrixAt(i, this.dummy.matrix);
    }

    this.tokenMesh.instanceColor!.needsUpdate = true;
    this.tokenMesh.instanceMatrix.needsUpdate = true;
  }

  /**
   * Continuous Progress Scalar Controller (Invariant 3)
   * Maps scroll or narrative position to s in [0, 3] with smooth interpolation.
   */
  public setProgressScalar(scalar: number): void {
    this.targetScalarProgress = Math.max(0, Math.min(3, scalar));
  }

  public setTheme(mode: ThemeMode): void {
    if (this.theme === mode) return;
    this.theme = mode;
    const pal = LEDGER_PALETTES[this.theme];

    const stockC = new THREE.Color(pal.stock);
    this.renderer.setClearColor(stockC, 1);
    this.scene.background = stockC;
    this.scene.fog!.color.copy(stockC);
    (this.scene.fog as THREE.FogExp2).density = pal.fogDensity;
    this.renderer.toneMappingExposure = pal.exposure;

    this.gridHelper.material.color.set(pal.grid);
    this.gridHelper.material.opacity = pal.gridOpacity;

    this.ambientLight.color.set(stockC);
    this.keyLight.color.set(mode === 'night' ? 0xfffaed : 0xfffae8);
    this.fillLight.color.set(pal.accent);

    this.thresholdMat.color.set(pal.risk);
    this.synapseMat.color.set(pal.accent);
    this.guardMat.color.set(pal.positive);

    // Recompute colors
    this.initFormations();
    this.updateTokenColors(this.scalarProgress);
  }

  private updateTokenColors(scalar: number): void {
    const s0 = Math.floor(scalar);
    const s1 = Math.min(3, s0 + 1);
    const t = scalar - s0;

    const colsA = this.stageColors[s0];
    const colsB = this.stageColors[s1];

    for (let i = 0; i < ThreeStageController.TOKEN_COUNT; i++) {
      const i3 = i * 3;
      const r = colsA[i3] + (colsB[i3] - colsA[i3]) * t;
      const g = colsA[i3 + 1] + (colsB[i3 + 1] - colsA[i3 + 1]) * t;
      const b = colsA[i3 + 2] + (colsB[i3 + 2] - colsA[i3 + 2]) * t;

      this.tempColor.setRGB(r, g, b);
      this.tokenMesh.setColorAt(i, this.tempColor);
    }
    this.tokenMesh.instanceColor!.needsUpdate = true;
  }

  private handleMouseMove(e: MouseEvent): void {
    this.isMouseActive = true;
    this.targetParallax.x = (e.clientX / window.innerWidth - 0.5) * 2;
    this.targetParallax.y = (e.clientY / window.innerHeight - 0.5) * 2;

    this.mouseNorm.x = (e.clientX / window.innerWidth) * 2 - 1;
    this.mouseNorm.y = -(e.clientY / window.innerHeight) * 2 + 1;
    this.raycaster.setFromCamera(this.mouseNorm, this.camera);
    this.raycaster.ray.intersectPlane(this.interactionPlane, this.mouseWorld);
  }

  private handleMouseLeave(): void {
    this.isMouseActive = false;
    this.targetParallax.x = 0;
    this.targetParallax.y = 0;
  }

  private handleResize(): void {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.updateStagePlacement();
    this.resizeLeaderCanvas();
  }

  private resizeLeaderCanvas(): void {
    if (!this.leaderCanvas || !this.leaderCtx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    this.leaderCanvas.width = window.innerWidth * dpr;
    this.leaderCanvas.height = window.innerHeight * dpr;
    this.leaderCtx.scale(dpr, dpr);
  }

  private updateHudOverlay(): void {
    if (!this.hudPinEl || !this.hudTagEl) return;

    this.tempVec.set(0, 9.0, 0);
    this.parallaxGroup.localToWorld(this.tempVec);
    this.tempVec.project(this.camera);

    if (this.tempVec.z > 1.0) {
      this.hudPinEl.style.opacity = '0';
      if (this.leaderCtx) this.leaderCtx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      return;
    }

    const screenX = (this.tempVec.x * 0.5 + 0.5) * window.innerWidth;
    const screenY = (-(this.tempVec.y * 0.5) + 0.5) * window.innerHeight;

    const minPinX = Math.max(window.innerWidth * 0.54, 500);
    const pinnedX = Math.min(Math.max(screenX, minPinX), window.innerWidth - 180);
    const pinnedY = Math.max(screenY - 45, 88);

    // Zero layout thrash: GPU transform only
    this.hudPinEl.style.transform = `translate3d(${pinnedX.toFixed(1)}px, ${pinnedY.toFixed(1)}px, 0) translate(-50%, -100%)`;
    this.hudPinEl.style.opacity = '1';

    const activeStageIndex = Math.min(3, Math.round(this.scalarProgress));
    this.hudTagEl.textContent = this.hudTitles[activeStageIndex];

    if (this.leaderCtx) {
      this.leaderCtx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      this.leaderCtx.lineWidth = 1.2;
      this.leaderCtx.strokeStyle = this.theme === 'night' ? 'rgba(240, 162, 74, 0.45)' : 'rgba(163, 79, 32, 0.35)';
      this.leaderCtx.fillStyle = this.theme === 'night' ? '#f0a24a' : '#a34f20';

      this.leaderCtx.beginPath();
      this.leaderCtx.arc(screenX, screenY, 3.5, 0, Math.PI * 2);
      this.leaderCtx.fill();

      const elbowY = (screenY + pinnedY) * 0.5;
      this.leaderCtx.beginPath();
      this.leaderCtx.moveTo(screenX, screenY);
      this.leaderCtx.lineTo(pinnedX, elbowY);
      this.leaderCtx.lineTo(pinnedX, pinnedY);
      this.leaderCtx.stroke();
    }
  }

  private startLoop(): void {
    // Invariant 1: Single clock via renderer.setAnimationLoop
    this.renderer.setAnimationLoop(() => {
      if (!this.isVisible) return;

      // Invariant 2: Frame-rate independent dt damping
      const dt = Math.min(this.clock.getDelta(), 0.04);
      const t = this.clock.getElapsedTime();

      // Smooth journey scalar progress (Invariant 3)
      const kJourney = 1.0 - Math.exp(-MOTION_TOKENS.damp.journey * dt);
      this.scalarProgress += (this.targetScalarProgress - this.scalarProgress) * kJourney;

      // Parallax
      if (!this.prefersReducedMotion) {
        const kParallax = 1.0 - Math.exp(-MOTION_TOKENS.damp.parallax * dt);
        this.curParallax.x += (this.targetParallax.x - this.curParallax.x) * kParallax;
        this.curParallax.y += (this.targetParallax.y - this.curParallax.y) * kParallax;
        this.parallaxGroup.rotation.y = this.curParallax.x * 0.035;
        this.parallaxGroup.rotation.x = -this.curParallax.y * 0.035;
      }

      // Compute formation interpolation between floor(s) and ceil(s)
      const s0 = Math.floor(this.scalarProgress);
      const s1 = Math.min(3, s0 + 1);
      const morphT = this.scalarProgress - s0;

      const posA = this.stageTargets[s0];
      const posB = this.stageTargets[s1];

      // Interactive mouse deflection
      this.localMouse.copy(this.mouseWorld);
      this.parallaxGroup.worldToLocal(this.localMouse);

      const kMorph = 1.0 - Math.exp(-4.8 * dt);
      const influenceRadius = 7.5;
      const influenceRadiusSq = influenceRadius * influenceRadius;

      for (let i = 0; i < ThreeStageController.TOKEN_COUNT; i++) {
        const i3 = i * 3;

        // Continuous formation target
        const targetX = posA[i3] + (posB[i3] - posA[i3]) * morphT;
        const targetY = posA[i3 + 1] + (posB[i3 + 1] - posA[i3 + 1]) * morphT;
        const targetZ = posA[i3 + 2] + (posB[i3 + 2] - posA[i3 + 2]) * morphT;

        const dx = this.currentPositions[i3] - this.localMouse.x;
        const dy = this.currentPositions[i3 + 1] - this.localMouse.y;
        const dz = this.currentPositions[i3 + 2] - this.localMouse.z;
        const distSq = dx * dx + dy * dy + dz * dz;

        if (this.isMouseActive && distSq < influenceRadiusSq && distSq > 0.001) {
          const dist = Math.sqrt(distSq);
          const factor = 1.0 - dist / influenceRadius;
          const force = factor * factor * MOTION_TOKENS.bounds.maxCursorImpulse;

          this.mouseDisplaceX[i] = (dx / dist) * force;
          this.mouseDisplaceY[i] = (dy / dist) * force;
          this.mouseDisplaceZ[i] = (dz / dist) * force + Math.sin(t * 4.0 + i * 0.25) * 0.7;
          this.mouseSpinVel[i] = factor * 0.12;
        } else {
          this.mouseDisplaceX[i] *= 0.88;
          this.mouseDisplaceY[i] *= 0.88;
          this.mouseDisplaceZ[i] *= 0.88;
          this.mouseSpinVel[i] *= 0.90;
        }

        const finalTargetX = targetX + this.mouseDisplaceX[i];
        const finalTargetY = targetY + this.mouseDisplaceY[i];
        const finalTargetZ = targetZ + this.mouseDisplaceZ[i];

        this.currentPositions[i3] += (finalTargetX - this.currentPositions[i3]) * kMorph;
        this.currentPositions[i3 + 1] += (finalTargetY - this.currentPositions[i3 + 1]) * kMorph;
        this.currentPositions[i3 + 2] += (finalTargetZ - this.currentPositions[i3 + 2]) * kMorph;

        this.currentRotations[i3] += this.mouseSpinVel[i] * 0.6 + 0.002;
        this.currentRotations[i3 + 1] += this.mouseSpinVel[i] * 1.2 + 0.003;

        this.dummy.position.set(this.currentPositions[i3], this.currentPositions[i3 + 1], this.currentPositions[i3 + 2]);
        this.dummy.rotation.set(this.currentRotations[i3], this.currentRotations[i3 + 1], 0);

        const scalePop = 1.0 + Math.min(Math.abs(this.mouseSpinVel[i]) * 3.5, 0.4);
        this.dummy.scale.set(scalePop, scalePop, scalePop);

        this.dummy.updateMatrix();
        this.tokenMesh.setMatrixAt(i, this.dummy.matrix);
      }

      this.tokenMesh.instanceMatrix.needsUpdate = true;
      this.updateTokenColors(this.scalarProgress);

      // Auxiliary geometry opacity
      const nearStage1 = 1.0 - Math.min(1.0, Math.abs(this.scalarProgress - 1.0));
      const nearStage2 = 1.0 - Math.min(1.0, Math.abs(this.scalarProgress - 2.0));
      const nearStage3 = 1.0 - Math.min(1.0, Math.abs(this.scalarProgress - 3.0));

      this.thresholdMat.opacity = nearStage1 * 0.38;
      this.synapseMat.opacity = nearStage2 * 0.35;
      this.guardMat.opacity = nearStage3 * 0.35;

      if (nearStage2 > 0.05) this.synapseMesh.rotation.y += 0.0025;
      if (nearStage3 > 0.05) this.guardMesh.rotation.z += 0.0035;

      this.updateHudOverlay();
      this.renderer.render(this.scene, this.camera);
    });
  }

  public dispose(): void {
    this.renderer.setAnimationLoop(null);
    this.renderer.dispose();
    this.tokenMesh.geometry.dispose();
    (this.tokenMesh.material as THREE.Material).dispose();
    this.thresholdLine.geometry.dispose();
    this.thresholdMat.dispose();
    this.synapseMesh.geometry.dispose();
    this.synapseMat.dispose();
    this.guardMesh.geometry.dispose();
    this.guardMat.dispose();
  }
}
