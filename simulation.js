import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ─── Constants ────────────────────────────────────────────────────────────────

const G_BASE       = 1.0;
const DT_BASE      = 0.005;
const SOFTENING    = 0.5;       // physics softening
const GRID_EPS     = 1.5;       // grid distortion softening
const WELL_SCALE   = 1.0;
const MAX_DIP      = -14;
const MAX_HISTORY  = 20000;     // maximum trail points stored per body
const SUBSTEPS     = 8;
const GRID_SEGS    = 100;       // 101×101 = ~10k vertices
const GRID_SIZE    = 64;

const BODY_CFG = [
  { color: 0xff4400, emissive: 0xff2200, radius: 0.9, spinSpeed: 0.4 },
  { color: 0x4488ff, emissive: 0x1133ee, radius: 0.65, spinSpeed: 0.65 },
  { color: 0xffcc00, emissive: 0xffaa00, radius: 0.75, spinSpeed: 0.5  },
];

// ─── State ────────────────────────────────────────────────────────────────────

let renderer, scene, camera, controls, clock;
let gridMesh, starField;
let bodies = [];      // { pos, vel, mass, mesh, trail, light }
let trails = [];      // Trail objects
let paused = false;
let simTime = 0;
let currentG = G_BASE;
let speedMult = 1.0;
let trailVisible = 350;   // how many trail points to show (MAX_HISTORY = all)
let focusedBody = null;   // null | 0 | 1 | 2

// Canvas / paint mode
let canvasMode = false;
let brushStyle  = 0;      // 0=round 1=calligraphy 2=spray 3=ink 4=marker
let paintCtx    = null;
let lastScreenPos = [];   // THREE.Vector2 per body, screen coords

const BRUSH_COLORS = ['#cc2200', '#1155ee', '#cc8800'];

// ─── Physics ──────────────────────────────────────────────────────────────────

function computeDerivatives(state, masses, G) {
  const n = masses.length;
  const deriv = new Float64Array(state.length);

  for (let i = 0; i < n; i++) {
    const ix = i * 6, ivx = i * 6 + 3;
    // velocity → position derivative
    deriv[ix]     = state[ivx];
    deriv[ix + 1] = state[ivx + 1];
    deriv[ix + 2] = state[ivx + 2];

    // acceleration from all other bodies
    let ax = 0, ay = 0, az = 0;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const jx = j * 6;
      const dx = state[jx]     - state[ix];
      const dy = state[jx + 1] - state[ix + 1];
      const dz = state[jx + 2] - state[ix + 2];
      const r2 = dx*dx + dy*dy + dz*dz + SOFTENING * SOFTENING;
      const r3 = r2 * Math.sqrt(r2);
      const f = G * masses[j] / r3;
      ax += f * dx;
      ay += f * dy;
      az += f * dz;
    }
    deriv[ivx]     = ax;
    deriv[ivx + 1] = ay;
    deriv[ivx + 2] = az;
  }
  return deriv;
}

function rk4Step(bodies, dt, G) {
  const n = bodies.length;
  const masses = bodies.map(b => b.mass);

  // pack state
  const s0 = new Float64Array(n * 6);
  for (let i = 0; i < n; i++) {
    s0[i*6]   = bodies[i].pos.x;
    s0[i*6+1] = bodies[i].pos.y;
    s0[i*6+2] = bodies[i].pos.z;
    s0[i*6+3] = bodies[i].vel.x;
    s0[i*6+4] = bodies[i].vel.y;
    s0[i*6+5] = bodies[i].vel.z;
  }

  const k1 = computeDerivatives(s0, masses, G);

  const s1 = new Float64Array(n * 6);
  for (let i = 0; i < n * 6; i++) s1[i] = s0[i] + 0.5 * dt * k1[i];
  const k2 = computeDerivatives(s1, masses, G);

  const s2 = new Float64Array(n * 6);
  for (let i = 0; i < n * 6; i++) s2[i] = s0[i] + 0.5 * dt * k2[i];
  const k3 = computeDerivatives(s2, masses, G);

  const s3 = new Float64Array(n * 6);
  for (let i = 0; i < n * 6; i++) s3[i] = s0[i] + dt * k3[i];
  const k4 = computeDerivatives(s3, masses, G);

  // apply
  for (let i = 0; i < n; i++) {
    const s = i * 6;
    bodies[i].pos.x = s0[s]   + dt/6 * (k1[s]   + 2*k2[s]   + 2*k3[s]   + k4[s]);
    bodies[i].pos.y = s0[s+1] + dt/6 * (k1[s+1] + 2*k2[s+1] + 2*k3[s+1] + k4[s+1]);
    bodies[i].pos.z = s0[s+2] + dt/6 * (k1[s+2] + 2*k2[s+2] + 2*k3[s+2] + k4[s+2]);
    bodies[i].vel.x = s0[s+3] + dt/6 * (k1[s+3] + 2*k2[s+3] + 2*k3[s+3] + k4[s+3]);
    bodies[i].vel.y = s0[s+4] + dt/6 * (k1[s+4] + 2*k2[s+4] + 2*k3[s+4] + k4[s+4]);
    bodies[i].vel.z = s0[s+5] + dt/6 * (k1[s+5] + 2*k2[s+5] + 2*k3[s+5] + k4[s+5]);
  }
}

function randomInitialConditions() {
  const masses = BODY_CFG.map(() => 1.0 + Math.random() * 2.0);
  const totalMass = masses.reduce((a, b) => a + b, 0);
  const R = 7 + Math.random() * 3;

  const positions = [];
  const velocities = [];

  for (let i = 0; i < 3; i++) {
    const angle = (i / 3) * Math.PI * 2 + (Math.random() - 0.5) * 0.8;
    const r = R * (0.7 + Math.random() * 0.6);
    const x = Math.cos(angle) * r;
    const z = Math.sin(angle) * r;
    const y = (Math.random() - 0.5) * 2;
    positions.push(new THREE.Vector3(x, y, z));

    const vScale = Math.sqrt(G_BASE * totalMass / r) * (0.7 + Math.random() * 0.4);
    const vx = -Math.sin(angle) * vScale;
    const vz =  Math.cos(angle) * vScale;
    const vy = (Math.random() - 0.5) * 0.5;
    velocities.push(new THREE.Vector3(vx, vy, vz));
  }

  // zero net momentum
  const cmVel = new THREE.Vector3();
  for (let i = 0; i < 3; i++) {
    cmVel.addScaledVector(velocities[i], masses[i]);
  }
  cmVel.divideScalar(totalMass);
  for (let i = 0; i < 3; i++) {
    velocities[i].sub(cmVel);
  }

  // zero center of mass position
  const cmPos = new THREE.Vector3();
  for (let i = 0; i < 3; i++) {
    cmPos.addScaledVector(positions[i], masses[i]);
  }
  cmPos.divideScalar(totalMass);
  for (let i = 0; i < 3; i++) {
    positions[i].sub(cmPos);
  }

  return { masses, positions, velocities };
}

function figure8InitialConditions() {
  // Chenciner-Montgomery figure-8 (scaled to our coordinate system)
  const s = 6;
  const positions = [
    new THREE.Vector3(-0.97000436 * s, 0, 0.24308753 * s),
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3( 0.97000436 * s, 0, -0.24308753 * s),
  ];
  const v3x = 0.93240737 * s * 0.18, v3z = 0.86473146 * s * 0.18;
  const velocities = [
    new THREE.Vector3( v3x * 0.5, 0,  v3z * 0.5),
    new THREE.Vector3(-v3x,       0, -v3z      ),
    new THREE.Vector3( v3x * 0.5, 0,  v3z * 0.5),
  ];
  return {
    masses: [1.0, 1.0, 1.0],
    positions,
    velocities,
  };
}

// ─── Glow Texture ─────────────────────────────────────────────────────────────

function makeGlowTexture(size = 128) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  grad.addColorStop(0,   'rgba(255,255,255,1)');
  grad.addColorStop(0.2, 'rgba(255,255,255,0.6)');
  grad.addColorStop(0.5, 'rgba(255,255,255,0.15)');
  grad.addColorStop(1,   'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

// ─── Paint / Canvas Mode ──────────────────────────────────────────────────────

function worldToScreen(pos) {
  const v = new THREE.Vector3(pos.x, pos.y, pos.z);
  v.project(camera);
  return new THREE.Vector2(
    (v.x + 1) / 2 * window.innerWidth,
    (-v.y + 1) / 2 * window.innerHeight
  );
}

function clearPaintCanvas() {
  const pc = document.getElementById('paintCanvas');
  if (!paintCtx) return;
  paintCtx.clearRect(0, 0, pc.width, pc.height);
  paintCtx.fillStyle = '#ffffff';
  paintCtx.fillRect(0, 0, pc.width, pc.height);
}

function drawBrushStroke(idx, from, to) {
  const ctx = paintCtx;
  const color = BRUSH_COLORS[idx];
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const dist = Math.hypot(dx, dy);
  if (dist < 0.5) return;

  ctx.save();

  switch (brushStyle) {
    case 0: // Round
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = color;
      ctx.lineWidth = 5;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.globalAlpha = 0.85;
      ctx.stroke();
      break;

    case 1: { // Calligraphy — width varies with angle
      const angle = Math.atan2(dy, dx);
      const w = Math.abs(Math.sin(angle - Math.PI / 4)) * 12 + 1;
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = color;
      ctx.lineWidth = w;
      ctx.lineCap = 'round';
      ctx.globalAlpha = 0.88;
      ctx.stroke();
      break;
    }

    case 2: { // Spray — scattered dots around path
      const steps = Math.max(1, Math.floor(dist / 4));
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.35;
      for (let s = 0; s <= steps; s++) {
        const t = s / steps;
        const cx = from.x + dx * t;
        const cy = from.y + dy * t;
        const r = 12;
        for (let k = 0; k < 12; k++) {
          const a = Math.random() * Math.PI * 2;
          const rr = Math.random() * r;
          ctx.beginPath();
          ctx.arc(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr, 1 + Math.random() * 1.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      break;
    }

    case 3: // Ink — thin with glow
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.lineCap = 'round';
      ctx.globalAlpha = 0.95;
      ctx.shadowBlur = 5;
      ctx.shadowColor = color;
      ctx.stroke();
      break;

    case 4: // Marker — wide, semi-transparent
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = color;
      ctx.lineWidth = 16;
      ctx.lineCap = 'square';
      ctx.globalAlpha = 0.22;
      ctx.stroke();
      break;
  }

  ctx.restore();
}

function toggleCanvasMode() {
  canvasMode = !canvasMode;
  const pc = document.getElementById('paintCanvas');
  const brushPanel = document.getElementById('brushPanel');
  const btn = document.getElementById('canvasModeBtn');

  if (canvasMode) {
    pc.width  = window.innerWidth;
    pc.height = window.innerHeight;
    pc.style.width  = window.innerWidth  + 'px';
    pc.style.height = window.innerHeight + 'px';
    pc.style.display = 'block';
    paintCtx = pc.getContext('2d');
    clearPaintCanvas();
    lastScreenPos = bodies.map(b => worldToScreen(b.pos));

    scene.background = new THREE.Color(0xffffff);
    scene.fog = null;
    gridMesh.visible = false;
    starField.visible = false;
    // dim body lights in canvas mode
    bodies.forEach(b => b.mesh.children.forEach(c => { if (c.isPointLight) c.intensity = 0; }));

    brushPanel.style.display = 'block';
    btn.textContent = 'SPACE MODE';
    btn.classList.add('active');
  } else {
    pc.style.display = 'none';
    paintCtx = null;

    scene.background = new THREE.Color(0x000008);
    scene.fog = new THREE.FogExp2(0x000010, 0.008);
    gridMesh.visible = true;
    starField.visible = true;
    bodies.forEach(b => b.mesh.children.forEach(c => { if (c.isPointLight) c.intensity = 2.5; }));

    brushPanel.style.display = 'none';
    btn.textContent = 'CANVAS MODE';
    btn.classList.remove('active');
  }
}

// ─── Trail ────────────────────────────────────────────────────────────────────

class Trail {
  constructor(color) {
    // Ring buffer storing all history up to MAX_HISTORY points
    this.hist = new Float32Array(MAX_HISTORY * 3);
    this.head  = 0;   // next write slot
    this.total = 0;   // total points stored (capped at MAX_HISTORY)

    // Contiguous buffer written to GPU each frame
    this.buf = new Float32Array(MAX_HISTORY * 3);

    this.geo = new THREE.BufferGeometry();
    const attr = new THREE.BufferAttribute(this.buf, 3);
    attr.setUsage(THREE.DynamicDrawUsage);
    this.geo.setAttribute('position', attr);
    this.geo.setDrawRange(0, 0);

    this.mat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.6,
      depthWrite: false,
    });
    this.line = new THREE.Line(this.geo, this.mat);
  }

  addPoint(v) {
    this.hist[this.head * 3]     = v.x;
    this.hist[this.head * 3 + 1] = v.y;
    this.hist[this.head * 3 + 2] = v.z;
    this.head  = (this.head + 1) % MAX_HISTORY;
    this.total = Math.min(this.total + 1, MAX_HISTORY);
  }

  // Call every frame with the desired visible count
  updateVisible(n) {
    const show = Math.min(n, this.total);
    if (show === 0) {
      this.geo.setDrawRange(0, 0);
      return;
    }
    // The oldest visible point is `show` slots before head (mod MAX_HISTORY)
    const start = (this.head - show + MAX_HISTORY * 2) % MAX_HISTORY;
    const hist = this.hist;
    const buf  = this.buf;

    if (start + show <= MAX_HISTORY) {
      buf.set(hist.subarray(start * 3, (start + show) * 3), 0);
    } else {
      const firstPart = MAX_HISTORY - start;
      buf.set(hist.subarray(start * 3), 0);
      buf.set(hist.subarray(0, (show - firstPart) * 3), firstPart * 3);
    }
    this.geo.setDrawRange(0, show);
    this.geo.attributes.position.needsUpdate = true;
  }

  reset() {
    this.head  = 0;
    this.total = 0;
    this.geo.setDrawRange(0, 0);
  }
}

// ─── Scene Setup ──────────────────────────────────────────────────────────────

function initRenderer() {
  const canvas = document.getElementById('canvas');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
}

function initScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000008);
  scene.fog = new THREE.FogExp2(0x000010, 0.008);
}

function initCamera() {
  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 28, 38);
  camera.lookAt(0, 0, 0);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.minDistance = 8;
  controls.maxDistance = 120;
  controls.target.set(0, 0, 0);
}

function initGrid() {
  const geo = new THREE.PlaneGeometry(GRID_SIZE, GRID_SIZE, GRID_SEGS, GRID_SEGS);
  geo.rotateX(-Math.PI / 2);

  const mat = new THREE.MeshStandardMaterial({
    color: 0x0033aa,
    wireframe: true,
    transparent: true,
    opacity: 0.35,
    emissive: 0x001155,
    emissiveIntensity: 0.8,
  });

  gridMesh = new THREE.Mesh(geo, mat);
  scene.add(gridMesh);
}

function initStarfield() {
  const count = 3000;
  const pos = new Float32Array(count * 3);
  for (let i = 0; i < count * 3; i++) {
    pos[i] = (Math.random() - 0.5) * 500;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.25, sizeAttenuation: true });
  starField = new THREE.Points(geo, mat);
  scene.add(starField);
}

function initBodies(ic) {
  // Remove old bodies from scene
  bodies.forEach(b => {
    scene.remove(b.mesh);
  });
  trails.forEach(t => scene.remove(t.line));

  bodies = [];
  trails = [];

  const glowTex = makeGlowTexture();

  for (let i = 0; i < 3; i++) {
    const cfg = BODY_CFG[i];
    const mass = ic.masses[i];

    // Body mesh
    const geo = new THREE.SphereGeometry(cfg.radius, 32, 32);
    const mat = new THREE.MeshStandardMaterial({
      color: cfg.color,
      emissive: cfg.emissive,
      emissiveIntensity: 0.7,
      roughness: 0.3,
      metalness: 0.1,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(ic.positions[i]);
    scene.add(mesh);

    // Glow sprite
    const spriteMat = new THREE.SpriteMaterial({
      map: glowTex,
      color: cfg.color,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const glow = new THREE.Sprite(spriteMat);
    glow.scale.set(cfg.radius * 7, cfg.radius * 7, 1);
    mesh.add(glow);

    // Point light
    const light = new THREE.PointLight(cfg.color, 2.5, 35);
    mesh.add(light);

    // Trail
    const trail = new Trail(cfg.color);
    scene.add(trail.line);

    bodies.push({
      pos: ic.positions[i].clone(),
      vel: ic.velocities[i].clone(),
      mass,
      mesh,
      spinSpeed: cfg.spinSpeed,
      trail,
    });
    trails.push(trail);
  }

  // Ambient light
  scene.children
    .filter(c => c.isAmbientLight)
    .forEach(c => scene.remove(c));
  scene.add(new THREE.AmbientLight(0x111133, 0.5));

  simTime = 0;

  // Sync mass sliders
  for (let i = 0; i < 3; i++) {
    const slider = document.getElementById(`mass${i}`);
    slider.value = bodies[i].mass.toFixed(1);
    document.getElementById(`mass${i}-val`).textContent = (+slider.value).toFixed(1);
  }
}

// ─── Grid Update ──────────────────────────────────────────────────────────────

function updateGridDistortion() {
  const geo = gridMesh.geometry;
  const posAttr = geo.attributes.position;
  const count = posAttr.count;

  for (let vi = 0; vi < count; vi++) {
    const x = posAttr.getX(vi);
    const z = posAttr.getZ(vi);

    let phi = 0;
    for (const b of bodies) {
      const dx = x - b.pos.x;
      const dz = z - b.pos.z;
      const r = Math.sqrt(dx*dx + dz*dz + GRID_EPS * GRID_EPS);
      phi += -currentG * b.mass / r;
    }

    const y = Math.max(phi * WELL_SCALE, MAX_DIP);
    posAttr.setY(vi, y);
  }

  posAttr.needsUpdate = true;
  geo.computeVertexNormals();
}

// ─── Animation Loop ───────────────────────────────────────────────────────────

let trailFrameCounter = 0;

function animate() {
  requestAnimationFrame(animate);

  if (!paused) {
    const delta = Math.min(clock.getDelta(), 0.05); // cap for tab-switch spikes
    const dt = DT_BASE * speedMult;

    for (let s = 0; s < SUBSTEPS; s++) {
      rk4Step(bodies, dt / SUBSTEPS, currentG);
    }

    simTime += dt;

    // Update body meshes
    for (const b of bodies) {
      b.mesh.position.copy(b.pos);
      b.mesh.rotation.y += b.spinSpeed * delta;
    }

    // Add trail points every other frame to avoid oversampling
    trailFrameCounter++;
    if (trailFrameCounter % 2 === 0) {
      for (const b of bodies) {
        b.trail.addPoint(b.pos);
      }
    }

    updateGridDistortion();

    document.getElementById('time-display').textContent = `T = ${simTime.toFixed(2)}`;
  } else {
    clock.getDelta(); // drain delta so it doesn't spike on unpause
  }

  // Update trail visibility (every frame, reflects slider changes instantly)
  for (const b of bodies) {
    b.trail.updateVisible(canvasMode ? 0 : trailVisible);
  }

  // Paint brush strokes on 2D canvas
  if (canvasMode && paintCtx) {
    for (let i = 0; i < bodies.length; i++) {
      const cur = worldToScreen(bodies[i].pos);
      drawBrushStroke(i, lastScreenPos[i], cur);
      lastScreenPos[i] = cur;
    }
  }

  // Smoothly track focused body
  if (focusedBody !== null && bodies[focusedBody]) {
    controls.target.lerp(bodies[focusedBody].pos, 0.08);
  }

  controls.update();
  renderer.render(scene, camera);
}

// ─── UI ───────────────────────────────────────────────────────────────────────

function setFocus(idx) {
  const focusBtns = [0, 1, 2].map(i => document.getElementById(`focus${i}`));
  if (focusedBody === idx) {
    // clicking same button again = release focus
    focusedBody = null;
    focusBtns.forEach(b => b.classList.remove('active'));
  } else {
    focusedBody = idx;
    focusBtns.forEach((b, i) => b.classList.toggle('active', i === idx));
  }
}

function bindUI() {
  const speedSlider = document.getElementById('speed');
  const gSlider     = document.getElementById('gSlider');
  const trailSlider = document.getElementById('trailSlider');
  const pauseBtn    = document.getElementById('pauseBtn');
  const resetBtn    = document.getElementById('resetBtn');
  const fig8Btn     = document.getElementById('fig8Btn');

  speedSlider.addEventListener('input', () => {
    speedMult = parseFloat(speedSlider.value);
    document.getElementById('speed-val').textContent = speedMult.toFixed(1);
  });

  gSlider.addEventListener('input', () => {
    currentG = parseFloat(gSlider.value);
    document.getElementById('g-val').textContent = currentG.toFixed(1);
  });

  trailSlider.addEventListener('input', () => {
    const v = parseInt(trailSlider.value);
    trailVisible = v;
    const label = v >= MAX_HISTORY ? '∞' : String(v);
    document.getElementById('trail-val').textContent = label;
  });

  for (let i = 0; i < 3; i++) {
    const slider = document.getElementById(`mass${i}`);
    const valEl  = document.getElementById(`mass${i}-val`);
    slider.addEventListener('input', () => {
      bodies[i].mass = parseFloat(slider.value);
      valEl.textContent = bodies[i].mass.toFixed(1);
    });
    document.getElementById(`focus${i}`).addEventListener('click', () => setFocus(i));
  }

  document.getElementById('canvasModeBtn').addEventListener('click', toggleCanvasMode);

  document.getElementById('clearPaintBtn').addEventListener('click', clearPaintCanvas);

  for (let i = 0; i < 5; i++) {
    document.getElementById(`brush${i}`).addEventListener('click', () => {
      brushStyle = i;
      document.querySelectorAll('.brush-btn').forEach((b, j) => b.classList.toggle('active', j === i));
    });
  }

  pauseBtn.addEventListener('click', () => {
    paused = !paused;
    pauseBtn.textContent = paused ? 'RESUME' : 'PAUSE';
    pauseBtn.classList.toggle('active', paused);
  });

  resetBtn.addEventListener('click', () => {
    focusedBody = null;
    [0, 1, 2].forEach(i => document.getElementById(`focus${i}`).classList.remove('active'));
    initBodies(randomInitialConditions());
    fig8Btn.classList.remove('active');
  });

  fig8Btn.addEventListener('click', () => {
    initBodies(figure8InitialConditions());
    fig8Btn.classList.add('active');
  });

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    if (canvasMode) {
      const pc = document.getElementById('paintCanvas');
      pc.width  = window.innerWidth;
      pc.height = window.innerHeight;
      pc.style.width  = window.innerWidth  + 'px';
      pc.style.height = window.innerHeight + 'px';
      clearPaintCanvas();
    }
  });
}

// ─── Entry Point ──────────────────────────────────────────────────────────────

function main() {
  initRenderer();
  initScene();
  initCamera();
  initGrid();
  initStarfield();
  initBodies(randomInitialConditions());
  bindUI();
  clock = new THREE.Clock();
  animate();
}

main();
