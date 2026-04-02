# Three-Body Problem Simulation

A real-time, browser-based 3D simulation of the gravitational three-body problem with spacetime curvature visualization.

## Running

Requires a local HTTP server (ES module `importmap` does not work over `file://`):

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080` in any modern browser.

---

## What It Shows

Three celestial bodies orbit each other under mutual Newtonian gravity. The blue wireframe grid beneath them represents spacetime curvature — each body pulls the grid downward in proportion to its mass and proximity, forming gravitational wells. As the bodies move, the wells move with them in real time.

The system is **chaotic**: tiny differences in starting conditions lead to wildly different long-term trajectories. Every reset produces a unique simulation.

---

## Files

| File | Purpose |
|---|---|
| `index.html` | Page shell, CDN import map, HUD markup, styles |
| `simulation.js` | All physics, 3D scene, animation loop, and UI logic |

---

## Physics

### Integration: Runge-Kutta 4 (RK4)

The equations of motion are integrated using the classic fourth-order Runge-Kutta method. RK4 is used instead of simpler Euler integration because Euler accumulates energy error rapidly — orbits either collapse or spiral outward within seconds. RK4 keeps energy drift low enough for visually stable simulations.

Each frame runs **8 sub-steps** of RK4 at `dt = 0.005 / 8` for additional stability.

### Gravitational Force

Newton's law of universal gravitation between bodies `i` and `j`:

```
F = G * mᵢ * mⱼ / (r² + ε²)^(3/2)  ×  (rⱼ - rᵢ)
```

The **softening parameter** `ε = 0.5` prevents the force from approaching infinity when two bodies pass close to each other, avoiding numerical blow-up.

### Initial Conditions

Random runs use a **constrained randomization** strategy:

1. Bodies are placed at random angles on a circle, with ±30% radius jitter.
2. Each body is given a velocity tangent to the circle, scaled near the circular orbit speed.
3. The **center-of-mass velocity is subtracted** from all bodies, so the system has zero net momentum and stays centered.
4. Small random z-offsets give true 3D motion.

This avoids the common failure mode of fully random initial conditions, which almost always cause immediate ejection of one body.

**Figure-8 mode** loads the exact Chenciner-Montgomery (2000) choreography — a mathematically proven stable periodic orbit where all three equal masses chase each other along a figure-eight curve.

### Spacetime Grid

The grid is a `PlaneGeometry(64×64, 100 segments)` lying flat in the XZ plane (~10,000 vertices). Every frame, each vertex is displaced downward by the combined Newtonian gravitational potential:

```
y = clamp( Σᵢ  −G·mᵢ / √(Δx² + Δz² + ε²),  −14,  0 )
```

This is computed on the CPU. At ~10,000 vertices and 3 bodies it runs comfortably at 60 fps on modern hardware. The grid epsilon (`ε = 1.5`) is larger than the physics epsilon to produce smooth, visually clean wells.

---

## Controls

### Camera

| Action | Result |
|---|---|
| Left-drag | Orbit around the scene |
| Right-drag / two-finger drag | Pan |
| Scroll / pinch | Zoom |

### HUD Panel (bottom-left)

| Control | Description |
|---|---|
| **Speed** | Simulation speed multiplier (0.1× – 5×). Does not affect physics accuracy — sub-stepping adapts. |
| **Gravity G** | Scales the gravitational constant. Higher values produce tighter, faster orbits and deeper grid wells. |
| **Trail** | Controls how many past positions are drawn per body. At the maximum (`∞`) the full trajectory from the start of the simulation is shown. All history is retained even when the slider is reduced — pulling it back up reveals the hidden path. |
| **Mass 1/2/3** | Individual body masses (0.5 – 5). Changes take effect immediately on the next physics step. |
| **◎** (per body) | Focus the camera on that body. The orbit controls recentre on the moving body so you can watch its local motion. Click again to release. |
| **Reset** | Randomizes starting positions, velocities, and masses. Clears all trails and releases camera focus. |
| **Pause / Resume** | Freezes the simulation. The scene remains interactive (camera still works). |
| **Figure-8** | Loads the Chenciner-Montgomery stable orbit. All three bodies have equal mass and follow the same closed figure-eight path indefinitely (in the absence of numerical drift). |

---

## Implementation Notes

### Trail Ring Buffer

Each body maintains a `Float32Array` ring buffer of up to 20,000 positions. `addPoint` is O(1) — it writes into the next slot and advances the head pointer. `updateVisible(n)` copies the last `n` points into a contiguous GPU buffer using `subarray` + `set`, handling the wrap-around case. This runs every frame and costs a single `memcpy`-equivalent regardless of history length.

### Camera Focus

When a body is focused, `OrbitControls.target` is lerped toward the body's current world position each frame (`lerp factor = 0.08`). This gives a smooth follow that does not jerk when the body accelerates. The user retains full orbit/zoom control around the moving target.

### Glow Effect

Each body has a `THREE.Sprite` child with a radial-gradient `CanvasTexture` and `AdditiveBlending`. No post-processing or render passes are used — the effect is purely composited in the forward pass.

### Lighting

Each body carries a `THREE.PointLight` as a child mesh. As bodies move, their lights move with them, dynamically illuminating the grid beneath each gravitational well.

---

## Dependencies

Loaded entirely from CDN via `importmap` — no `npm install` or build step.

| Library | Version | Purpose |
|---|---|---|
| [Three.js](https://threejs.org) | 0.158.0 | 3D rendering (WebGL) |
| Three.js OrbitControls | 0.158.0 | Camera interaction |
