# LAST ZONE — standalone battle royale prototype

A modular third-person 3D battle royale prototype for web and Android, built with Three.js and Capacitor.

## Structure
- `src/core`: deterministic math, gameplay configuration, input and synthesized audio.
- `src/world`: terrain, lighting, environment geometry, collision and safe-zone boundary.
- `src/game`: actor model/animation, projectile simulation, bot combat AI and match lifecycle.
- `index.html`: landscape HUD, lobby and touch controls.

## Run locally
1. Install Node.js 20 or newer.
2. Run `npm install`.
3. Run `npm run check`.
4. Serve the repository root over HTTP and open it in a modern browser.

The browser uses an import map for the pinned Three.js 0.160.0 module. The Android workflow copies the source tree into the Capacitor web directory and bundles Three.js locally.

## Controls
- PC: WASD, mouse look, left click fire, right click aim, R reload, Q switch weapon, Shift sprint.
- Touch: left virtual stick to move, drag the right side to look, FIRE/AIM/SPRINT buttons, reload and weapon switch.

## Current implementation scope
The project has a hand-built low-poly soldier rig, a procedural terrain and prop layout, third-person camera, hitscan-free projectile travel with swept collision tests, weapon-specific damage/fire cadence/reload values, 24 independent AI opponents, line-of-sight checks, a shrinking zone, minimap and synthesized Web Audio gunshots/UI sounds.

This is an original prototype, not a PUBG asset or source-code copy. It is not yet a production-ready PUBG-scale game: no online multiplayer/server authority, full navmesh pathfinding, interior traversal, vehicle physics, parachute phase, inventory/loot economy, motion-captured animations or recorded weapon sound library are included.