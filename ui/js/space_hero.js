/**
 * SatQuery AI — Cinematic Orbital Space Hero Engine
 * Features 3D revolving satellites along dotted constellation orbits,
 * scanning radar pulses, and a hyper-drive warp transition on "ENTER".
 */
class SpaceHeroEngine {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    
    this.width = 0;
    this.height = 0;
    this.particles = [];
    this.satellites = [];
    this.orbitDottedNodes = [];
    this.radarPulse = 0;
    this.time = 0;
    
    // Zoom & Warp state
    this.isWarping = false;
    this.warpProgress = 0;
    this.zoomFactor = 1.0;
    this.warpSpeedMultiplier = 1.0;
    
    // Mouse parallax
    this.mouseX = 0;
    this.mouseY = 0;
    this.targetMouseX = 0;
    this.targetMouseY = 0;

    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    this.init();
  }

  init() {
    this.resize();
    window.addEventListener('resize', () => this.resize());
    window.addEventListener('mousemove', (e) => {
      this.targetMouseX = (e.clientX / window.innerWidth - 0.5) * 45;
      this.targetMouseY = (e.clientY / window.innerHeight - 0.5) * 45;
    });

    this.createParticles();
    this.createSatellites();
    this.createDottedNodes();

    if (!this.reducedMotion) {
      requestAnimationFrame(() => this.loop());
    } else {
      this.drawStaticFrame();
    }
  }

  resize() {
    this.width = this.canvas.width = window.innerWidth;
    this.height = this.canvas.height = window.innerHeight;
  }

  createParticles() {
    this.particles = [];
    const count = Math.min(150, Math.floor(window.innerWidth / 10));
    for (let i = 0; i < count; i++) {
      this.particles.push({
        x: Math.random() * this.width,
        y: Math.random() * this.height,
        z: Math.random() * 2 + 0.5,
        radius: Math.random() * 2.0 + 0.4,
        alpha: Math.random() * 0.7 + 0.2,
        speed: Math.random() * 0.4 + 0.1
      });
    }
  }

  createSatellites() {
    this.satellites = [
      { angle: 0, radiusX: 310, radiusY: 140, tilt: -0.2, speed: 0.012, color: '#00f2fe', label: 'SENTINEL-2A', size: 6 },
      { angle: Math.PI * 0.65, radiusX: 380, radiusY: 170, tilt: 0.35, speed: 0.008, color: '#4facfe', label: 'LANDSAT-9', size: 7 },
      { angle: Math.PI * 1.3, radiusX: 240, radiusY: 110, tilt: -0.5, speed: 0.016, color: '#00ffaa', label: 'EOS-SAR', size: 5 },
      { angle: Math.PI * 1.85, radiusX: 450, radiusY: 200, tilt: 0.15, speed: 0.006, color: '#ffcc00', label: 'RISAT-1B', size: 6 }
    ];
  }

  createDottedNodes() {
    // Generate constellation dotted nodes along the orbits
    this.orbitDottedNodes = [];
    for (let i = 0; i < 64; i++) {
      this.orbitDottedNodes.push({
        angleOffset: (i / 64) * Math.PI * 2,
        pulseOffset: Math.random() * Math.PI * 2
      });
    }
  }

  triggerEnterWarp(onComplete) {
    this.isWarping = true;
    let duration = 1200; // ms
    let startTime = performance.now();

    const animateWarp = (now) => {
      let elapsed = now - startTime;
      let progress = Math.min(1, elapsed / duration);
      this.warpProgress = progress;

      // Accelerate revolve speed and zoom inward
      this.warpSpeedMultiplier = 1.0 + progress * 15.0;
      this.zoomFactor = 1.0 + progress * 2.5;

      if (progress < 1) {
        requestAnimationFrame(animateWarp);
      } else {
        if (onComplete) onComplete();
      }
    };

    requestAnimationFrame(animateWarp);
  }

  resetWarp() {
    this.isWarping = false;
    this.warpProgress = 0;
    this.zoomFactor = 1.0;
    this.warpSpeedMultiplier = 1.0;
  }

  loop() {
    this.time += 0.015;
    this.radarPulse = (this.radarPulse + 1.5) % 360;

    // Smooth mouse parallax
    this.mouseX += (this.targetMouseX - this.mouseX) * 0.05;
    this.mouseY += (this.targetMouseY - this.mouseY) * 0.05;

    this.draw();
    requestAnimationFrame(() => this.loop());
  }

  draw() {
    const ctx = this.ctx;
    const cx = this.width / 2 + this.mouseX;
    const cy = this.height / 2 + this.mouseY;

    ctx.clearRect(0, 0, this.width, this.height);

    // Deep Space Radial Background
    const bgGradient = ctx.createRadialGradient(cx, cy, 40, cx, cy, Math.max(this.width, this.height) * 0.85);
    bgGradient.addColorStop(0, 'rgba(11, 24, 48, 0.95)');
    bgGradient.addColorStop(0.5, 'rgba(6, 12, 26, 0.98)');
    bgGradient.addColorStop(1, 'rgba(2, 4, 10, 1)');
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, this.width, this.height);

    // Cosmic Particles with warp streaking
    this.particles.forEach(p => {
      p.y -= p.speed * (1 + this.warpProgress * 10);
      if (p.y < 0) p.y = this.height;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius * (1 + this.warpProgress), 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 242, 254, ${p.alpha * (1 - this.warpProgress * 0.5)})`;
      ctx.shadowBlur = 10;
      ctx.shadowColor = '#00f2fe';
      ctx.fill();
    });
    ctx.shadowBlur = 0;

    const baseGlobeRadius = Math.min(this.width, this.height) * 0.26;
    const currentGlobeRadius = baseGlobeRadius * this.zoomFactor;

    // Outer Atmosphere Glow
    const atmosphereGlow = ctx.createRadialGradient(cx, cy, currentGlobeRadius * 0.8, cx, cy, currentGlobeRadius * 1.4);
    atmosphereGlow.addColorStop(0, 'rgba(0, 242, 254, 0.35)');
    atmosphereGlow.addColorStop(0.5, 'rgba(79, 172, 254, 0.15)');
    atmosphereGlow.addColorStop(1, 'rgba(0, 242, 254, 0)');

    ctx.beginPath();
    ctx.arc(cx, cy, currentGlobeRadius * 1.4, 0, Math.PI * 2);
    ctx.fillStyle = atmosphereGlow;
    ctx.fill();

    // Globe Latitude & Longitude Dotted Grid
    ctx.save();
    ctx.setLineDash([4, 6]); // Dotted grid lines
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.22)';
    ctx.lineWidth = 1.2;

    for (let i = -3; i <= 3; i++) {
      ctx.beginPath();
      ctx.ellipse(cx, cy + i * (currentGlobeRadius / 4), currentGlobeRadius * Math.cos((i * Math.PI) / 8), currentGlobeRadius * 0.35, 0, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.setLineDash([3, 8]);
    for (let a = 0; a < Math.PI; a += Math.PI / 4) {
      ctx.beginPath();
      ctx.ellipse(cx, cy, currentGlobeRadius * Math.sin(a + this.time * 0.15), currentGlobeRadius, a, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();

    // Radar Scanning Ring Pulse
    const scanRadius = (this.radarPulse / 360) * (currentGlobeRadius * 1.5);
    ctx.beginPath();
    ctx.arc(cx, cy, scanRadius, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(0, 242, 254, ${Math.max(0, 1 - scanRadius / (currentGlobeRadius * 1.5))})`;
    ctx.lineWidth = 1.8;
    ctx.stroke();

    // --- REVOLVING SATELLITES ALONG DOTTED ORBITAL PATHS ---
    this.satellites.forEach(s => {
      // Advance revolve angle
      s.angle += s.speed * this.warpSpeedMultiplier;

      const rx = s.radiusX * this.zoomFactor;
      const ry = s.radiusY * this.zoomFactor;

      // Draw Dotted Orbital Path Ring
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(s.tilt);
      
      // Dotted trajectory line
      ctx.beginPath();
      ctx.setLineDash([4, 10]); // Distinct dotted orbit
      ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(0, 242, 254, ${0.28 + Math.sin(this.time * 2) * 0.1})`;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Dotted Nodes along the path
      this.orbitDottedNodes.forEach(node => {
        const dotAngle = node.angleOffset;
        const nx = Math.cos(dotAngle) * rx;
        const ny = Math.sin(dotAngle) * ry;

        ctx.beginPath();
        ctx.arc(nx, ny, 1.8, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 242, 254, ${0.4 + Math.sin(this.time * 3 + node.pulseOffset) * 0.3})`;
        ctx.fill();
      });

      // Calculate revolving satellite 3D position
      const sx = Math.cos(s.angle) * rx;
      const sy = Math.sin(s.angle) * ry;

      // Satellite Trail / Glow Effect
      const trailGradient = ctx.createRadialGradient(sx, sy, 1, sx, sy, 16);
      trailGradient.addColorStop(0, s.color);
      trailGradient.addColorStop(1, 'rgba(0, 242, 254, 0)');
      ctx.beginPath();
      ctx.arc(sx, sy, 16, 0, Math.PI * 2);
      ctx.fillStyle = trailGradient;
      ctx.fill();

      // Satellite Core Point
      ctx.beginPath();
      ctx.arc(sx, sy, s.size, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.shadowBlur = 15;
      ctx.shadowColor = s.color;
      ctx.fill();

      // Orbital Pulse Ring surrounding satellite
      ctx.beginPath();
      ctx.arc(sx, sy, s.size + 6 + Math.sin(this.time * 5) * 3, 0, Math.PI * 2);
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([]);
      ctx.stroke();

      // Satellite Tag Label
      ctx.font = '700 11px Inter, sans-serif';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
      ctx.fillText(s.label, sx + 12, sy + 4);

      ctx.restore();
    });

    ctx.shadowBlur = 0;
  }

  drawStaticFrame() {
    this.draw();
  }
}

// Auto-initialize when canvas exists
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('heroSpaceCanvas')) {
    window.heroEngine = new SpaceHeroEngine('heroSpaceCanvas');
  }
});
