import { useEffect, useRef } from "react";

// Map any weather_condition string → animation type
const CONDITION_ANIM = {
  clear:        "sunny",
  "clear-night": "starry",
  partly_cloudy:"cloudy",
  "partly_cloudy-night":"starry",
  cloudy:       "cloudy",
  "cloudy-night":"cloudy",
  rain:         "rainy",
  "rain-night": "rainy",
  heavy_rain:   "rainy",
  "heavy_rain-night": "rainy",
  drizzle:      "drizzle",
  "drizzle-night": "drizzle",
  snow:         "snowy",
  "snow-night": "snowy",
  heavy_snow:   "snowy",
  "heavy_snow-night": "snowy",
  fog:          "foggy",
  "fog-night":  "foggy",
  thunderstorm: "thunderstorm",
  "thunderstorm-night": "thunderstorm",
  windy:        "cloudy",
  "windy-night": "cloudy",
  hail:         "rainy",
  "hail-night": "rainy",
};

/* ─── Utility ──────────────────────────────────── */
function rand(min, max) { return Math.random() * (max - min) + min; }

function starryNightAnim() {
  let stars = [];
  return {
    init(ctx, W, H) {
      stars = Array.from({ length: 80 }, () => ({
        x: rand(0, W),
        y: rand(0, H * 0.7),
        r: rand(0.5, 2.2),
        alpha: rand(0.1, 1),
        speed: rand(0.005, 0.02),
        dir: Math.random() > 0.5 ? 1 : -1
      }));
    },
    draw(ctx, W, H) {
      // Glow from a gentle moon
      const cx = W * 0.85, cy = H * 0.15;
      const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, 300);
      grad.addColorStop(0, "rgba(255, 255, 255, 0.15)");
      grad.addColorStop(1, "rgba(255, 255, 255, 0)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
      
      stars.forEach(s => {
        s.alpha += s.speed * s.dir;
        if (s.alpha > 1) { s.alpha = 1; s.dir = -1; }
        if (s.alpha < 0) { 
          s.alpha = 0; s.dir = 1; 
          s.x = rand(0, W); 
          s.y = rand(0, H * 0.7); 
        }
        
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${s.alpha})`;
        ctx.fill();
      });
    }
  }
}

/* ─── Animators ─────────────────────────────────
   Each animator returns { init(ctx, W, H), draw(ctx, W, H, dt) }
   and must release no DOM resources (pure canvas).
───────────────────────────────────────────────── */

function sunnyClear() {
  let rays = [];
  return {
    init(ctx, W, H) {
      rays = Array.from({ length: 14 }, (_, i) => ({
        angle: (i / 14) * Math.PI * 2,
        speed: rand(0.0003, 0.0008),
        len: rand(0.8, 1.4),
        width: rand(0.04, 0.08), // arc width
        alpha: rand(0.03, 0.07),
      }));
    },
    draw(ctx, W, H) {
      // Place sun in the top right to avoid UI card overlap
      const cx = W * 0.85, cy = H * 0.12;
      const dim = Math.max(W, H);

      // Rotating rays
      rays.forEach(r => {
        r.angle += r.speed;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(r.angle) * dim * r.len, cy + Math.sin(r.angle) * dim * r.len);
        ctx.lineTo(cx + Math.cos(r.angle + r.width) * dim * r.len, cy + Math.sin(r.angle + r.width) * dim * r.len);
        ctx.fillStyle = `rgba(253, 224, 71, ${r.alpha})`; // Bright yellow rays
        ctx.fill();
      });

      // Bright solid sun disk
      ctx.beginPath();
      ctx.arc(cx, cy, 45, 0, Math.PI * 2);
      ctx.fillStyle = "#fde047";
      ctx.fill();

      // Strong volumetric glow around the sun
      const grad = ctx.createRadialGradient(cx, cy, 45, cx, cy, 300);
      grad.addColorStop(0, "rgba(250, 204, 21, 0.6)");
      grad.addColorStop(0.4, "rgba(234, 179, 8, 0.15)");
      grad.addColorStop(1, "rgba(234, 179, 8, 0)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
    },
  };
}

function rainyAnim(heavy = false) {
  let drops = [];
  const count = heavy ? 220 : 120;
  return {
    init(ctx, W, H) {
      drops = Array.from({ length: count }, () => ({
        x: rand(0, W),
        y: rand(-H, 0),
        len: rand(10, heavy ? 25 : 18),
        speed: rand(6, heavy ? 18 : 12),
        alpha: rand(0.15, 0.4),
        width: rand(0.8, 1.6),
      }));
    },
    draw(ctx, W, H) {
      ctx.save();
      drops.forEach(d => {
        d.y += d.speed;
        d.x -= d.speed * 0.15;  // slight angle
        if (d.y > H) { d.y = rand(-40, 0); d.x = rand(0, W); }

        ctx.beginPath();
        ctx.moveTo(d.x, d.y);
        ctx.lineTo(d.x - d.len * 0.12, d.y + d.len);
        ctx.lineWidth = d.width;
        ctx.strokeStyle = `rgba(130,180,230,${d.alpha})`;
        ctx.stroke();
      });
      ctx.restore();
    },
  };
}

function drizzleAnim() {
  let drops = [];
  return {
    init(ctx, W, H) {
      drops = Array.from({ length: 70 }, () => ({
        x: rand(0, W),
        y: rand(-H, 0),
        len: rand(6, 12),
        speed: rand(2, 5),
        alpha: rand(0.1, 0.25),
        width: rand(0.5, 1),
      }));
    },
    draw(ctx, W, H) {
      drops.forEach(d => {
        d.y += d.speed;
        if (d.y > H) { d.y = rand(-30, 0); d.x = rand(0, W); }
        ctx.beginPath();
        ctx.moveTo(d.x, d.y);
        ctx.lineTo(d.x - d.len * 0.08, d.y + d.len);
        ctx.lineWidth = d.width;
        ctx.strokeStyle = `rgba(160,200,240,${d.alpha})`;
        ctx.stroke();
      });
    },
  };
}

function snowyAnim(heavy = false) {
  let flakes = [];
  const count = heavy ? 180 : 90;
  return {
    init(ctx, W, H) {
      flakes = Array.from({ length: count }, () => ({
        x: rand(0, W),
        y: rand(-H, H),
        r: rand(heavy ? 2 : 1, heavy ? 5 : 3.5),
        speed: rand(0.5, heavy ? 2.5 : 1.8),
        drift: rand(-0.4, 0.4),
        alpha: rand(0.4, 0.85),
        phase: rand(0, Math.PI * 2),
      }));
    },
    draw(ctx, W, H, t) {
      flakes.forEach(f => {
        f.y += f.speed;
        f.x += f.drift + Math.sin(t * 0.001 + f.phase) * 0.3;
        if (f.y > H + 10) { f.y = -10; f.x = rand(0, W); }
        if (f.x > W + 10) f.x = -10;
        if (f.x < -10) f.x = W + 10;

        ctx.beginPath();
        ctx.arc(f.x, f.y, f.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(230,240,255,${f.alpha})`;
        ctx.fill();
      });
    },
  };
}

function foggyAnim() {
  let layers = [];
  return {
    init(ctx, W, H) {
      layers = Array.from({ length: 6 }, (_, i) => ({
        y: H * (i / 6) + rand(-20, 20),
        speed: rand(0.2, 0.6) * (i % 2 === 0 ? 1 : -1),
        alpha: rand(0.04, 0.12),
        scale: rand(1.2, 2.5),
        offset: rand(0, W),
      }));
    },
    draw(ctx, W, H) {
      layers.forEach(l => {
        l.offset += l.speed;
        if (l.offset > W * 2) l.offset -= W * 3;
        if (l.offset < -W * 2) l.offset += W * 3;

        const grad = ctx.createLinearGradient(l.offset - W * l.scale / 2, 0, l.offset + W * l.scale / 2, 0);
        grad.addColorStop(0,   "rgba(180,190,200,0)");
        grad.addColorStop(0.3, `rgba(180,190,200,${l.alpha})`);
        grad.addColorStop(0.7, `rgba(180,190,200,${l.alpha})`);
        grad.addColorStop(1,   "rgba(180,190,200,0)");

        ctx.fillStyle = grad;
        ctx.fillRect(l.offset - W * l.scale / 2, l.y - 40, W * l.scale, 80);
      });
    },
  };
}

function cloudyAnim() {
  let clouds = [];
  return {
    init(ctx, W, H) {
      clouds = Array.from({ length: 6 }, (_, i) => ({
        x: rand(-200, W),
        y: rand(H * 0.05, H * 0.4),
        scale: rand(0.6, 1.5),
        speed: rand(0.15, 0.4),
        alpha: rand(0.2, 0.5),
      }));
    },
    draw(ctx, W, H) {
      clouds.forEach(c => {
        c.x += c.speed;
        if (c.x > W + 200) c.x = -200;

        ctx.save();
        ctx.globalAlpha = c.alpha;
        ctx.translate(c.x, c.y);
        ctx.scale(c.scale, c.scale);
        
        // Draw a realistic fluffy cloud shape via bezier curves
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.bezierCurveTo(-50, 0, -50, -50, 0, -50);
        ctx.bezierCurveTo(20, -90, 80, -70, 80, -50);
        ctx.bezierCurveTo(120, -50, 120, 0, 80, 0);
        ctx.closePath();
        
        ctx.fillStyle = "#cbd5e1"; // Semi-translucent cloud
        ctx.fill();
        ctx.restore();
      });
    },
  };
}

function thunderstormAnim() {
  const rain = rainyAnim(true);
  let lightning = { active: false, alpha: 0, timer: 0, nextAt: rand(80, 200) };
  let frame = 0;
  return {
    init(ctx, W, H) { rain.init(ctx, W, H); },
    draw(ctx, W, H, t) {
      rain.draw(ctx, W, H, t);
      frame++;

      if (!lightning.active && frame >= lightning.nextAt) {
        lightning.active = true;
        lightning.alpha = 0.7;
        lightning.timer = 0;
        lightning.nextAt = frame + rand(100, 300);
      }

      if (lightning.active) {
        lightning.alpha -= 0.045;
        lightning.timer++;
        if (lightning.alpha <= 0) {
          lightning.active = false;
        } else {
          // Flash the whole canvas white
          ctx.fillStyle = `rgba(220,230,255,${lightning.alpha})`;
          ctx.fillRect(0, 0, W, H);

          // Draw a jagged lightning bolt
          if (lightning.timer < 5) {
            const bx = rand(W * 0.2, W * 0.8);
            ctx.beginPath();
            ctx.moveTo(bx, 0);
            let cy = 0;
            while (cy < H * 0.65) {
              cy += rand(20, 50);
              ctx.lineTo(bx + rand(-30, 30), cy);
            }
            ctx.strokeStyle = `rgba(255,255,200,${lightning.alpha * 1.5})`;
            ctx.lineWidth = rand(1, 4);
            ctx.stroke();
          }
        }
      }
    },
  };
}

/* ─── Main Component ────────────────────────────── */
export default function WeatherBackground({ condition }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const rafRef = useRef(null);

  const animType = CONDITION_ANIM[condition] || null;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    // Resize handler
    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
      if (animRef.current?.init) {
        animRef.current.init(ctx, canvas.width, canvas.height);
      }
    };

    // Pick animator
    switch (animType) {
      case "sunny":        animRef.current = sunnyClear();    break;
      case "starry":       animRef.current = starryNightAnim(); break;
      case "rainy":        animRef.current = rainyAnim(true); break;
      case "drizzle":      animRef.current = drizzleAnim();   break;
      case "snowy":        animRef.current = snowyAnim(false);break;
      case "foggy":        animRef.current = foggyAnim();     break;
      case "cloudy":       animRef.current = cloudyAnim();    break;
      case "thunderstorm": animRef.current = thunderstormAnim(); break;
      default:             animRef.current = null;            break;
    }

    window.addEventListener("resize", resize);
    resize();

    if (!animRef.current) {
      cancelAnimationFrame(rafRef.current);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return () => window.removeEventListener("resize", resize);
    }

    // Animation loop
    let startTime = null;
    const loop = (ts) => {
      if (!startTime) startTime = ts;
      const t = ts - startTime;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      animRef.current.draw(ctx, canvas.width, canvas.height, t);
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [animType]);

  if (!animType) return null;

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 0,
      }}
    />
  );
}
