// 공용 포스 그래프 렌더러 — 전체 그래프(/graph)와 포스트 로컬 그래프가 함께 사용
export type NodeKind = "section" | "category" | "post";

export interface GraphData {
  nodes: { id: string; label: string; kind: NodeKind; section: string; url?: string }[];
  edges: { source: string; target: string; kind: "tree" | "tag" }[];
}

export interface GraphOptions {
  /** 이 노드를 강조(크게, 링 표시)하고 초기 중앙에 배치 */
  centerId?: string;
  /** 캔버스 높이 계산 함수 */
  height?: () => number;
  /** 휠 줌 / 빈 곳 드래그 팬 허용 여부 (미니 그래프에선 끔) */
  enableZoomPan?: boolean;
  /** 초기 배율 */
  initialScale?: number;
  /** 라벨 표시: full = 섹션/카테고리 상시 + 포스트는 줌/호버, mini = 호버/센터만 */
  labelMode?: "full" | "mini";
}

const SECTION_COLORS: Record<string, string> = {
  papers: "#22d3ee",
  study: "#4ade80",
  projects: "#818cf8",
};

interface NodeDatum {
  id: string;
  label: string;
  kind: NodeKind;
  section: string;
  url?: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  fixed: boolean;
}

export function initForceGraph(canvas: HTMLCanvasElement, data: GraphData, opts: GraphOptions = {}) {
  const ctx = canvas.getContext("2d")!;
  const enableZoomPan = opts.enableZoomPan ?? true;
  const labelMode = opts.labelMode ?? "full";

  const idx = new Map<string, number>();
  const nodes: NodeDatum[] = data.nodes.map((n, i) => {
    idx.set(n.id, i);
    const angle = (i / data.nodes.length) * Math.PI * 2;
    const dist = n.kind === "section" ? 40 : n.kind === "category" ? 120 : 220;
    const isCenter = n.id === opts.centerId;
    return {
      ...n,
      x: isCenter ? 0 : Math.cos(angle) * dist + (Math.random() - 0.5) * 30,
      y: isCenter ? 0 : Math.sin(angle) * dist + (Math.random() - 0.5) * 30,
      vx: 0,
      vy: 0,
      r: isCenter ? 9 : n.kind === "section" ? 13 : n.kind === "category" ? 8 : 5,
      fixed: false,
    };
  });
  const centerIdx = opts.centerId != null ? (idx.get(opts.centerId) ?? -1) : -1;

  const edges = data.edges
    .filter((e) => idx.has(e.source) && idx.has(e.target))
    .map((e) => ({ source: idx.get(e.source)!, target: idx.get(e.target)!, kind: e.kind }));

  const neighbors: Set<number>[] = nodes.map(() => new Set());
  for (const e of edges) {
    neighbors[e.source]!.add(e.target);
    neighbors[e.target]!.add(e.source);
  }

  // ---- 뷰 상태 ----
  let W = 0;
  let H = 0;
  let scale = opts.initialScale ?? 1;
  let panX = 0;
  let panY = 0;
  let hovered = -1;
  let dragging = -1;
  let panning = false;
  let lastMx = 0;
  let lastMy = 0;
  let downX = 0;
  let downY = 0;
  let alpha = 1;

  function resize() {
    const rect = canvas.parentElement!.getBoundingClientRect();
    W = rect.width;
    H = opts.height ? opts.height() : Math.max(420, Math.min(680, window.innerHeight * 0.65));
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener("resize", () => {
    resize();
    alpha = Math.max(alpha, 0.3);
  });

  const toWorld = (mx: number, my: number) => ({
    x: (mx - W / 2 - panX) / scale,
    y: (my - H / 2 - panY) / scale,
  });

  function pick(mx: number, my: number): number {
    const { x, y } = toWorld(mx, my);
    for (let i = nodes.length - 1; i >= 0; i--) {
      const n = nodes[i]!;
      const dx = n.x - x;
      const dy = n.y - y;
      if (dx * dx + dy * dy < Math.pow(n.r + 4, 2)) return i;
    }
    return -1;
  }

  // ---- 물리 ----
  function tick() {
    if (alpha < 0.005) return;
    alpha *= 0.995;

    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i]!;
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j]!;
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1) d2 = 1;
        const d = Math.sqrt(d2);
        const rep = (1400 / d2) * alpha;
        const fx = (dx / d) * rep;
        const fy = (dy / d) * rep;
        a.vx -= fx;
        a.vy -= fy;
        b.vx += fx;
        b.vy += fy;
      }
    }

    for (const e of edges) {
      const a = nodes[e.source]!;
      const b = nodes[e.target]!;
      const rest = e.kind === "tag" ? 130 : a.kind === "section" ? 90 : 60;
      const k = e.kind === "tag" ? 0.008 : 0.03;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = k * (d - rest) * alpha;
      const fx = (dx / d) * f;
      const fy = (dy / d) * f;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }

    for (const n of nodes) {
      n.vx += -n.x * 0.005 * alpha;
      n.vy += -n.y * 0.005 * alpha;
      if (!n.fixed) {
        n.vx *= 0.85;
        n.vy *= 0.85;
        n.x += n.vx;
        n.y += n.vy;
      } else {
        n.vx = 0;
        n.vy = 0;
      }
    }
  }

  // ---- 렌더링 ----
  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.translate(W / 2 + panX, H / 2 + panY);
    ctx.scale(scale, scale);

    const hasHover = hovered >= 0;
    const isDim = (i: number) => hasHover && i !== hovered && !neighbors[hovered]!.has(i);

    for (const e of edges) {
      const a = nodes[e.source]!;
      const b = nodes[e.target]!;
      const dim = hasHover && !(e.source === hovered || e.target === hovered);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.lineWidth = (e.kind === "tag" ? 0.7 : 1.2) / scale;
      if (e.kind === "tag") ctx.setLineDash([4 / scale, 4 / scale]);
      else ctx.setLineDash([]);
      ctx.strokeStyle = dim
        ? "rgba(253,115,24,0.06)"
        : e.kind === "tag"
          ? "rgba(253,228,113,0.35)"
          : "rgba(253,115,24,0.35)";
      ctx.stroke();
    }
    ctx.setLineDash([]);

    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i]!;
      const color = SECTION_COLORS[n.section] ?? "#a1a1aa";
      const dim = isDim(i);
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = dim ? "rgba(161,161,170,0.15)" : color;
      if (!dim && (i === hovered || n.kind === "section" || i === centerIdx)) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 14;
      }
      ctx.fill();
      ctx.shadowBlur = 0;
      if (n.kind !== "post" || i === centerIdx) {
        ctx.lineWidth = 1.5 / scale;
        ctx.strokeStyle = dim ? "rgba(161,161,170,0.1)" : i === centerIdx ? "#fafafa" : "#060a13";
        ctx.stroke();
      }
    }

    // 라벨
    ctx.textAlign = "center";
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i]!;
      const dim = isDim(i);
      let showLabel: boolean;
      if (labelMode === "mini") {
        showLabel = i === centerIdx || i === hovered || (hasHover && neighbors[hovered]!.has(i));
      } else {
        showLabel =
          n.kind !== "post" || scale > 1.15 || i === hovered || (hasHover && neighbors[hovered]!.has(i));
      }
      if (!showLabel || dim) continue;
      const size = n.kind === "section" ? 13 : n.kind === "category" ? 11 : 10;
      ctx.font = `${n.kind === "post" ? "" : "700 "}${size / scale}px "JetBrains Mono", monospace`;
      const maxLen = labelMode === "mini" ? 22 : 34;
      const label = n.label.length > maxLen ? n.label.slice(0, maxLen - 2) + "…" : n.label;
      const ly = n.y + n.r + 14 / scale;
      ctx.fillStyle = "rgba(13,3,15,0.75)";
      const tw = ctx.measureText(label).width;
      ctx.fillRect(n.x - tw / 2 - 3 / scale, ly - 10 / scale, tw + 6 / scale, 13 / scale);
      ctx.fillStyle = i === hovered || i === centerIdx ? "#fafafa" : "#a1a1aa";
      ctx.fillText(label, n.x, ly);
    }

    ctx.restore();
  }

  function loop() {
    tick();
    draw();
    requestAnimationFrame(loop);
  }
  loop();

  // ---- 인터랙션 ----
  canvas.addEventListener("mousedown", (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    downX = mx;
    downY = my;
    const i = pick(mx, my);
    if (i >= 0) {
      dragging = i;
      nodes[i]!.fixed = true;
      alpha = Math.max(alpha, 0.3);
    } else if (enableZoomPan) {
      panning = true;
    }
    lastMx = mx;
    lastMy = my;
  });

  canvas.addEventListener("mousemove", (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    if (dragging >= 0) {
      const w = toWorld(mx, my);
      nodes[dragging]!.x = w.x;
      nodes[dragging]!.y = w.y;
      alpha = Math.max(alpha, 0.3);
    } else if (panning) {
      panX += mx - lastMx;
      panY += my - lastMy;
    } else {
      hovered = pick(mx, my);
      canvas.style.cursor = hovered >= 0 ? "pointer" : enableZoomPan ? "grab" : "default";
    }
    lastMx = mx;
    lastMy = my;
  });

  window.addEventListener("mouseup", () => {
    if (dragging >= 0) {
      const moved = Math.hypot(lastMx - downX, lastMy - downY);
      const n = nodes[dragging]!;
      n.fixed = false;
      if (moved < 5 && n.url) window.location.href = n.url;
    }
    dragging = -1;
    panning = false;
  });

  canvas.addEventListener("mouseleave", () => {
    hovered = -1;
  });

  if (enableZoomPan) {
    canvas.addEventListener(
      "wheel",
      (ev) => {
        ev.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const mx = ev.clientX - rect.left - W / 2;
        const my = ev.clientY - rect.top - H / 2;
        const factor = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
        const next = Math.min(3.5, Math.max(0.35, scale * factor));
        panX = mx - ((mx - panX) * next) / scale;
        panY = my - ((my - panY) * next) / scale;
        scale = next;
      },
      { passive: false },
    );

    // 터치: 한 손가락 = 팬
    let touchId: number | null = null;
    canvas.addEventListener("touchstart", (ev) => {
      const t = ev.touches[0]!;
      touchId = t.identifier;
      const rect = canvas.getBoundingClientRect();
      lastMx = t.clientX - rect.left;
      lastMy = t.clientY - rect.top;
    });
    canvas.addEventListener(
      "touchmove",
      (ev) => {
        ev.preventDefault();
        const t = [...ev.touches].find((x) => x.identifier === touchId);
        if (!t) return;
        const rect = canvas.getBoundingClientRect();
        const mx = t.clientX - rect.left;
        const my = t.clientY - rect.top;
        panX += mx - lastMx;
        panY += my - lastMy;
        lastMx = mx;
        lastMy = my;
      },
      { passive: false },
    );
  }
}
