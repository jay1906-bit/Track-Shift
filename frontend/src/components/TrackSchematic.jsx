import React, { useMemo } from "react";
import BattleViewport from "./BattleViewport.jsx";
import {
  attackerDistanceM,
  fracOnCircuit,
  isFiniteNumber,
  lookaheadWindowM,
  targetDistanceM,
} from "../data/formatReplay.js";

const ZONE = {
  straight: "#5a6570",
  brake: "#a9844a",
  corner: "#6d8296",
};

const C = {
  track: "#2a323c",
  window: "#d7dee6",
  attacker: "#3dd68c",
  target: "#e8eaed",
};

function wrap01(frac) {
  if (!Number.isFinite(frac)) return null;
  const w = frac % 1;
  return w < 0 ? w + 1 : w;
}

function wrapM(distance, length) {
  if (!isFiniteNumber(distance) || !isFiniteNumber(length) || length <= 0) return null;
  const w = distance % length;
  return w < 0 ? w + length : w;
}

function pointAt(points, frac) {
  const f = wrap01(frac);
  if (f == null || !points.length) return null;
  if (f <= points[0][2]) return { x: points[0][0], y: points[0][1], i: 0 };
  for (let i = 1; i < points.length; i += 1) {
    if (f <= points[i][2]) {
      const a = points[i - 1];
      const b = points[i];
      const span = b[2] - a[2] || 1e-9;
      const t = (f - a[2]) / span;
      return {
        x: a[0] + t * (b[0] - a[0]),
        y: a[1] + t * (b[1] - a[1]),
        i,
      };
    }
  }
  const last = points[points.length - 1];
  return { x: last[0], y: last[1], i: points.length - 1 };
}

function tangentAt(points, frac) {
  const a = pointAt(points, frac - 0.003);
  const b = pointAt(points, frac + 0.003);
  if (!a || !b) return { dx: 1, dy: 0, nx: 0, ny: 1 };
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  return { dx: dx / len, dy: dy / len, nx: -dy / len, ny: dx / len };
}

function sampleRun(points, start, end, step) {
  const out = [];
  for (let f = start; f <= end + 1e-9; f += step) out.push(pointAt(points, Math.min(f, end)));
  return out.filter(Boolean);
}

function windowRuns(points, startFrac, span, step = 0.003) {
  const start = wrap01(startFrac);
  if (start == null || !Number.isFinite(span) || span <= 0) return [];
  const end = start + span;
  if (end <= 1) return [sampleRun(points, start, end, step)];
  return [sampleRun(points, start, 1, step), sampleRun(points, 0, end - 1, step)];
}

function pathFromPoints(pts) {
  if (!pts.length) return "";
  return pts.map((p, i) => `${i ? "L" : "M"} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(" ");
}

function zoneTypeAt(zones, distM, length) {
  const d = wrapM(distM, length);
  if (d == null) return "straight";
  for (const zone of zones) {
    const start = zone.start_m;
    const end = zone.end_m;
    if (!isFiniteNumber(start) || !isFiniteNumber(end)) continue;
    if (end >= start && d >= start && d < end) return zone.zone_type || "straight";
    if (end < start && (d >= start || d < end)) return zone.zone_type || "straight";
  }
  return "straight";
}

function layout(points, width, height, pad) {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const bw = Math.max(1, maxX - minX);
  const bh = Math.max(1, maxY - minY);
  const scale = Math.min((width - pad * 2) / bw, (height - pad * 2) / bh);
  const ox = (width - bw * scale) / 2;
  const oy = (height - bh * scale) / 2;
  const map = (x, y) => ({
    x: ox + (x - minX) * scale,
    y: oy + (maxY - y) * scale,
  });
  const mapped = points.map((p) => {
    const s = map(p[0], p[1]);
    return [s.x, s.y, p[2]];
  });
  return { mapped, map, scale, bw, bh };
}

function zonePathsFromMapped(mapped, zones, length) {
  const zonePaths = [];
  if (!zones.length || !isFiniteNumber(length)) return zonePaths;
  for (let i = 1; i < mapped.length; i += 1) {
    const midFrac = (mapped[i - 1][2] + mapped[i][2]) / 2;
    const type = zoneTypeAt(zones, midFrac * length, length);
    const d = `M ${mapped[i - 1][0].toFixed(2)} ${mapped[i - 1][1].toFixed(2)} L ${mapped[i][0].toFixed(2)} ${mapped[i][1].toFixed(2)}`;
    const prev = zonePaths[zonePaths.length - 1];
    if (prev && prev.type === type) prev.d += ` L ${mapped[i][0].toFixed(2)} ${mapped[i][1].toFixed(2)}`;
    else zonePaths.push({ type, d });
  }
  return zonePaths;
}

function CircuitOverview({ event, circuit, attacker, target, length, windowM }) {
  const points = circuit.polyline;
  const zones = circuit.zones || [];
  const { mapped } = useMemo(() => layout(points, 1000, 520, 40), [points]);

  const attFrac = fracOnCircuit(attacker, length);
  const tgtFrac = fracOnCircuit(target, length);
  const winFrac = isFiniteNumber(windowM) && length > 0 ? windowM / length : null;
  const att = attFrac == null ? null : pointAt(mapped, attFrac);
  const tgt = tgtFrac == null ? null : pointAt(mapped, tgtFrac);
  const sf = pointAt(mapped, 0);
  const sfT = tangentAt(mapped, 0);
  const dir = pointAt(mapped, 0.016);
  const dirT = tangentAt(mapped, 0.016);
  const closed = mapped.concat([[mapped[0][0], mapped[0][1], 1]]);
  const basePath = pathFromPoints(closed.map((p) => ({ x: p[0], y: p[1] })));
  const zonePaths = zonePathsFromMapped(mapped, zones, length);
  const windowPaths =
    attFrac == null || winFrac == null || winFrac <= 0
      ? []
      : windowRuns(mapped, attFrac, winFrac).map(pathFromPoints);

  const loc =
    att && tgt
      ? {
          x: (att.x + tgt.x) / 2,
          y: (att.y + tgt.y) / 2,
          r: Math.max(26, Math.hypot(att.x - tgt.x, att.y - tgt.y) * 0.65 + 20),
        }
      : att
        ? { x: att.x, y: att.y, r: 28 }
        : null;

  return (
    <div className="aq-track">
      <div className="aq-map-label">Circuit</div>
      <svg viewBox="0 0 1000 520" width="100%" height="100%" role="img" aria-label="Full circuit">
        <path d={basePath} fill="none" stroke={C.track} strokeWidth="16" strokeLinejoin="round" strokeLinecap="round" />
        {zonePaths.map((seg, i) => (
          <path key={`z-${i}`} d={seg.d} fill="none" stroke={ZONE[seg.type] || ZONE.straight} strokeWidth="10" strokeLinejoin="round" />
        ))}
        {windowPaths.map((d, i) => (
          <path key={`win-${i}`} d={d} fill="none" stroke={C.window} strokeWidth="9" strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {sf && sfT ? (
          <line x1={sf.x + sfT.nx * 10} y1={sf.y + sfT.ny * 10} x2={sf.x - sfT.nx * 10} y2={sf.y - sfT.ny * 10} stroke="#c5ccd4" strokeWidth="3" />
        ) : null}
        {dir && dirT ? (
          <polygon
            points={`${dir.x + dirT.dx * 9},${dir.y + dirT.dy * 9} ${dir.x - dirT.dx * 5 + dirT.nx * 4.5},${dir.y - dirT.dy * 5 + dirT.ny * 4.5} ${dir.x - dirT.dx * 5 - dirT.nx * 4.5},${dir.y - dirT.dy * 5 - dirT.ny * 4.5}`}
            fill="#9aa3ad"
          />
        ) : null}
        {loc ? <circle cx={loc.x} cy={loc.y} r={loc.r} fill="none" stroke="#d7dee6" strokeWidth="1.4" strokeDasharray="5 4" opacity="0.7" /> : null}
        {tgt ? <circle cx={tgt.x} cy={tgt.y} r="7" fill={C.target} stroke="#08090b" strokeWidth="2" /> : null}
        {att ? <circle cx={att.x} cy={att.y} r="8" fill={C.attacker} stroke="#08090b" strokeWidth="2" /> : null}
        {att && winFrac ? (
          <text x={Math.min(970, att.x + 16)} y={Math.max(18, att.y - 14)} fill={C.window} fontSize="13" fontWeight="600">
            Next 400 m
          </text>
        ) : null}
      </svg>
    </div>
  );
}

function ZoneFallback({ event, circuit, attacker, target, length, windowM }) {
  const zones = circuit?.zones || [];
  const W = 920;
  const H = 360;
  const padX = 70;
  const padY = 70;
  const rw = W - padX * 2;
  const rh = H - padY * 2;
  const r = 48;
  const perim = 2 * (rw + rh - 2 * r) + 2 * Math.PI * r;

  function atFrac(frac) {
    const d = wrap01(frac) * perim;
    const straightW = rw - 2 * r;
    const straightH = rh - 2 * r;
    const a1 = straightW;
    const a2 = a1 + (Math.PI * r) / 2;
    const a3 = a2 + straightH;
    const a4 = a3 + (Math.PI * r) / 2;
    const a5 = a4 + straightW;
    const a6 = a5 + (Math.PI * r) / 2;
    const a7 = a6 + straightH;
    const x0 = padX + r;
    const y0 = padY;
    if (d <= a1) return { x: x0 + d, y: y0 };
    if (d <= a2) {
      const t = (d - a1) / ((Math.PI * r) / 2);
      return { x: padX + rw - r + Math.sin((t * Math.PI) / 2) * r, y: padY + r - Math.cos((t * Math.PI) / 2) * r };
    }
    if (d <= a3) return { x: padX + rw, y: padY + r + (d - a2) };
    if (d <= a4) {
      const t = (d - a3) / ((Math.PI * r) / 2);
      return { x: padX + rw - r + Math.cos((t * Math.PI) / 2) * r, y: padY + rh - r + Math.sin((t * Math.PI) / 2) * r };
    }
    if (d <= a5) return { x: padX + rw - r - (d - a4), y: padY + rh };
    if (d <= a6) {
      const t = (d - a5) / ((Math.PI * r) / 2);
      return { x: padX + r - Math.sin((t * Math.PI) / 2) * r, y: padY + rh - r + Math.cos((t * Math.PI) / 2) * r };
    }
    if (d <= a7) return { x: padX, y: padY + rh - r - (d - a6) };
    const t = (d - a7) / ((Math.PI * r) / 2);
    return { x: padX + r - Math.cos((t * Math.PI) / 2) * r, y: padY + r - Math.sin((t * Math.PI) / 2) * r };
  }

  const attFrac = fracOnCircuit(attacker, length);
  const tgtFrac = fracOnCircuit(target, length);
  const att = attFrac == null ? null : atFrac(attFrac);
  const tgt = tgtFrac == null ? null : atFrac(tgtFrac);
  const winFrac = isFiniteNumber(windowM) && length > 0 ? windowM / length : null;
  const winPts =
    attFrac == null || winFrac == null ? [] : Array.from({ length: 40 }, (_, i) => atFrac(attFrac + (winFrac * i) / 39));

  return (
    <div className="aq-track">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="100%">
        <rect x={padX} y={padY} width={rw} height={rh} rx={r} ry={r} fill="none" stroke={C.track} strokeWidth="16" />
        {zones.map((zone) => {
          const start = zone.start_m / length;
          const span = (zone.end_m - zone.start_m) / length;
          const pts = Array.from({ length: 12 }, (_, i) => atFrac(start + (span * i) / 11));
          return <path key={zone.zone_id} d={pathFromPoints(pts)} fill="none" stroke={ZONE[zone.zone_type] || ZONE.straight} strokeWidth="10" />;
        })}
        {winPts.length ? <path d={pathFromPoints(winPts)} fill="none" stroke={C.window} strokeWidth="8" /> : null}
        {tgt ? <circle cx={tgt.x} cy={tgt.y} r="7" fill={C.target} /> : null}
        {att ? <circle cx={att.x} cy={att.y} r="7" fill={C.attacker} /> : null}
      </svg>
      <div className="aq-track-meta">Zone-table schematic · FastF1 X/Y unavailable for this race · not GPS</div>
    </div>
  );
}

function LinearSchematic({ event, attacker, target, length, windowM }) {
  const attPct = isFiniteNumber(length) && attacker != null ? (attacker / length) * 100 : null;
  const tgtPct = isFiniteNumber(length) && target != null ? (target / length) * 100 : null;
  const winPct = isFiniteNumber(length) && windowM != null ? (windowM / length) * 100 : 0;
  return (
    <div className="aq-track">
      <div style={{ position: "relative", height: 14, background: "#1c232c", margin: "40px 12px 24px" }}>
        {attPct != null && windowM != null && (
          <div style={{ position: "absolute", left: `${attPct}%`, width: `${Math.min(100 - attPct, winPct)}%`, height: "100%", background: "rgba(215,222,230,0.4)" }} />
        )}
        {tgtPct != null && <div style={{ position: "absolute", left: `${tgtPct}%`, top: -5, width: 10, height: 10, borderRadius: "50%", background: C.target, transform: "translateX(-50%)" }} />}
        {attPct != null && <div style={{ position: "absolute", left: `${attPct}%`, top: -5, width: 10, height: 10, borderRadius: "50%", background: C.attacker, transform: "translateX(-50%)" }} />}
      </div>
      <div className="aq-track-meta">Along-track bar · circuit outline unavailable</div>
      <div style={{ fontSize: 13, color: "#8b929c", marginTop: 6 }}>{event?.zone_type ? `Zone: ${event.zone_type}` : "Zone unavailable"}</div>
    </div>
  );
}

export default function TrackSchematic({ event, circuit }) {
  const length = circuit?.circuit_length_m;
  const attacker = attackerDistanceM(event, length);
  const target = targetDistanceM(event, length);
  const windowM = lookaheadWindowM(event);
  const polyline = circuit?.polyline;

  if (Array.isArray(polyline) && polyline.length >= 50 && isFiniteNumber(length)) {
    return (
      <div className="aq-track-stack">
        <CircuitOverview event={event} circuit={circuit} attacker={attacker} target={target} length={length} windowM={windowM} />
        <BattleViewport event={event} circuit={circuit} />
        <div className="aq-track-meta">
          <span>FastF1 flying-lap outline · not live GPS</span>
          <span className="aq-legend"><i style={{ background: ZONE.straight }} />Straight</span>
          <span className="aq-legend"><i style={{ background: ZONE.brake }} />Brake</span>
          <span className="aq-legend"><i style={{ background: ZONE.corner }} />Corner</span>
          <span className="aq-legend"><i style={{ background: C.window, height: 3 }} />Next 400m</span>
        </div>
      </div>
    );
  }
  if ((circuit?.zones || []).length && isFiniteNumber(length) && length > 0) {
    return (
      <div className="aq-track-stack">
        <ZoneFallback event={event} circuit={circuit} attacker={attacker} target={target} length={length} windowM={windowM} />
        <BattleViewport event={event} circuit={circuit} />
      </div>
    );
  }
  return (
    <div className="aq-track-stack">
      <LinearSchematic event={event} attacker={attacker} target={target} length={length} windowM={windowM} />
      <BattleViewport event={event} circuit={circuit} />
    </div>
  );
}
