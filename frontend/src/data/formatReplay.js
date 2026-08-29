/** Display helpers. Do not invent values; return null when data is missing. */

export function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function isValidProbability(value) {
  return isFiniteNumber(value) && value >= 0 && value <= 1;
}

export function formatProbabilityPct(value) {
  if (!isValidProbability(value)) return null;
  return Math.round(value * 100);
}

export function formatEnergy(value) {
  if (!isFiniteNumber(value)) return null;
  return value.toFixed(2);
}

export function formatGapM(value) {
  if (!isFiniteNumber(value)) return null;
  return value.toFixed(1);
}

export function formatSpeed(value) {
  if (!isFiniteNumber(value)) return null;
  return value.toFixed(0);
}

export function formatSharePct(value) {
  if (!isFiniteNumber(value) || value < 0) return null;
  return Math.round(value * 100);
}

export function formatExact(value, digits = 6) {
  if (!isFiniteNumber(value)) return null;
  return value.toFixed(digits);
}

export function formatLap(value) {
  if (!isFiniteNumber(value)) return null;
  return String(Math.round(value));
}

export function formatSessionTime(value) {
  if (!value) return null;
  const text = String(value).trim().replace("T", " ");
  return text.length > 19 ? text.slice(0, 19) : text;
}

export function contextHeadline(event) {
  if (!event) return "Immediate 400m context unavailable";
  if (event.lookahead_truncated) return "Lookahead truncated";
  if (event.strategy_spend_compatible) return "Deploy-compatible";
  if (event.strategy_spend_compatible === false) return "Poor deployment window";
  return "Immediate 400m context unavailable";
}

export function canDecide(event) {
  if (!event) return false;
  if (!isValidProbability(event.p_hat)) return false;
  if (event.strategy_action !== "ATTACK" && event.strategy_action !== "SAVE") return false;
  return true;
}

export function raceLabel(race) {
  const circuit = race?.circuit || race?.race_id || "Unknown race";
  if (race?.split === "locked_test") return `${circuit} · locked`;
  return circuit;
}

export function eventLabel(event) {
  if (!event) return "Unknown event";
  const lap = formatLap(event.event_start_lap) ?? "?";
  const pair = `${event.attacker || "?"} → ${event.target || "?"}`;
  const action = event.strategy_action || "—";
  const pct = formatProbabilityPct(event.p_hat);
  const p = pct == null ? "p n/a" : `${pct}%`;
  return `L${lap} · ${pair} · ${action} · ${p}`;
}

export function historicalOutcome(event) {
  const value = event?.analysis_only_overtake_success;
  if (value === true || value === 1) return "PASS";
  if (value === false || value === 0) return "NO PASS";
  if (typeof value === "number" && Number.isFinite(value)) {
    if (value === 1) return "PASS";
    if (value === 0) return "NO PASS";
  }
  return null;
}

export function zoneSegmentsAhead(startM, coveredM, zones) {
  if (!isFiniteNumber(startM) || !isFiniteNumber(coveredM) || coveredM <= 0 || !Array.isArray(zones)) {
    return [];
  }
  const end = startM + coveredM;
  const segs = [];
  for (const zone of zones) {
    const a = Math.max(startM, Number(zone.start_m));
    const b = Math.min(end, Number(zone.end_m));
    if (Number.isFinite(a) && Number.isFinite(b) && b > a) {
      segs.push({
        type: zone.zone_type || "straight",
        start: a,
        end: b,
        length: b - a,
      });
    }
  }
  return segs;
}

export function wrapDistance(distance, length) {
  if (!isFiniteNumber(distance) || !isFiniteNumber(length) || length <= 0) return null;
  const wrapped = distance % length;
  return wrapped < 0 ? wrapped + length : wrapped;
}

export function attackerDistanceM(event, circuitLength) {
  if (isFiniteNumber(event?.track_distance_at_start_m) && isFiniteNumber(circuitLength)) {
    return wrapDistance(event.track_distance_at_start_m, circuitLength);
  }
  if (isFiniteNumber(event?.track_position_frac) && isFiniteNumber(circuitLength)) {
    return wrapDistance(event.track_position_frac * circuitLength, circuitLength);
  }
  return null;
}

export function targetDistanceM(event, circuitLength) {
  const attacker = attackerDistanceM(event, circuitLength);
  if (attacker == null || !isFiniteNumber(event?.gap_at_start_m)) return null;
  return wrapDistance(attacker + event.gap_at_start_m, circuitLength);
}

export function lookaheadWindowM(event) {
  if (!isFiniteNumber(event?.LookaheadCoveredM)) return null;
  return Math.max(0, Math.min(400, event.LookaheadCoveredM));
}

export function fracOnCircuit(distanceM, circuitLength) {
  if (!isFiniteNumber(distanceM) || !isFiniteNumber(circuitLength) || circuitLength <= 0) return null;
  return wrapDistance(distanceM, circuitLength) / circuitLength;
}
