import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronLeft, ChevronRight, ChevronUp } from "lucide-react";
import DecisionStack from "./DecisionStack.jsx";
import TrackSchematic from "./TrackSchematic.jsx";
import {
  canDecide,
  eventLabel,
  formatExact,
  formatLap,
  formatSessionTime,
  formatSpeed,
  isFiniteNumber,
  raceLabel,
} from "../data/formatReplay.js";

const C = {
  attack: "#3dd68c",
  save: "#d4a017",
  stop: "#c45c66",
  muted: "#8b929c",
};

const DEMO_SCENARIOS = [
  { id: "australia_attack", title: "High opportunity" },
  { id: "energy_floor", title: "Energy veto" },
  { id: "spend_context", title: "400 m window veto" },
];

function metric3(value) {
  if (!isFiniteNumber(value)) return "—";
  return value.toFixed(3);
}

function metricPct(value) {
  if (!isFiniteNumber(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function speed(value) {
  const s = formatSpeed(value);
  return s == null ? "—" : `${s} km/h`;
}

function Stat({ k, v }) {
  return (
    <div className="aq-stat">
      <b>{k}</b>
      <span>{v}</span>
    </div>
  );
}

function padIndex(value, width = 2) {
  return String(value).padStart(width, "0");
}

export default function TrackShiftGlassDashboard() {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState(null);
  const [raceId, setRaceId] = useState("");
  const [eventId, setEventId] = useState("");
  const [view, setView] = useState("opportunity");
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [valDetailsOpen, setValDetailsOpen] = useState(false);
  const [browseOpen, setBrowseOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch("/replay.json")
      .then((res) => {
        if (!res.ok) throw new Error(`Replay artifact missing (${res.status}).`);
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        setPayload(data);
        const params = new URLSearchParams(window.location.search);
        const qEvent = params.get("event");
        const qRace = params.get("race");
        const qView = params.get("view");
        const initial = qEvent || data.smoke_test?.event_id || data.curated?.[0]?.event_id || data.events?.[0]?.event_id;
        const initialEvent = (data.events || []).find((row) => row.event_id === initial) || data.events?.[0];
        setRaceId(qRace || initialEvent?.race_id || "");
        setEventId(initialEvent?.event_id || "");
        if (qView === "validation" || qView === "opportunity") setView(qView);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || "Could not load replay.json");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const eventsByRace = useMemo(() => {
    const map = {};
    for (const event of payload?.events || []) {
      if (!map[event.race_id]) map[event.race_id] = [];
      map[event.race_id].push(event);
    }
    for (const id of Object.keys(map)) {
      map[id].sort((a, b) => {
        const lapA = a.event_start_lap ?? 0;
        const lapB = b.event_start_lap ?? 0;
        if (lapA !== lapB) return lapA - lapB;
        const tA = String(a.event_start_time || "");
        const tB = String(b.event_start_time || "");
        if (tA !== tB) return tA.localeCompare(tB);
        return String(a.event_id).localeCompare(String(b.event_id));
      });
    }
    return map;
  }, [payload]);

  const eventMap = useMemo(() => {
    const map = {};
    for (const event of payload?.events || []) map[event.event_id] = event;
    return map;
  }, [payload]);

  const event = eventMap[eventId] || null;
  const raceEvents = eventsByRace[raceId] || [];
  const circuit = payload?.circuits?.[raceId] || null;
  const decided = canDecide(event);
  const action = decided ? event.strategy_action : null;
  const actionColor = action === "ATTACK" ? C.attack : action === "SAVE" ? C.save : C.stop;
  const eventIndex = raceEvents.findIndex((row) => row.event_id === eventId);
  const eventOrdinal = eventIndex >= 0 ? eventIndex + 1 : 0;

  const laps = useMemo(() => {
    const set = new Set();
    for (const row of raceEvents) {
      if (isFiniteNumber(row.event_start_lap)) set.add(row.event_start_lap);
    }
    return [...set].sort((a, b) => a - b);
  }, [raceEvents]);

  const curatedById = useMemo(() => {
    const map = {};
    for (const item of payload?.curated || []) map[item.id] = item;
    return map;
  }, [payload]);

  const demoItems = DEMO_SCENARIOS.map((slot) => {
    const item = curatedById[slot.id];
    if (!item) return null;
    return { ...slot, ...item };
  }).filter(Boolean);

  function chooseRace(nextRaceId) {
    setRaceId(nextRaceId);
    const first = (eventsByRace[nextRaceId] || [])[0];
    setEventId(first?.event_id || "");
  }

  function chooseCurated(id) {
    const item = (payload?.curated || []).find((row) => row.id === id);
    if (!item) return;
    setRaceId(item.race_id);
    setEventId(item.event_id);
    setView("opportunity");
  }

  const chooseLap = useCallback((lap) => {
    const match = raceEvents.find((row) => row.event_start_lap === lap);
    if (match) setEventId(match.event_id);
  }, [raceEvents]);

  const stepEvent = useCallback((delta) => {
    if (!raceEvents.length || eventIndex < 0) return;
    const next = Math.min(raceEvents.length - 1, Math.max(0, eventIndex + delta));
    setEventId(raceEvents[next].event_id);
  }, [raceEvents, eventIndex]);

  useEffect(() => {
    function onKey(e) {
      if (view !== "opportunity") return;
      const tag = String(e.target?.tagName || "").toLowerCase();
      if (tag === "select" || tag === "input" || tag === "textarea") return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        stepEvent(-1);
      }
      if (e.key === "ArrowRight") {
        e.preventDefault();
        stepEvent(1);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [view, stepEvent]);

  if (error) {
    return (
      <div className="aq-app">
        <div className="aq-brand">APEXIQ</div>
        <p style={{ color: C.stop, marginTop: 16 }}>Insufficient data for strategy decision.</p>
        <p style={{ color: C.muted, marginTop: 8 }}>{error} Run python scripts/build_frontend_replay.py</p>
      </div>
    );
  }

  if (!payload) {
    return <div className="aq-app" style={{ color: C.muted }}>Loading historical replay…</div>;
  }

  const selectedLap = event?.event_start_lap;
  const minLap = laps[0];
  const maxLap = laps[laps.length - 1];
  const lapSpan = minLap == null || maxLap == null || maxLap === minLap ? 1 : maxLap - minLap;

  return (
    <div className="aq-app">
      <div className="aq-shell">
        <header className="aq-top">
          <div>
            <div className="aq-brand">APEXIQ</div>
            <div className="aq-tagline">Overtake opportunity → ATTACK or SAVE</div>
            {view === "opportunity" ? (
              <>
                <div className="aq-racehead">
                  {event?.circuit || "—"}
                  {event?.split === "locked_test" ? " · locked confirmation" : ""}
                </div>
                <div className="aq-lap">
                  Lap {formatLap(event?.event_start_lap) ?? "—"}
                  {event?.attacker && event?.target ? ` · ${event.attacker} → ${event.target}` : ""}
                </div>
              </>
            ) : (
              <div className="aq-racehead">How reliable is the opportunity model?</div>
            )}
          </div>
          <div className="aq-nav">
            <span className="aq-badge">Historical replay · frozen model</span>
            <button className={`aq-tab${view === "opportunity" ? " is-on" : ""}`} onClick={() => setView("opportunity")}>
              Race Replay
            </button>
            <button className={`aq-tab${view === "validation" ? " is-on" : ""}`} onClick={() => setView("validation")}>
              Model Validation
            </button>
          </div>
        </header>

        {view === "opportunity" ? (
          <>
            <div className="aq-replay">
              <label className="aq-field aq-race-field">
                <span>Race</span>
                <select value={raceId} onChange={(e) => chooseRace(e.target.value)}>
                  {(payload.races || []).map((race) => (
                    <option key={race.race_id} value={race.race_id}>
                      {raceLabel(race)}
                    </option>
                  ))}
                </select>
              </label>

              <div className="aq-controller">
                <span className="aq-k">Race replay</span>
                <div className="aq-controller-row">
                  <button className="aq-step" onClick={() => stepEvent(-1)} disabled={eventIndex <= 0} aria-label="Previous event">
                    <ChevronLeft size={18} /> Previous
                  </button>
                  <div className="aq-controller-pos">
                    EVENT {padIndex(eventOrdinal, 3)} / {padIndex(raceEvents.length, 3)}
                  </div>
                  <button className="aq-step" onClick={() => stepEvent(1)} disabled={eventIndex < 0 || eventIndex >= raceEvents.length - 1} aria-label="Next event">
                    Next <ChevronRight size={18} />
                  </button>
                </div>
              </div>
            </div>

            {laps.length > 1 ? (
              <div className="aq-timeline" role="slider" aria-label="Lap timeline" aria-valuemin={minLap} aria-valuemax={maxLap} aria-valuenow={selectedLap}>
                <div className="aq-timeline-labels">
                  <span>Lap {formatLap(minLap)}</span>
                  <span>Lap {formatLap(selectedLap) ?? "—"}</span>
                  <span>Lap {formatLap(maxLap)}</span>
                </div>
                <div
                  className="aq-timeline-track"
                  onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    const x = (e.clientX - rect.left) / Math.max(1, rect.width);
                    const guessed = Math.round(minLap + x * lapSpan);
                    const nearest = laps.reduce((best, lap) => (Math.abs(lap - guessed) < Math.abs(best - guessed) ? lap : best), laps[0]);
                    chooseLap(nearest);
                  }}
                >
                  {laps.map((lap) => (
                    <i
                      key={lap}
                      className={lap === selectedLap ? "is-on" : ""}
                      style={{ left: `${((lap - minLap) / lapSpan) * 100}%` }}
                    />
                  ))}
                  {isFiniteNumber(selectedLap) ? (
                    <b style={{ left: `${((selectedLap - minLap) / lapSpan) * 100}%` }} />
                  ) : null}
                </div>
              </div>
            ) : null}

            <div className="aq-scenarios">
              <span className="aq-k">Demo scenarios</span>
              <div className="aq-scenario-row">
                {demoItems.map((item) => (
                  <button
                    key={item.id}
                    className={`aq-scenario${eventId === item.event_id ? " is-on" : ""}`}
                    onClick={() => chooseCurated(item.id)}
                  >
                    {item.title}
                  </button>
                ))}
              </div>
            </div>

            <div className="aq-stage">
              <section className="aq-map">
                <TrackSchematic event={event} circuit={circuit} />
              </section>
              <aside className="aq-rail">
                <DecisionStack event={event} decided={decided} action={action} actionColor={actionColor} />
                <div className="aq-meta" style={{ marginTop: 10 }}>
                  Frozen GBDT · values at event t0. DEFEND is not implemented.
                </div>
              </aside>
            </div>

            <div className="aq-details">
              <button className="aq-toggle" onClick={() => setDetailsOpen((v) => !v)}>
                Technical details
                {detailsOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>
              {detailsOpen && event && (
                <>
                  <div className="aq-stats">
                    <Stat k="event_id" v={event.event_id} />
                    <Stat k="Session timestamp" v={formatSessionTime(event.event_start_time) || "—"} />
                    <Stat k="track_position_frac" v={isFiniteNumber(event.track_position_frac) ? event.track_position_frac.toFixed(3) : "—"} />
                    <Stat k="track_distance_at_start_m" v={isFiniteNumber(event.track_distance_at_start_m) ? `${event.track_distance_at_start_m.toFixed(1)} m` : "—"} />
                    <Stat k="exact p_hat" v={formatExact(event.p_hat) || "—"} />
                    <Stat k="strategy_reason_code" v={event.strategy_reason_code || "—"} />
                    <Stat k="Attacker speed" v={speed(event.attacker_speed_at_start_kmh)} />
                    <Stat k="Target speed at same point" v={speed(event.target_speed_at_attacker_position_kmh)} />
                    <Stat k="Closing speed (+ catching)" v={speed(event.closing_speed_at_start_kmh)} />
                    <Stat k="Relative speed (att. − tgt)" v={speed(event.relative_speed_same_position_kmh)} />
                    <Stat k="Spatial time gap" v={isFiniteNumber(event.spatial_time_gap_s) ? `${event.spatial_time_gap_s.toFixed(2)} s` : "—"} />
                    <Stat k="Compound" v={event.compound || "—"} />
                    <Stat k="Tyre age" v={isFiniteNumber(event.tyre_age) ? String(event.tyre_age) : "—"} />
                    <Stat k="Stint" v={isFiniteNumber(event.stint) ? String(event.stint) : "—"} />
                  </div>
                  <button className="aq-toggle aq-toggle-sub" onClick={() => setBrowseOpen((v) => !v)}>
                    Browse events
                    {browseOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </button>
                  {browseOpen ? (
                    <label className="aq-field aq-browse">
                      <span>All events in this race</span>
                      <select value={eventId} onChange={(e) => setEventId(e.target.value)}>
                        {raceEvents.map((row) => (
                          <option key={row.event_id} value={row.event_id}>
                            {eventLabel(row)}
                          </option>
                        ))}
                      </select>
                    </label>
                  ) : null}
                </>
              )}
            </div>
          </>
        ) : (
          <ValidationView payload={payload} open={valDetailsOpen} setOpen={setValDetailsOpen} />
        )}

        <div className="aq-foot">APEXIQ · Historical replay · frozen model</div>
      </div>
    </div>
  );
}

function BarRow({ label, value, max, selected, text, accent }) {
  const pct = isFiniteNumber(value) && isFiniteNumber(max) && max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  const color = accent || (selected ? C.attack : "#6d8296");
  return (
    <div className={`aq-barrow${selected ? " is-on" : ""}`}>
      <div className="aq-barrow-label">{label}</div>
      <div className="aq-barrow-track">
        <i style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="aq-barrow-val">{text}</div>
    </div>
  );
}

function ValidationView({ payload, open, setOpen }) {
  const m = payload.metrics || {};
  const loro = m.development_loro || {};
  const locked = m.locked_test || {};
  const st = m.strategy || {};
  const ds = m.dataset || {};
  const hp = m.hyperparameters || {};
  const comparison = [...(m.model_comparison || [])].sort((a, b) => (b.pr_auc || 0) - (a.pr_auc || 0));
  const maxPr = Math.max(...comparison.map((row) => row.pr_auc || 0), 0.001);
  const ablation = [
    { name: "P only", value: st.ablation_attack_rate?.p_only },
    { name: "P + energy", value: st.ablation_attack_rate?.p_energy },
    { name: "P + energy + 400m", value: st.ablation_attack_rate?.p_energy_400m, highlight: true },
  ];
  const maxAb = Math.max(...ablation.map((row) => row.value || 0), 0.001);

  return (
    <div className="aq-val">
      <section className="aq-panel">
        <h1>Model validation</h1>
        <div className="aq-val-model">Shallow GBDT · frozen historical replay</div>
        <div className="aq-q">How good is the model?</div>
        <div className="aq-metrics">
          <div className="aq-metric">
            <div className="aq-k">PR-AUC</div>
            <div className="aq-v">{metric3(loro.pr_auc)}</div>
            <p>Ranking quality for the minority positive class.</p>
          </div>
          <div className="aq-metric">
            <div className="aq-k">Brier</div>
            <div className="aq-v">{metric3(loro.brier)}</div>
            <p>Probability quality; lower is better.</p>
          </div>
          <div className="aq-metric">
            <div className="aq-k">ROC-AUC</div>
            <div className="aq-v">{metric3(loro.roc_auc)}</div>
            <p>Ranking metric, not accuracy.</p>
          </div>
          <div className="aq-metric">
            <div className="aq-k">Log loss</div>
            <div className="aq-v">{metric3(loro.log_loss)}</div>
            <p>Penalizes incorrect probability estimates.</p>
          </div>
        </div>
      </section>

      <section className="aq-panel">
        <div className="aq-q">Why was this model selected?</div>
        <div className="aq-barlist">
          {comparison.map((row) => (
            <BarRow
              key={row.id}
              label={row.name}
              value={row.pr_auc}
              max={maxPr}
              selected={row.selected}
              text={`${metric3(row.pr_auc)} PR-AUC · ${metric3(row.brier)} Brier${row.selected ? " · selected" : ""}`}
            />
          ))}
        </div>
        <div className="aq-meta">Leave-one-race-out on 7 development races. Lower Brier is better.</div>
      </section>

      <section className="aq-panel">
        <div className="aq-split2">
          <div>
            <div className="aq-q">Did it generalize to unseen races?</div>
            <div className="aq-flow">
              <b>7 development races</b>
              <em>→</em>
              <b>Leave-one-race-out</b>
              <em>→</em>
              <b>Model frozen</b>
              <em>→</em>
              <b>Belgium + Miami locked confirmation</b>
            </div>
            <div className="aq-k" style={{ marginTop: 14 }}>Locked confirmation</div>
            <div className="aq-locked">
              <Stat k="Belgium PR-AUC" v={metric3(locked.belgium_pr_auc)} />
              <Stat k="Miami PR-AUC" v={metric3(locked.miami_pr_auc)} />
            </div>
            <div className="aq-meta">Not used for model selection.</div>
          </div>
          <div>
            <div className="aq-q">What does the strategy layer add?</div>
            <div className="aq-barlist">
              {ablation.map((row) => (
                <BarRow
                  key={row.name}
                  label={row.name}
                  value={row.value}
                  max={maxAb}
                  selected={Boolean(row.highlight)}
                  accent={row.highlight ? C.save : "#6d8296"}
                  text={metricPct(row.value)}
                />
              ))}
            </div>
            <div className="aq-meta">
              The immediate 400m context creates the largest reduction in ATTACK decisions. This does not prove strategy optimality.
            </div>
          </div>
        </div>
      </section>

      <section>
        <button className="aq-toggle" onClick={() => setOpen((v) => !v)}>
          Technical details
          {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        {open && (
          <div className="aq-stats">
            <Stat k="Labelled events" v={ds.n_labelled ?? "—"} />
            <Stat k="Positives" v={ds.n_pos ?? "—"} />
            <Stat k="Negatives" v={ds.n_neg ?? "—"} />
            <Stat k="Pooled locked Brier" v={metric3(locked.pooled_brier)} />
            <Stat k="Pooled locked PR-AUC" v={metric3(locked.pooled_pr_auc)} />
            <Stat k="GBDT max_depth" v={hp.max_depth ?? "—"} />
            <Stat k="GBDT max_iter" v={hp.max_iter ?? "—"} />
            <Stat k="Learning rate" v={hp.learning_rate ?? "—"} />
            <Stat k="Development ATTACK" v={metricPct(st.development_attack_rate)} />
            <Stat k="Locked-test ATTACK" v={metricPct(st.locked_test_attack_rate)} />
          </div>
        )}
      </section>
    </div>
  );
}
