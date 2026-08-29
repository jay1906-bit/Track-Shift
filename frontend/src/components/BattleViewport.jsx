import React from "react";
import {
  attackerDistanceM,
  contextHeadline,
  formatGapM,
  formatSharePct,
  lookaheadWindowM,
  zoneSegmentsAhead,
} from "../data/formatReplay.js";

const ZONE = {
  straight: "#5a6570",
  brake: "#a9844a",
  corner: "#6d8296",
};

export default function BattleViewport({ event, circuit }) {
  const length = circuit?.circuit_length_m;
  const attackerM = attackerDistanceM(event, length);
  const windowM = lookaheadWindowM(event);
  const gap = formatGapM(event?.gap_at_start_m);
  const segs = zoneSegmentsAhead(attackerM, windowM ?? 0, circuit?.zones || []);
  const covered = windowM == null ? null : Math.round(windowM);
  const deployPct = formatSharePct(event?.strategy_deploy_share);
  const ctx = event ? contextHeadline(event) : "—";
  const truncated = Boolean(event?.lookahead_truncated);
  const totalSeg = segs.reduce((sum, seg) => sum + seg.length, 0) || 1;

  return (
    <div className="aq-battle">
      <div className="aq-map-label">Battle</div>
      <div className="aq-battle-grid">
        <div className="aq-battle-pair" aria-label="Attacker and target">
          <div className="aq-battle-car is-target">
            <span className="aq-k">Target</span>
            <b>{event?.target || "—"}</b>
            <i />
          </div>
          <div className="aq-battle-gap">
            <em />
            <strong>{gap == null ? "Gap unavailable" : `${gap} m`}</strong>
            <em />
          </div>
          <div className="aq-battle-car is-attacker">
            <i />
            <b>{event?.attacker || "—"}</b>
            <span className="aq-k">Attacker</span>
          </div>
        </div>

        <div className="aq-battle-window">
          <div className="aq-k">Next 400 m</div>
          <div className="aq-battle-strip" role="img" aria-label="Next 400 metre track mix">
            {segs.length ? (
              segs.map((seg, i) => (
                <i
                  key={`${seg.start}-${i}`}
                  style={{
                    flexGrow: seg.length / totalSeg,
                    background: ZONE[seg.type] || ZONE.straight,
                  }}
                  title={`${seg.type} · ${Math.round(seg.length)} m`}
                />
              ))
            ) : (
              <i className="aq-battle-strip-empty" />
            )}
          </div>
          <div className="aq-battle-window-copy">
            <b>{ctx}</b>
            <span>
              {covered == null ? "Coverage unavailable" : truncated ? `${covered} m available` : `${covered} m`}
              {deployPct == null ? "" : ` · ${deployPct}% deploy`}
              {event?.zone_type ? ` · ${event.zone_type}` : ""}
            </span>
          </div>
        </div>
      </div>
      <div className="aq-track-meta">Along-track schematic from gap and zone table · not live GPS</div>
    </div>
  );
}
