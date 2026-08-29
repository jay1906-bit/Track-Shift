import React from "react";
import {
  contextHeadline,
  formatEnergy,
  formatGapM,
  formatProbabilityPct,
  formatSharePct,
  historicalOutcome,
  isFiniteNumber,
  isValidProbability,
} from "../data/formatReplay.js";

export default function DecisionStack({ event, decided, action, actionColor }) {
  const p = formatProbabilityPct(event?.p_hat);
  const energy = formatEnergy(event?.EstimatedEnergyIndex);
  const ctx = contextHeadline(event);
  const gap = formatGapM(event?.gap_at_start_m);
  const deployPct = formatSharePct(event?.strategy_deploy_share);
  const covered = isFiniteNumber(event?.LookaheadCoveredM)
    ? Math.round(Math.min(400, Math.max(0, event.LookaheadCoveredM)))
    : null;
  const energyPct = isFiniteNumber(event?.EstimatedEnergyIndex)
    ? Math.max(0, Math.min(100, event.EstimatedEnergyIndex * 100))
    : 0;
  const pPct = isValidProbability(event?.p_hat) ? event.p_hat * 100 : 0;
  const windowTone = event?.strategy_spend_compatible ? "#3dd68c" : event?.strategy_spend_compatible === false ? "#d4a017" : "#8b929c";
  const outcome = historicalOutcome(event);

  return (
    <div className="aq-story">
      <div className="aq-story-step">
        <div className="aq-k">Battle</div>
        <div className="aq-pair">
          {event?.attacker || "—"} → {event?.target || "—"}
        </div>
        <div className="aq-gap">{gap == null ? "Gap unavailable" : `${gap} m`}</div>
      </div>

      <div className="aq-story-arrow" aria-hidden="true">↓</div>

      <div className="aq-story-step">
        <div className="aq-k">Opportunity</div>
        <div className="aq-v">{p == null ? "—" : `${p}%`}</div>
        <div className="aq-caption">P(pass within 30s)</div>
        <div className="aq-bar">
          <i style={{ width: `${pPct}%`, background: "#3dd68c" }} />
        </div>
        <div className="aq-meta">Frozen GBDT · historical replay</div>
      </div>

      <div className="aq-story-arrow" aria-hidden="true">↓</div>

      <div className="aq-story-step">
        <div className="aq-k">Resource</div>
        <div className="aq-v">
          {energy == null ? "—" : energy} <span>/ 1.00</span>
        </div>
        <div className="aq-caption">Simulated Energy Index</div>
        <div className="aq-bar">
          <i style={{ width: `${energyPct}%`, background: "#6d8296" }} />
        </div>
        <div className="aq-meta">Not battery / SOC</div>
      </div>

      <div className="aq-story-arrow" aria-hidden="true">↓</div>

      <div className="aq-story-step">
        <div className="aq-k">Next 400 m</div>
        <div className="aq-v aq-v-sm" style={{ color: windowTone }}>{event ? ctx : "—"}</div>
        <div className="aq-window-row">
          <span>Zone {event?.zone_type || "—"}</span>
          <span>{deployPct == null ? "Deploy —" : `${deployPct}% deploy`}</span>
          <span>{covered == null ? "Coverage —" : `${covered} m`}{event?.lookahead_truncated ? " · truncated" : ""}</span>
        </div>
      </div>

      <div className="aq-story-arrow" aria-hidden="true">↓</div>

      <div className="aq-strategy">
        <div className="aq-k">Strategy decision</div>
        <div className="aq-strategy-action" style={{ color: actionColor }}>
          {decided ? action : "NO DECISION"}
        </div>
        <div className="aq-reason">
          {decided ? event.strategy_reason : "Insufficient data for strategy decision."}
        </div>
      </div>

      {outcome ? (
        <>
          <div className="aq-story-arrow" aria-hidden="true">↓</div>
          <div className="aq-outcome">
            <div className="aq-k">Historical outcome</div>
            <div className="aq-outcome-value">{outcome}</div>
            <div className="aq-meta">Historical outcome · not used by the decision</div>
          </div>
        </>
      ) : null}
    </div>
  );
}
