import { ArrowRight, RadioTower } from 'lucide-react'
import { PanelTitle } from './BrainMonitor'

export default function Ledger({ state }) {
  return <section className="panel col-span-12 xl:col-span-7">
    <PanelTitle kicker="05 / SAFETY & HANDOVER" title="Decision ledger" aside={`TERM ${state.term}`} />
    <div className="ledger">
      {state.ledger.map((event, index) => <div className="ledger-row" key={`${event.tick}-${event.title}`}>
        <span className="ledger-time">T+{String(event.tick * 50).padStart(4, '0')}</span>
        <i className={`ledger-dot ${event.tone}`} />
        <div><strong>{event.title}</strong><p>{event.detail}</p></div>
        {index === 0 && <span className="latest">LATEST</span>}
      </div>)}
    </div>
    <div className="mt-4 grid gap-3 sm:grid-cols-3">
      <SafetyCard label="Safety floor" value={`${state.metrics.safety_floor_ms.toFixed(0)} ms`} note="Strictly held" />
      <SafetyCard label="Buffered updates" value={state.handover_updates} note="No client work lost" />
      <SafetyCard label="Active leader" value={state.leader.toUpperCase()} note={`Term ${state.term}`} />
    </div>
    <div className="mt-4 flex items-center gap-2 rounded-xl border border-emerald/15 bg-emerald/[.05] px-4 py-3 text-xs text-emerald">
      <RadioTower size={15}/><b>Raft voting and log safety remain unchanged.</b><ArrowRight size={14}/><span className="text-mist/60">ML advice changes timing and provisional gating only.</span>
    </div>
  </section>
}

function SafetyCard({ label, value, note }) {
  return <div className="safety-card"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>
}

