import { CheckCircle2, CircleDashed, Clock3, DatabaseZap, ShieldCheck } from 'lucide-react'
import { PanelTitle } from './BrainMonitor'

export default function EvidencePanel({ state }) {
  const contract = state.dual_loop.self_labeler
  const queued = state.label_queue[0]
  const item = queued || (contract && { peer: contract.peer_id, status: contract.resolved_label === 'UNRESOLVED' ? 'discarded' : contract.resolved_label ? 'resolved' : 'pending',
    remaining_ms: contract.horizon_remaining_ms, boot_id: contract.boot_id_delta ? 4 : 3,
    evidence: contract.boot_id_delta ? 'Boot ID 4 → 5' : contract.witness_ack ? 'Witness alive · same Boot ID' : 'Awaiting protocol evidence',
    resolved_label: contract.resolved_label, confidence: contract.confidence })
  const planner = state.dual_loop.planner_and_fl
  const verdict = item?.resolved_label || (state.scenario === 'leader' ? 'CRASH' : state.scenario === 'isolation' ? 'UNRESOLVED' : 'LINK_FAILURE')
  return <section className="panel col-span-12 xl:col-span-5">
    <PanelTitle kicker="04 / PROTOCOL EVIDENCE" title="Frozen suspicion queue" aside="HORIZON H = 2,000 MS" />
    {item ? <div className={`evidence-card ${item.status}`}>
      <div className="flex items-start justify-between gap-4">
        <div className="flex gap-3"><span className="evidence-icon"><DatabaseZap size={18}/></span>
          <div><p className="section-label">TARGET PEER</p><h3>{item.peer.toUpperCase()} · BOOT #{item.boot_id}</h3></div>
        </div>
        <span className="countdown"><Clock3 size={13}/>{item.remaining_ms} ms</span>
      </div>
      <div className="evidence-flow">
        <EvidenceStep done title="Suspicion frozen" detail="xₜ copied, not mutated" />
        <EvidenceStep done={item.status !== 'pending'} title="Protocol evidence" detail={item.evidence} />
        <EvidenceStep done={item.status !== 'pending'} title={item.status === 'discarded' ? 'UNRESOLVED' : 'Verdict committed'} detail={item.status === 'discarded' ? 'Excluded from training' : 'Confidence-weighted label'} />
      </div>
      {item.status !== 'pending' && <div className={`stamp ${item.status}`}>{item.status === 'discarded' ? 'UNRESOLVED · DISCARDED' : `${verdict} · ${verdict === 'CRASH' ? '1.00' : '0.91'}`}</div>}
    </div> : <div className="empty-state"><CircleDashed/><p>No active suspicion</p><span>Inject a fault to freeze telemetry evidence.</span></div>}
    <div className="timeout-gauge">
      <div><span>Safety boundary</span><b>{planner.recommended_timeout_ms} ms</b></div>
      <div className="gauge-track"><i style={{ width: `${Math.max(0, Math.min(100, (planner.recommended_timeout_ms - 135) / 365 * 100))}%` }}/><em /></div>
      <div className="gauge-labels"><span>135 ms floor</span><span>500 ms link clamp</span></div>
    </div>
    <div className="mt-4 grid grid-cols-2 gap-3">
      <Kpi icon={ShieldCheck} value={`${(state.metrics.label_noise * 100).toFixed(2)}%`} label="Label noise" />
      <Kpi icon={CheckCircle2} value={`${(state.metrics.cold_start_f1 * 100).toFixed(1)}%`} label="Cold-start F1" />
    </div>
  </section>
}

function EvidenceStep({ done, title, detail }) {
  return <div className={done ? 'done' : ''}><i>{done ? <CheckCircle2 size={14}/> : <CircleDashed size={14}/>}</i><span><b>{title}</b><small>{detail}</small></span></div>
}

function Kpi({ icon: Icon, value, label }) {
  return <div className="kpi"><Icon size={16}/><div><strong>{value}</strong><span>{label}</span></div></div>
}

