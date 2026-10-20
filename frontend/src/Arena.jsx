import { Activity, ShieldCheck, TriangleAlert } from 'lucide-react'
import { PanelTitle } from './BrainMonitor'

export default function Arena({ arena, scenario }) {
  return <section className="panel col-span-12 overflow-hidden xl:col-span-7">
    <PanelTitle kicker="02 / CONTROLLED COMPARISON" title="Same fault. Two protocol responses." aside="SYNCHRONIZED INPUT" />
    <div className="arena-grid">
      <ProtocolCard title="STANDARD RAFT" icon={TriangleAlert} tone="danger" data={arena.standard}
        subtitle="Timing-only suspicion" scenario={scenario} />
      <div className="versus"><span>VS</span></div>
      <ProtocolCard title="DUAL-LOOP RAFT" icon={ShieldCheck} tone="success" data={arena.dual}
        subtitle="Cause-aware · safety bounded" scenario={scenario} />
    </div>
  </section>
}

function ProtocolCard({ title, subtitle, icon: Icon, tone, data, scenario }) {
  const isDual = tone === 'success'
  const verdict = scenario === 'leader' ? 'CRASH' : scenario === 'drift' ? 'HEALTHY-SLOW' : scenario === 'isolation' ? 'UNRESOLVED' : 'LINK_FAILURE'
  return <div className={`protocol-card ${tone}`}>
    <div className="flex items-center gap-3">
      <span className="protocol-icon"><Icon size={17}/></span>
      <div><h3>{title}</h3><p>{subtitle}</p></div>
      <span className={`ml-auto live-dot ${tone}`}>LIVE</span>
    </div>
    <div className="mt-5 grid grid-cols-3 gap-3">
      <Metric value={data.spurious_handovers} label="Spurious handovers" />
      <Metric value={`${data.stall_ms} ms`} label="FL stall" />
      <Metric value={data.spurious_per_hour.toLocaleString()} label="Elections / hour" compact />
    </div>
    <div className="mt-4 flex items-center gap-3 rounded-xl border border-white/5 bg-black/20 p-3">
      <Activity size={16} className={isDual ? 'text-emerald' : 'text-coral'} />
      <span className="text-[11px] uppercase tracking-[.12em] text-mist/55">Decision</span>
      <strong className={isDual ? 'text-emerald' : 'text-coral'}>{isDual ? verdict : 'TIMEOUT ONLY'}</strong>
    </div>
  </div>
}

function Metric({ value, label, compact }) {
  return <div><strong className={compact ? 'text-sm' : 'text-2xl'}>{value}</strong><span>{label}</span></div>
}

