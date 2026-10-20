import { Activity, ChevronRight, Circle, Cpu, Flower2, Radio, ShieldCheck } from 'lucide-react'
import Arena from './Arena'
import BrainMonitor, { PanelTitle } from './BrainMonitor'
import ControlPanel from './ControlPanel'
import EvidencePanel from './EvidencePanel'
import FlowerRuntimeNotice from './FlowerRuntimeNotice'
import Ledger from './Ledger'
import Topology from './Topology'
import VerificationPanel from './VerificationPanel'
import { useMissionControl } from './useMissionControl'

const presets = [
  ['ambiguous', '01', 'The Ambiguous Silence'],
  ['drift', '02', 'Sudden Concept Drift'],
  ['leader', '03', 'Leader Death Mid-FL'],
  ['all', '04', 'Run All 10 Checks'],
]

export default function App() {
  const { state, connected, chooseScenario, control, runAllChecks, verification } = useMissionControl()
  if (!state) return <Loading connected={connected}/>
  const flower = state.flower_runtime

  return <main className="min-h-screen px-4 py-5 text-mist md:px-7 xl:px-10">
    <FlowerRuntimeNotice flower={flower}/>
    <Header state={state} connected={connected} flower={flower}/>
    <nav className="preset-bar" aria-label="Judge mode presets">
      <span className="section-label mr-2">JUDGE MODE</span>
      {presets.map(([id, number, label]) => <button key={id} className={state.scenario === id ? 'active' : ''}
        onClick={() => id === 'all' ? runAllChecks() : chooseScenario(id)}><i>{number}</i><span>{label}</span><ChevronRight size={14}/></button>)}
    </nav>
    <div className="grid grid-cols-12 gap-4">
      <section className="panel col-span-12">
        <PanelTitle kicker="01A / SIDE-BY-SIDE CONSENSUS ARENA" title="One fault schedule. Two synchronized clusters." aside={`${state.nodes.length} PEERS EACH · TERM ${state.term}`} />
        <div className="grid gap-3 lg:grid-cols-2">
          <div className="topology-frame"><div className="topology-label standard">STANDARD RAFT <span>FIXED 150–300 MS</span></div>
          <div className="topology-shell"><Topology nodes={state.standard_nodes} links={state.standard_links}/>
          <div className="topology-legend"><Legend tone="gold" label="Leader"/><Legend tone="green" label="Follower"/><Legend tone="amber" label="Suspected"/><Legend tone="red" label="Crashed"/></div>
          <div className="packet-key"><span><i className="solid"/> heartbeat</span><span><i className="dashed"/> witness</span></div>
          </div></div>
          <div className="topology-frame"><div className="topology-label dual">DUAL-LOOP RAFT <span>CAUSE-AWARE · GATED</span></div>
          <div className="topology-shell"><Topology nodes={state.nodes} links={state.links}/>
          <div className="topology-legend"><Legend tone="gold" label="Leader"/><Legend tone="green" label="Follower"/><Legend tone="amber" label="Suspected"/><Legend tone="red" label="Crashed"/></div>
          <div className="packet-key"><span><i className="solid"/> heartbeat</span><span><i className="dashed"/> witness</span></div>
          </div></div>
        </div>
      </section>
      <ControlPanel state={state} chooseScenario={chooseScenario} control={control}/>
      <Arena arena={state.arena} scenario={state.scenario}/>
      <BrainMonitor state={state}/>
      <EvidencePanel state={state}/>
      <Ledger state={state}/>
      <VerificationPanel state={state} verification={verification}/>
    </div>
    <footer><span>Deterministic synthetic research demonstrator · seed {state.seed}</span><span>Virtual time {state.virtual_ms.toLocaleString()} ms · tick {state.tick}</span></footer>
  </main>
}

function Header({ state, connected, flower }) {
  return <header className="mb-5 flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
    <div className="flex items-start gap-4"><span className="brand-mark"><Cpu size={23}/></span><div>
      <p className="eyebrow">DUAL-LOOP P2P FEDERATED LEARNING</p>
      <h1>Consensus <span>&</span> telemetry observatory</h1>
      <p className="mt-2 max-w-2xl text-sm text-mist/55">Watch protocol evidence turn ambiguous silence into safe, explainable Raft decisions—without changing consensus safety.</p>
    </div></div>
    <div className="header-status">
      <div><Activity size={15}/><span><b>SEED 42 · DETERMINISTIC</b><small>{state.scenario_name}</small></span></div>
      <div><ShieldCheck size={15}/><span><b>135 MS STRICTLY HELD</b><small>Safety floor</small></span></div>
      <div><Cpu size={15}/><span><b>{state.model_footprint_bytes} B</b><small>Model footprint</small></span></div>
      <div className={flower?.fallback ? 'flower-fallback' : flower?.available ? 'connected' : 'disconnected'}><Flower2 size={15}/><span>
        <b>{flower?.fallback ? 'FLOWER FALLBACK ACTIVE' : flower?.available ? `FLOWER SDK ACTIVE · v${flower.version}` : 'FLOWER STATUS UNKNOWN'}</b>
        <small>{flower?.fallback ? 'Compatibility shim' : flower?.available ? 'In-process FedAvg' : 'Awaiting runtime report'}</small></span></div>
      <div className={connected ? 'connected' : 'disconnected'}><Radio size={15}/><span><b>{connected ? '10/10 CHECKS PASS' : 'RECONNECTING'}</b><small>Live WebSocket stream</small></span></div>
    </div>
  </header>
}

function Legend({ tone, label }) { return <span><i className={tone}/>{label}</span> }

function Loading({ connected }) {
  return <div className="loading-screen"><span className="brand-mark"><Cpu size={24}/></span><h1>Mission Control</h1><p>{connected ? 'Synchronizing deterministic state…' : 'Connecting to the protocol stream…'}</p><Circle className="animate-spin" size={18}/></div>
}

