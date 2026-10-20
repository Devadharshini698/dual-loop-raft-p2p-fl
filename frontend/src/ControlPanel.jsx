import { Activity, Pause, Play, RotateCcw, ShieldAlert, Skull, StepForward, Waves } from 'lucide-react'
import { PanelTitle } from './BrainMonitor'

const actions = [
  { id: 'ambiguous', icon: ShieldAlert, title: 'Inject asymmetric loss', note: 'Same Boot ID · witnesses alive', tone: 'amber' },
  { id: 'leader', icon: Skull, title: 'Crash & reboot leader', note: 'Provisional handover path', tone: 'coral' },
  { id: 'drift', icon: Waves, title: 'Trigger Regime B drift', note: 'ADWIN spike · 3× local LR', tone: 'violet' },
  { id: 'isolation', icon: Activity, title: 'Isolate local observer', note: 'UNRESOLVED · discard safely', tone: 'slate' },
]

export default function ControlPanel({ state, chooseScenario, control }) {
  const speeds = [.25, 1, 2]
  const speedIndex = Math.max(0, speeds.indexOf(state.speed))
  return <section className="panel col-span-12 xl:col-span-5">
    <PanelTitle kicker="01B / CHAOS INJECTION" title="Switchboard" aside="DETERMINISTIC" />
    <div className="space-y-2.5">
      {actions.map(({ id, icon: Icon, title, note, tone }) => <button key={id}
        className={`chaos-button ${tone} ${state.scenario === id ? 'active' : ''}`}
        onClick={() => chooseScenario(id)}>
        <span className="chaos-icon"><Icon size={18}/></span>
        <span><b>{title}</b><small>{note}</small></span>
        <i>{state.scenario === id ? 'ARMED' : 'RUN'}</i>
      </button>)}
    </div>
    <div className="mt-5 rounded-2xl border border-white/5 bg-black/20 p-4">
      <div className="mb-3 flex items-center justify-between">
        <p className="section-label">Playback</p><span className="font-mono text-xs text-mist">1 tick = 50 virtual ms</span>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button className="icon-button" aria-label={state.playing ? 'Pause simulation' : 'Play simulation'}
          onClick={() => control(state.playing ? 'pause' : 'play')}>
          {state.playing ? <Pause size={16}/> : <Play size={16}/>}<span>{state.playing ? 'PAUSE' : 'PLAY'}</span>
        </button>
        <button className="icon-button" onClick={() => control('step', 1)}><StepForward size={16}/><span>+1</span></button>
        <button className="icon-button" onClick={() => control('step', 10)}><StepForward size={16}/><span>+10</span></button>
        <button className="icon-button" onClick={() => control('reset')}><RotateCcw size={16}/><span>SEED 42</span></button>
        <label className="speed-slider"><span>{speeds[speedIndex]}×</span><input aria-label="Playback speed" type="range" min="0" max="2" step="1" value={speedIndex}
          onChange={(event) => control('play', speeds[Number(event.target.value)])}/></label>
      </div>
    </div>
  </section>
}

