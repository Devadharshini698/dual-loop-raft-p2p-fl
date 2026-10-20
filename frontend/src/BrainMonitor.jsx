import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const tones = { link: '#f4a65d', crash: '#ff6577', healthy: '#4de0a5', unresolved: '#8d9a96' }
const labels = { link: 'LINK FAILURE', crash: 'CRASH', healthy: 'HEALTHY-SLOW', unresolved: 'UNRESOLVED' }

export default function BrainMonitor({ state }) {
  return <section className="panel col-span-12 xl:col-span-7">
    <PanelTitle kicker="03 / DUAL-LOOP BRAIN" title="Telemetry & drift oscilloscope" aside={`ENCODER v${state.encoder_version}`} />
    <div className="grid gap-5 lg:grid-cols-[1fr_1.05fr]">
      <div>
        <p className="section-label">10-feature normalized vector · peer 3</p>
        <div className="mt-3 space-y-2">
          {state.features.map((feature, index) => <div className="feature-row" key={feature.name}>
            <span className="feature-index">{String(index + 1).padStart(2, '0')}</span>
            <span className="truncate text-[11px] text-mist/75">{feature.name}</span>
            <span className="feature-track"><i style={{ width: `${Math.max(2, feature.value * 100)}%` }} /></span>
            <span className="font-mono text-[10px] text-mist">{feature.value.toFixed(2)}</span>
          </div>)}
        </div>
      </div>
      <div>
        <p className="section-label">Split classifier confidence</p>
        <div className="mt-3 grid grid-cols-2 gap-2">
          {Object.entries(state.probabilities).map(([key, value]) => <div className="prob-card" key={key}>
            <div className="flex items-center justify-between gap-2">
              <span>{labels[key]}</span><b style={{ color: tones[key] }}>{Math.round(value * 100)}%</b>
            </div>
            <div className="prob-track"><i style={{ width: `${value * 100}%`, background: tones[key] }} /></div>
          </div>)}
        </div>
        <div className="mt-4 h-[170px] rounded-xl border border-white/5 bg-black/20 p-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={state.timeline} margin={{ top: 8, right: 8, left: -26, bottom: -8 }}>
              <XAxis dataKey="tick" tick={{ fill: '#687d76', fontSize: 9 }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 1]} tick={{ fill: '#687d76', fontSize: 9 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#0b1715', border: '1px solid #25443e', borderRadius: 10, fontSize: 11 }} />
              <ReferenceLine y={.72} stroke="#f4a65d" strokeDasharray="4 5" />
              <Line type="monotone" dataKey="drift" stroke="#a981ff" strokeWidth={2.4} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="error" stroke="#ff6577" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="accuracy" stroke="#4de0a5" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 flex gap-4 text-[10px] font-semibold uppercase tracking-wider text-mist/55">
          <Legend color="#a981ff" label="Drift"/><Legend color="#ff6577" label="Error"/><Legend color="#4de0a5" label="Accuracy"/>
          <span className="ml-auto">Recovery {state.recovery_sample}/23</span>
        </div>
      </div>
    </div>
  </section>
}

function Legend({ color, label }) {
  return <span className="flex items-center gap-1.5"><i className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />{label}</span>
}

export function PanelTitle({ kicker, title, aside }) {
  return <div className="mb-5 flex items-start justify-between gap-4">
    <div><p className="section-label">{kicker}</p><h2 className="mt-1 text-lg font-semibold tracking-tight text-white">{title}</h2></div>
    {aside && <span className="tag">{aside}</span>}
  </div>
}

