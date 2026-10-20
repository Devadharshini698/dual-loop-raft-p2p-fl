import { useCallback, useEffect, useRef, useState } from 'react'

const apiRoot = import.meta.env.DEV ? 'http://127.0.0.1:8501' : ''
const wsRoot = import.meta.env.DEV
  ? 'ws://127.0.0.1:8501'
  : `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`

export function useMissionControl() {
  const [state, setState] = useState(null)
  const [connected, setConnected] = useState(false)
  const [verification, setVerification] = useState(null)
  const retry = useRef(null)
  const socketRef = useRef(null)

  useEffect(() => {
    let socket
    let active = true
    const connect = () => {
      socket = new WebSocket(`${wsRoot}/ws/observatory`)
      socketRef.current = socket
      socket.onopen = () => setConnected(true)
      socket.onmessage = (event) => setState(JSON.parse(event.data))
      socket.onerror = () => socket.close()
      socket.onclose = () => {
        setConnected(false)
        if (active) retry.current = setTimeout(connect, 900)
      }
    }
    connect()
    return () => {
      active = false
      clearTimeout(retry.current)
      socket?.close()
    }
  }, [])

  const post = useCallback(async (path, payload) => {
    const response = await fetch(`${apiRoot}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload ? JSON.stringify(payload) : undefined,
    })
    if (!response.ok) throw new Error(await response.text())
    setState(await response.json())
  }, [])

  const command = useCallback((payload) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(payload))
      return
    }
    if (payload.action === 'step') post('/api/control', { action: 'step' })
  }, [post])

  const chooseScenario = useCallback((name) => {
    const presets = { ambiguous: 'preset_1_ambiguity', drift: 'preset_2_drift', leader: 'preset_3_handover' }
    if (presets[name]) command({ action: 'load_preset', preset_id: presets[name] })
    else command({ action: 'inject_fault', fault_type: 'AMBIGUOUS_ISOLATION', target_node: 3 })
  }, [command])

  const control = useCallback((action, value) => {
    if (action === 'step') command({ action, ticks: value || 1 })
    else if (action === 'play') command({ action, speed_ms: 100 / (value || 1) })
    else command({ action, seed: 42 })
  }, [command])

  const runAllChecks = useCallback(async () => {
    command({ action: 'load_preset', preset_id: 'full_10_checks' })
    const response = await fetch(`${apiRoot}/api/run-live-checks`, { method: 'POST' })
    if (!response.ok) throw new Error(await response.text())
    setVerification(await response.json())
  }, [command])

  return {
    state,
    connected,
    chooseScenario,
    control,
    runAllChecks,
    verification,
  }
}

