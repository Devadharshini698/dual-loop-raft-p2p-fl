import { useCallback, useEffect, useRef, useState } from 'react'

const apiRoot = ''
const wsRoot = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`

export function useMissionControl() {
  const [state, setState] = useState(null)
  const [connected, setConnected] = useState(false)
  const [verification, setVerification] = useState(null)
  const [error, setError] = useState(null)
  const retry = useRef(null)
  const socketRef = useRef(null)

  useEffect(() => {
    let socket
    let active = true
    const connect = () => {
      socket = new WebSocket(`${wsRoot}/ws/observatory`)
      socketRef.current = socket
      socket.onopen = () => { setConnected(true); setError(null) }
      socket.onmessage = (event) => {
        const frame = JSON.parse(event.data)
        if (frame.error) setError(frame.error)
        else setState(frame)
      }
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

  const command = useCallback((payload, fallback) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(payload))
      return
    }
    fallback?.()
  }, [post])

  const chooseScenario = useCallback((name) => {
    const presets = { ambiguous: 'preset_1_ambiguity', drift: 'preset_2_drift', leader: 'preset_3_handover' }
    const payload = presets[name]
      ? { action: 'load_preset', preset_id: presets[name] }
      : { action: 'inject_fault', fault_type: 'AMBIGUOUS_ISOLATION', target_node: 3 }
    command(payload, () => post(`/api/scenario/${name}`))
  }, [command, post])

  const control = useCallback((action, value) => {
    let payload
    if (action === 'step') payload = { action, ticks: value || 1 }
    else if (action === 'play') payload = { action, speed_ms: 100 / (value || 1) }
    else payload = { action, seed: 42 }
    command(payload, () => post('/api/control', {
      action, speed: action === 'play' ? value || 1 : undefined,
    }))
  }, [command, post])

  const runAllChecks = useCallback(async () => {
    command({ action: 'load_preset', preset_id: 'full_10_checks' })
    try {
      setError(null)
      const response = await fetch(`${apiRoot}/api/run-live-checks`, { method: 'POST' })
      if (!response.ok) throw new Error(await response.text())
      setVerification(await response.json())
    } catch (caught) {
      setError(caught.message || 'Verification failed')
    }
  }, [command])

  return {
    state,
    connected,
    chooseScenario,
    control,
    runAllChecks,
    verification,
    error,
  }
}

