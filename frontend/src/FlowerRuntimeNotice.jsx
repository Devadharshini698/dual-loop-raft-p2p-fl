import { TriangleAlert } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

export default function FlowerRuntimeNotice({ flower }) {
  const fallback = flower?.fallback === true
  const [open, setOpen] = useState(fallback)
  const acknowledgeRef = useRef(null)

  useEffect(() => {
    if (fallback) setOpen(true)
  }, [fallback])

  useEffect(() => {
    if (!open) return undefined
    const previous = document.activeElement
    acknowledgeRef.current?.focus()
    const handleKey = (event) => {
      if (event.key === 'Escape') setOpen(false)
      if (event.key === 'Tab') {
        event.preventDefault()
        acknowledgeRef.current?.focus()
      }
    }
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('keydown', handleKey)
      previous?.focus?.()
    }
  }, [open])

  if (!fallback || !open) return null

  return <div className="runtime-modal" role="alertdialog" aria-modal="true"
    aria-labelledby="flower-runtime-title" aria-describedby="flower-runtime-copy">
    <section className="runtime-dialog">
      <span className="runtime-warning-icon"><TriangleAlert size={25}/></span>
      <p className="section-label">RUNTIME DISCLOSURE</p>
      <h2 id="flower-runtime-title">Flower fallback is active</h2>
      <p id="flower-runtime-copy">{flower.message}</p>
      <p className="runtime-detail">This is Flower-ready integration with a zero-dependency fallback. Install the showcase requirements and restart the Observatory to activate the real Flower SDK path.</p>
      <button ref={acknowledgeRef} onClick={() => setOpen(false)}>Acknowledge and continue</button>
    </section>
  </div>
}
