import { useEffect, useRef } from 'react'
import cytoscape from 'cytoscape'

const positions = {
  'node-1': { x: 300, y: 70 },
  'node-2': { x: 90, y: 235 },
  'node-3': { x: 230, y: 300 },
  'node-4': { x: 380, y: 300 },
  'node-5': { x: 510, y: 235 },
}

const stylesheet = [
  { selector: 'node', style: {
    width: 74, height: 74, 'background-color': '#12312c', 'border-width': 2,
    'border-color': '#38d99b', label: 'data(label)', color: '#dcece7',
    'font-size': 10, 'font-weight': 700, 'text-valign': 'center',
    'text-margin-y': -2, 'text-wrap': 'wrap', 'text-max-width': 58,
    'line-height': 1.35, 'overlay-opacity': 0,
  } },
  { selector: 'node[status="leader"]', style: {
    'background-color': '#3b2e12', 'border-color': '#f1c75b', 'border-width': 4,
  } },
  { selector: 'node[status="suspect"]', style: {
    'background-color': '#3b2612', 'border-color': '#f49b45',
  } },
  { selector: 'node[status="crashed"]', style: {
    'background-color': '#41171f', 'border-color': '#ff5a6d', opacity: .72,
  } },
  { selector: 'edge', style: {
    width: 2, 'line-color': '#2e6157', 'target-arrow-color': '#2e6157',
    'target-arrow-shape': 'triangle', 'curve-style': 'bezier', opacity: .78,
  } },
  { selector: 'edge[kind="witness"]', style: {
    'line-color': '#a981ff', 'target-arrow-color': '#a981ff', 'line-style': 'dashed', width: 3,
  } },
  { selector: 'edge[kind="drop"]', style: {
    'line-color': '#f49b45', 'target-arrow-color': '#f49b45', 'line-style': 'dashed', opacity: .48,
  } },
  { selector: 'edge[kind="fl"]', style: {
    'line-color': '#46d7e8', 'target-arrow-color': '#46d7e8', width: 5,
  } },
]

export default function Topology({ nodes, links }) {
  const host = useRef(null)
  const graph = useRef(null)

  useEffect(() => {
    graph.current = cytoscape({ container: host.current, elements: [], style: stylesheet,
      layout: { name: 'preset' }, userZoomingEnabled: false, userPanningEnabled: false })
    return () => graph.current?.destroy()
  }, [])

  useEffect(() => {
    if (!graph.current) return
    const elements = [
      ...nodes.map((node) => ({ data: { ...node, label: `${node.label}\n${node.role.toUpperCase()}\nBOOT ${node.boot_id}` }, position: positions[node.id] })),
      ...links.map((link, index) => ({ data: { id: `e-${index}`, ...link } })),
    ]
    graph.current.elements().remove()
    graph.current.add(elements)
    graph.current.layout({ name: 'preset', fit: true, padding: 34 }).run()
  }, [nodes, links])

  return <div ref={host} className="h-[340px] w-full" aria-label="Live five-node Raft topology" />
}

