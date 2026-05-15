import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import ManualClipMode from './components/ManualClipMode'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
// Direct TCP address used for chunk uploads to bypass proxy timeout on large files
const UPLOAD_API = import.meta.env.VITE_UPLOAD_API_URL || 'http://127.0.0.1:8001'
const CHUNK_SIZE = 5 * 1024 * 1024 // 5 MB
const CHUNK_RETRIES = 3

const s = {
  app: { maxWidth: 860, margin: '0 auto', padding: '32px 16px' },
  h1: { fontSize: 26, fontWeight: 700, marginBottom: 28, letterSpacing: -0.5 },

  // Upload zone
  dropzone: (drag) => ({
    border: `2px dashed ${drag ? '#6c8fff' : '#333'}`,
    borderRadius: 12,
    padding: '48px 24px',
    textAlign: 'center',
    background: drag ? '#1a1f2e' : '#161616',
    transition: 'all .2s',
    cursor: 'pointer',
  }),
  dropText: { color: '#888', fontSize: 14, marginTop: 8 },

  // Upload progress
  uploadBox: { background: '#161616', borderRadius: 12, padding: 20, marginTop: 20 },
  progressTrack: { height: 6, background: '#222', borderRadius: 3, overflow: 'hidden', margin: '10px 0' },
  progressFill: (pct, color = '#6c8fff') => ({
    height: '100%', width: `${pct}%`, background: color, borderRadius: 3, transition: 'width .3s',
  }),

  // Project list
  section: { marginTop: 40 },
  sectionTitle: { fontSize: 16, fontWeight: 600, color: '#aaa', marginBottom: 14, textTransform: 'uppercase', letterSpacing: 1, fontSize: 12 },
  card: (selected) => ({
    background: selected ? '#1a1f2e' : '#161616',
    border: `1px solid ${selected ? '#6c8fff' : '#222'}`,
    borderRadius: 10,
    padding: '14px 16px',
    marginBottom: 10,
    cursor: 'pointer',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    transition: 'all .15s',
  }),
  cardName: { fontWeight: 500, fontSize: 15, marginBottom: 2 },
  cardMeta: { color: '#666', fontSize: 12 },

  // Status badge
  badge: (status) => {
    const colors = {
      uploaded: '#555', processing: '#b87a00', done: '#1a6b3a',
      error: '#6b1a1a', partial: '#3a3a00', transcribed: '#0a3a5a',
    }
    const text = {
      uploaded: '#aaa', processing: '#f0b429', done: '#4caf50',
      error: '#f44336', partial: '#e0e000', transcribed: '#4ab0ff',
    }
    return {
      background: colors[status] || '#333',
      color: text[status] || '#aaa',
      borderRadius: 4,
      padding: '2px 8px',
      fontSize: 11,
      fontWeight: 600,
      textTransform: 'uppercase',
      letterSpacing: 0.5,
    }
  },

  // Detail panel
  panel: { background: '#161616', border: '1px solid #222', borderRadius: 12, padding: 24, marginTop: 20 },
  panelTitle: { fontSize: 18, fontWeight: 600, marginBottom: 6 },
  processingStep: { color: '#f0b429', fontSize: 13, marginBottom: 10 },
  processingDetails: { color: '#888', fontSize: 12, marginBottom: 14 },

  // Clip grid
  clipGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 14, marginTop: 16 },
  clipCard: { background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 8, padding: 16 },
  clipCaption: { fontSize: 13, marginBottom: 10, lineHeight: 1.4 },
  clipMeta: { color: '#666', fontSize: 11, marginBottom: 12 },

  // Buttons
  btn: (variant = 'primary') => ({
    padding: variant === 'sm' ? '6px 14px' : '10px 20px',
    borderRadius: 6,
    border: 'none',
    fontWeight: 600,
    fontSize: variant === 'sm' ? 12 : 14,
    background: variant === 'danger' ? '#6b1a1a'
      : variant === 'ghost' ? 'transparent'
      : variant === 'secondary' ? '#2a2a2a'
      : variant === 'success' ? '#1a5c2a'
      : '#6c8fff',
    color: variant === 'ghost' ? '#888' : '#fff',
    transition: 'opacity .15s',
    cursor: 'pointer',
  }),
  btnRow: { display: 'flex', gap: 10, alignItems: 'center', marginTop: 16 },

  deleteBtn: {
    background: 'none', border: 'none', color: '#555', fontSize: 18, padding: '0 4px', lineHeight: 1,
    cursor: 'pointer',
  },
  empty: { color: '#555', fontSize: 14, padding: '24px 0', textAlign: 'center' },
}

function formatDate(iso) {
  return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

function formatDuration(sec) {
  if (!sec) return ''
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function toMMSS(seconds) {
  const s = Math.max(0, seconds || 0)
  const m = Math.floor(s / 60)
  const ss = Math.floor(s % 60)
  return `${m}:${ss.toString().padStart(2, '0')}`
}

function parseMMSS(str) {
  if (!str) return 0
  const parts = String(str).trim().split(':')
  if (parts.length === 2) return parseInt(parts[0] || 0) * 60 + parseFloat(parts[1] || 0)
  return parseFloat(str) || 0
}

// ── Upload component ────────────────────────────────────────────────────────

function UploadZone({ onUploaded }) {
  const [drag, setDrag] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)  // 0–100
  const [statusText, setStatusText] = useState('')
  const [error, setError] = useState('')
  const inputRef = useRef()

  async function handleFile(file) {
    if (!file) return
    setError('')
    setUploading(true)
    setProgress(0)
    setStatusText(`Preparing upload…`)

    try {
      const totalChunks = Math.ceil(file.size / CHUNK_SIZE)

      // 1. Init
      const initRes = await fetch(`${API}/api/upload/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: file.name, total_chunks: totalChunks, content_type: file.type || 'video/mp4', file_size: file.size }),
      })
      if (!initRes.ok) throw new Error('Upload init failed')
      const { upload_id } = await initRes.json()

      // 2. Chunks — resumable: track done chunks so a dropped connection
      //    can resume from where it left off rather than restarting from 0.
      const doneChunks = new Set()
      let resumeAttempt = 0
      while (doneChunks.size < totalChunks) {
        if (resumeAttempt > 0) {
          setStatusText(`Connection dropped — resuming (${doneChunks.size}/${totalChunks} chunks already done)…`)
          await new Promise(r => setTimeout(r, 5000))
        }
        let connectionFailed = false
        for (let i = 0; i < totalChunks; i++) {
          if (doneChunks.has(i)) continue
          const start = i * CHUNK_SIZE
          const blob = file.slice(start, start + CHUNK_SIZE)
          let success = false
          let lastErr
          for (let attempt = 0; attempt < CHUNK_RETRIES; attempt++) {
            if (attempt > 0) {
              const isTimeout = lastErr && (lastErr.includes('abort') || lastErr.includes('Abort') || lastErr.includes('timeout'))
              const delay = isTimeout ? 5000 : 1000 * attempt
              setStatusText(`Chunk ${i + 1}/${totalChunks} — retry ${attempt}/${CHUNK_RETRIES - 1}…`)
              await new Promise(r => setTimeout(r, delay))
            }
            const fd = new FormData()
            fd.append('file', blob, file.name)
            const controller = new AbortController()
            const timeout = setTimeout(() => controller.abort(), 300_000)
            try {
              const r = await fetch(`${UPLOAD_API}/api/upload/chunk/${upload_id}/${i}`, { method: 'POST', body: fd, signal: controller.signal })
              if (r.ok) { success = true; break }
              lastErr = `Chunk ${i} HTTP ${r.status}`
            } catch (e) {
              lastErr = e.message
            } finally {
              clearTimeout(timeout)
            }
          }
          if (!success) {
            // Treat as a dropped connection — break out and retry the whole pass
            connectionFailed = true
            break
          }
          doneChunks.add(i)
          setProgress(Math.round((doneChunks.size / totalChunks) * 90))
          setStatusText(`Uploading… ${doneChunks.size}/${totalChunks} chunks`)
        }
        if (connectionFailed) {
          resumeAttempt++
          if (resumeAttempt > CHUNK_RETRIES) throw new Error(`Upload failed after ${CHUNK_RETRIES} resume attempts (${doneChunks.size}/${totalChunks} chunks completed)`)
        }
      }

      // 3. Complete
      setStatusText('Finalizing…')
      const completeRes = await fetch(`${API}/api/upload/complete/${upload_id}`, { method: 'POST' })
      if (!completeRes.ok) throw new Error('Upload finalization failed')
      const project = await completeRes.json()
      setProgress(100)
      setStatusText('Upload complete!')
      setTimeout(() => { setUploading(false); setProgress(0); setStatusText('') }, 800)
      onUploaded(project)
    } catch (e) {
      setError(e.message)
      setUploading(false)
    }
  }

  function onDrop(e) {
    e.preventDefault(); setDrag(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div>
      <div
        style={s.dropzone(drag)}
        onClick={() => !uploading && inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
      >
        <div style={{ fontSize: 36 }}>🎬</div>
        <div style={{ fontWeight: 600, marginTop: 10 }}>
          {uploading ? statusText : 'Drop a video or click to browse'}
        </div>
        <div style={s.dropText}>MP4, MOV, MKV, AVI · any size</div>
        <input ref={inputRef} type="file" accept="video/*" style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])} />
      </div>
      {uploading && (
        <div style={s.uploadBox}>
          <div style={{ fontSize: 13, marginBottom: 6, color: '#aaa' }}>{statusText}</div>
          <div style={s.progressTrack}>
            <div style={s.progressFill(progress)} />
          </div>
          <div style={{ fontSize: 12, color: '#555', textAlign: 'right' }}>{progress}%</div>
        </div>
      )}
      {error && <div style={{ color: '#f44336', fontSize: 13, marginTop: 10 }}>Error: {error}</div>}
    </div>
  )
}

// ── Project detail panel ────────────────────────────────────────────────────

const SUBTITLE_STYLES = [
  { value: 'classic', label: 'Classic', desc: 'Montserrat Bold, black outline, bottom' },
  { value: 'keo',     label: 'Keo',     desc: 'Arial mixed-weight, drop shadow, no box' },
  { value: 'tovaritch', label: 'Tovaritch', desc: 'Impact ALL CAPS, red & white, no music' },
]

function ProjectPanel({ project: initial, onDeleted, onUpdated }) {
  const [project, setProject] = useState(initial)
  const [subtitleStyle, setSubtitleStyle] = useState('classic')
  const intervalRef = useRef()

  const poll = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/projects/${initial.id}`)
      if (r.ok) {
        const updated = await r.json()
        setProject(updated)
        if (onUpdated) onUpdated(updated)
      }
    } catch {}
  }, [initial.id])

  useEffect(() => {
    setProject(initial)
  }, [initial.id])

  useEffect(() => {
    if (project.status === 'processing') {
      intervalRef.current = setInterval(poll, 2500)
    } else {
      clearInterval(intervalRef.current)
    }
    return () => clearInterval(intervalRef.current)
  }, [project.status, poll])

  async function startAutoProcessing() {
    await fetch(`${API}/api/projects/${project.id}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subtitle_style: subtitleStyle }),
    })
    setProject(p => ({ ...p, status: 'processing', processing_step: 'starting', processing_progress: 0 }))
  }

  async function startManualTranscription() {
    await fetch(`${API}/api/projects/${project.id}/transcribe`, { method: 'POST' })
    setProject(p => ({ ...p, status: 'processing', processing_step: 'transcribing', processing_progress: 0 }))
  }

  async function retryFailed() {
    await fetch(`${API}/api/projects/${project.id}/retry-failed`, { method: 'POST' })
    setProject(p => ({ ...p, status: 'processing' }))
  }

  async function deleteProject() {
    if (!confirm(`Delete "${project.name}"?`)) return
    await fetch(`${API}/api/projects/${project.id}`, { method: 'DELETE' })
    onDeleted(project.id)
  }

  function handleRenderStarted(updated) {
    setProject(updated)
  }

  const clips = project.short_clips || []
  const doneClips = clips.filter(c => c.status === 'done')
  const errorClips = clips.filter(c => c.status === 'error')
  const pct = Math.round((project.processing_progress || 0) * 100)

  return (
    <div style={s.panel}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={s.panelTitle}>{project.name}</div>
          <div style={{ ...s.cardMeta, marginTop: 4 }}>
            {project.original_filename}
            {project.duration ? ` · ${formatDuration(project.duration)}` : ''}
            {' · '}{formatDate(project.created_at)}
          </div>
        </div>
        <span style={s.badge(project.status)}>{project.status}</span>
      </div>

      {project.status === 'processing' && (
        <div style={{ marginTop: 16 }}>
          <div style={s.processingStep}>{project.processing_step}</div>
          {project.processing_details && (
            <div style={s.processingDetails}>{project.processing_details}</div>
          )}
          <div style={s.progressTrack}>
            <div style={s.progressFill(pct, '#f0b429')} />
          </div>
          <div style={{ fontSize: 12, color: '#555', textAlign: 'right', marginTop: 4 }}>{pct}%</div>
        </div>
      )}

      {/* Mode selection — shown when video is uploaded but not yet started */}
      {project.status === 'uploaded' && (
        <div style={{ marginTop: 20 }}>
          {/* Style selector */}
          <div style={{ marginBottom: 18 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: '#555', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
              Subtitle Style
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              {SUBTITLE_STYLES.map(st => (
                <button
                  key={st.value}
                  onClick={() => setSubtitleStyle(st.value)}
                  style={{
                    flex: 1,
                    padding: '10px 12px',
                    borderRadius: 7,
                    border: `1px solid ${subtitleStyle === st.value ? '#6c8fff' : '#2a2a2a'}`,
                    background: subtitleStyle === st.value ? '#1a1f2e' : '#161616',
                    color: subtitleStyle === st.value ? '#c5d0ff' : '#777',
                    cursor: 'pointer',
                    textAlign: 'left',
                    transition: 'all .15s',
                  }}
                >
                  <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 2 }}>{st.label}</div>
                  <div style={{ fontSize: 11, opacity: 0.7 }}>{st.desc}</div>
                </button>
              ))}
            </div>
          </div>

          <div style={{ fontSize: 13, color: '#888', marginBottom: 14 }}>
            How do you want to select clips?
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <button
              style={{ ...s.btn(), padding: '16px 20px', fontSize: 14, borderRadius: 8, textAlign: 'left', lineHeight: 1.4 }}
              onClick={startAutoProcessing}
            >
              <div style={{ fontWeight: 700, marginBottom: 4 }}>Auto Select Clips</div>
              <div style={{ fontSize: 12, opacity: 0.7, fontWeight: 400 }}>Claude picks the best moments from your transcript</div>
            </button>
            <button
              style={{ ...s.btn('secondary'), padding: '16px 20px', fontSize: 14, borderRadius: 8, textAlign: 'left', lineHeight: 1.4 }}
              onClick={startManualTranscription}
            >
              <div style={{ fontWeight: 700, marginBottom: 4 }}>Manual Selection</div>
              <div style={{ fontSize: 12, opacity: 0.7, fontWeight: 400 }}>Transcribe, then choose exact clip boundaries yourself</div>
            </button>
          </div>
        </div>
      )}

      {project.status === 'error' && (
        <div style={{ marginTop: 12 }}>
          <div style={{ color: '#f44336', fontSize: 13, marginBottom: 10 }}>
            {project.processing_details || 'Processing failed.'}
          </div>
          <button style={s.btn()} onClick={startAutoProcessing}>Retry</button>
        </div>
      )}

      {/* Manual clip editor — shown after transcription-only step completes */}
      {project.status === 'transcribed' && (
        <div>
          <div style={{ marginTop: 12, marginBottom: 4, display: 'flex', gap: 10 }}>
            <button
              style={{ ...s.btn('ghost'), padding: '5px 12px', fontSize: 12 }}
              onClick={startAutoProcessing}
            >
              Switch to Auto Mode
            </button>
          </div>
          <ManualClipMode project={project} onRenderStarted={handleRenderStarted} />
        </div>
      )}

      {clips.length > 0 && project.status !== 'transcribed' && (
        <div style={{ marginTop: 24 }}>
          <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 14 }}>
            Clips ({doneClips.length} ready{errorClips.length > 0 ? `, ${errorClips.length} failed` : ''})
          </div>
          <div style={s.clipGrid}>
            {clips.map(clip => (
              <ClipCard key={clip.id} clip={clip} projectId={project.id} />
            ))}
          </div>
          {errorClips.length > 0 && project.status !== 'processing' && (
            <div style={s.btnRow}>
              <button style={s.btn()} onClick={retryFailed}>Retry Failed Clips</button>
            </div>
          )}
        </div>
      )}

      <div style={{ ...s.btnRow, marginTop: 24, borderTop: '1px solid #222', paddingTop: 16 }}>
        <button style={s.btn('danger')} onClick={deleteProject}>Delete Project</button>
      </div>
    </div>
  )
}

// ── Clip card ───────────────────────────────────────────────────────────────

function ClipCard({ clip, projectId }) {
  const ready = clip.status === 'done' && clip.storage_path
  const failed = clip.status === 'error'

  return (
    <div style={{
      ...s.clipCard,
      opacity: failed ? 0.5 : 1,
      border: failed ? '1px solid #3a1a1a' : '1px solid #2a2a2a',
    }}>
      <div style={s.clipCaption}>{clip.caption || 'Untitled clip'}</div>
      <div style={s.clipMeta}>
        {clip.start != null && clip.end != null
          ? `${formatDuration(clip.start)} – ${formatDuration(clip.end)}`
          : ''}
        {clip.score != null ? ` · score ${clip.score}` : ''}
      </div>
      {ready && (
        <a
          href={`${API}/api/clips/${projectId}/${clip.id}/download`}
          download
          style={{ ...s.btn('sm'), display: 'inline-block', textDecoration: 'none', textAlign: 'center' }}
        >
          Download
        </a>
      )}
      {failed && <div style={{ color: '#f44336', fontSize: 11 }}>{clip.error || 'Failed'}</div>}
      {!ready && !failed && (
        <div style={{ color: '#666', fontSize: 11 }}>
          {clip.status === 'done' ? 'Ready' : clip.status || 'Pending'}
        </div>
      )}
    </div>
  )
}

// ── Main App ────────────────────────────────────────────────────────────────

export default function App() {
  const [projects, setProjects] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API}/api/projects`)
      .then(r => r.json())
      .then(data => { setProjects(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  function onUploaded(project) {
    setProjects(prev => [project, ...prev])
    setSelectedId(project.id)
  }

  function onDeleted(id) {
    setProjects(prev => prev.filter(p => p.id !== id))
    if (selectedId === id) setSelectedId(null)
  }

  function onUpdated(updated) {
    setProjects(prev => prev.map(p => p.id === updated.id ? { ...p, status: updated.status } : p))
  }

  const selected = projects.find(p => p.id === selectedId)

  return (
    <div style={s.app}>
      <h1 style={s.h1}>VideoClip</h1>

      <UploadZone onUploaded={onUploaded} />

      <div style={s.section}>
        <div style={s.sectionTitle}>Projects</div>
        {loading && <div style={s.empty}>Loading…</div>}
        {!loading && projects.length === 0 && (
          <div style={s.empty}>No projects yet. Upload a video to get started.</div>
        )}
        {projects.map(p => (
          <div key={p.id} style={s.card(p.id === selectedId)} onClick={() => setSelectedId(p.id === selectedId ? null : p.id)}>
            <div>
              <div style={s.cardName}>{p.name}</div>
              <div style={s.cardMeta}>
                {formatDate(p.created_at)}
                {p.duration ? ` · ${formatDuration(p.duration)}` : ''}
                {(p.short_clips?.length > 0) ? ` · ${p.short_clips.length} clips` : ''}
              </div>
            </div>
            <span style={s.badge(p.status)}>{p.status}</span>
          </div>
        ))}
      </div>

      {selected && (
        <ProjectPanel
          key={selected.id}
          project={selected}
          onDeleted={onDeleted}
          onUpdated={onUpdated}
        />
      )}
    </div>
  )
}
