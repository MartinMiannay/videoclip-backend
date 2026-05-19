import { useState, useEffect, useRef, useMemo, useCallback } from 'react'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

function toMMSS(seconds) {
  const s = Math.max(0, seconds || 0)
  const m = Math.floor(s / 60)
  const ss = Math.floor(s % 60)
  return `${m}:${ss.toString().padStart(2, '0')}`
}

const btn = (variant = 'primary') => ({
  padding: variant === 'sm' ? '5px 12px' : '7px 16px',
  borderRadius: 6,
  border: 'none',
  fontWeight: 600,
  fontSize: variant === 'sm' ? 12 : 13,
  background:
    variant === 'danger'    ? '#6b1a1a' :
    variant === 'ghost'     ? 'transparent' :
    variant === 'secondary' ? '#2a2a2a' :
    variant === 'success'   ? '#1a5c2a' :
    '#6c8fff',
  color: variant === 'ghost' ? '#888' : '#fff',
  cursor: 'pointer',
  transition: 'opacity .15s',
})

export default function ManualClipMode({ project, onRenderStarted }) {
  const [words, setWords] = useState([])
  const [loading, setLoading] = useState(true)
  const [currentTime, setCurrentTime] = useState(0)
  const [playbackRate, setPlaybackRate] = useState(1)

  // Word-click selection
  const [selStart, setSelStart] = useState(null)   // word object
  const [selEnd, setSelEnd] = useState(null)         // word object

  // Clip form (shown once range is set)
  const [title, setTitle] = useState('')
  const [hookNote, setHookNote] = useState('')

  // Queue + render
  const [queuedClips, setQueuedClips] = useState([])
  const [rendering, setRendering] = useState(false)
  const [subtitleStyle, setSubtitleStyle] = useState('classic')
  const [cropZone, setCropZone] = useState('auto')

  const SUBTITLE_STYLES = [
    { value: 'classic',   label: 'Classic' },
    { value: 'keo',       label: 'Keo' },
    { value: 'tovaritch', label: 'Tovaritch' },
  ]

  const CROP_ZONES = [
    { value: 'auto',   label: 'Auto' },
    { value: 'left',   label: 'Gauche' },
    { value: 'center', label: 'Centre' },
    { value: 'right',  label: 'Droite' },
  ]

  const videoRef = useRef()
  const transcriptRef = useRef()

  // ── Load transcript words ─────────────────────────────────────────────────
  useEffect(() => {
    setLoading(true)
    fetch(`${API}/api/projects/${project.id}/transcript-words`)
      .then(r => r.json())
      .then(d => { setWords(d.transcript_words || []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [project.id])

  // ── Group words into paragraphs for display ───────────────────────────────
  const paragraphs = useMemo(() => {
    if (!words.length) return []
    const result = []
    let current = []
    for (let i = 0; i < words.length; i++) {
      if (i > 0 && words[i].start - words[i - 1].end > 0.5) {
        if (current.length) result.push(current)
        current = []
      }
      current.push(words[i])
    }
    if (current.length) result.push(current)
    return result
  }, [words])

  // ── Selection logic ───────────────────────────────────────────────────────
  const handleWordClick = useCallback((word) => {
    setSelStart(prev => {
      if (!prev) {
        // No start yet → this word is start (green)
        setSelEnd(null)
        return word
      }
      setSelEnd(prevEnd => {
        if (!prevEnd) {
          // Have start, no end yet
          if (word.start === prev.start) {
            // Clicking start word again → clear
            setSelStart(null)
            return null
          }
          // Set range, ensure chronological order
          if (word.start < prev.start) {
            setSelStart(word)
            return prev
          }
          return word
        }
        // Already have a full range → start fresh from this word
        setSelStart(word)
        return null
      })
      return prev
    })
  }, [])

  const clearSelection = useCallback(() => {
    setSelStart(null)
    setSelEnd(null)
    setTitle('')
    setHookNote('')
  }, [])

  // Derived selection values
  const selRange = selStart && selEnd
    ? { from: selStart.start, to: selEnd.end }
    : null
  const duration = selRange ? selRange.to - selRange.from : null

  // Word state helpers (memoised to avoid re-computing per word)
  const selStartTime = selStart?.start ?? -1
  const selEndTime   = selEnd?.end ?? -1

  // ── Word styling ──────────────────────────────────────────────────────────
  function wordStyle(word) {
    const isStart  = selStart && !selEnd && word.start === selStartTime
    const inRange  = selStart && selEnd &&
      word.start >= selStartTime && word.end <= selEndTime

    return {
      display: 'inline-block',
      cursor: 'pointer',
      borderRadius: 3,
      padding: '1px 3px',
      margin: '1px 1px',
      fontSize: 13,
      lineHeight: 1.75,
      userSelect: 'none',
      transition: 'background .08s',
      background: inRange ? 'rgba(108,143,255,0.22)' :
                  isStart  ? 'rgba(60,200,110,0.22)'  : 'transparent',
      color: inRange ? '#cce' : isStart ? '#8ef0b0' : '#aaa',
      outline: isStart ? '1px solid rgba(60,200,110,0.5)' : 'none',
    }
  }

  // ── Clip queue actions ────────────────────────────────────────────────────
  function addToQueue() {
    if (!selStart || !selEnd) { alert('Select a start and end word first'); return }
    if (!title.trim()) { alert('Title card text is required'); return }
    setQueuedClips(prev => [...prev, {
      id: Date.now(),
      start: selStart.start,
      end: selEnd.end,
      title: title.trim(),
      hook_note: hookNote.trim(),
    }])
    clearSelection()
  }

  function removeClip(id) {
    setQueuedClips(prev => prev.filter(c => c.id !== id))
  }

  async function renderAll() {
    if (!queuedClips.length) return
    setRendering(true)
    try {
      const res = await fetch(`${API}/api/projects/${project.id}/render-manual`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subtitle_style: subtitleStyle,
          crop_zone: cropZone,
          clips: queuedClips.map(c => ({
            start_seconds: c.start,
            end_seconds:   c.end,
            title:         c.title,
            hook_note:     c.hook_note,
          })),
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      onRenderStarted(await res.json())
    } catch (e) {
      alert(`Failed to start rendering: ${e.message}`)
      setRendering(false)
    }
  }

  function seekTo(seconds) {
    if (videoRef.current) {
      videoRef.current.currentTime = seconds
      videoRef.current.play()
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────
  const videoUrl = `${API}/api/projects/${project.id}/video`

  return (
    <div style={{ marginTop: 20 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div style={{ fontSize: 15, fontWeight: 600, color: '#aaa' }}>Manual Clip Selection</div>
        <div style={{ color: '#555', fontSize: 12 }}>{project.name}</div>
      </div>

      {/* Main grid: transcript | video */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, height: 480 }}>

        {/* ── Transcript panel ── */}
        <div style={{ background: '#111', border: '1px solid #222', borderRadius: 10, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ padding: '9px 14px', borderBottom: '1px solid #222', fontSize: 11, fontWeight: 600, color: '#555', textTransform: 'uppercase', letterSpacing: 1, flexShrink: 0, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Transcript — click start then end word</span>
            {selStart && (
              <button
                onClick={clearSelection}
                style={{ ...btn('ghost'), fontSize: 10, padding: '2px 8px', color: '#f88', border: '1px solid #433', borderRadius: 4 }}
              >
                Clear
              </button>
            )}
          </div>

          <div ref={transcriptRef} style={{ flex: 1, overflowY: 'auto', padding: '10px 14px' }}>
            {loading && <div style={{ color: '#555', fontSize: 13 }}>Loading transcript…</div>}
            {!loading && paragraphs.length === 0 && (
              <div style={{ color: '#555', fontSize: 13 }}>No transcript available.</div>
            )}

            {paragraphs.map((paraWords, pIdx) => (
              <div key={pIdx} style={{ marginBottom: 12 }}>
                {/* Timestamp badge — click to seek */}
                <span
                  onClick={() => seekTo(paraWords[0].start)}
                  style={{ color: '#3a3a3a', fontSize: 10, marginRight: 5, cursor: 'pointer', fontFamily: 'monospace', userSelect: 'none' }}
                  title={`Seek to ${toMMSS(paraWords[0].start)}`}
                >
                  {toMMSS(paraWords[0].start)}
                </span>

                {/* Individual clickable words */}
                {paraWords.map((word, wIdx) => (
                  <span
                    key={wIdx}
                    onClick={() => handleWordClick(word)}
                    style={wordStyle(word)}
                    title={`${toMMSS(word.start)} – ${toMMSS(word.end)}`}
                  >
                    {word.text}
                  </span>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* ── Video panel ── */}
        <div style={{ background: '#111', border: '1px solid #222', borderRadius: 10, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ flex: 1, background: '#000', minHeight: 0 }}>
            <video
              ref={videoRef}
              src={videoUrl}
              style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
              onTimeUpdate={e => setCurrentTime(e.target.currentTime)}
              controls={false}
            />
          </div>

          {/* Video controls */}
          <div style={{ padding: '10px 12px', borderTop: '1px solid #222', flexShrink: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <span style={{ color: '#888', fontSize: 11, fontFamily: 'monospace', minWidth: 38 }}>
                {toMMSS(currentTime)}
              </span>
              <input
                type="range"
                min={0}
                max={videoRef.current?.duration || 100}
                step={0.1}
                value={currentTime}
                onChange={e => { if (videoRef.current) videoRef.current.currentTime = parseFloat(e.target.value) }}
                style={{ flex: 1, accentColor: '#6c8fff', height: 4 }}
              />
              <span style={{ color: '#555', fontSize: 11, fontFamily: 'monospace', minWidth: 38, textAlign: 'right' }}>
                {videoRef.current?.duration ? toMMSS(videoRef.current.duration) : '--:--'}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <button
                style={{ ...btn('secondary'), padding: '5px 14px', fontSize: 13 }}
                onClick={() => videoRef.current && (videoRef.current.paused ? videoRef.current.play() : videoRef.current.pause())}
              >
                {videoRef.current?.paused !== false ? '▶' : '⏸'}
              </button>
              <div style={{ flex: 1 }} />
              {[0.5, 1, 1.5, 2].map(rate => (
                <button
                  key={rate}
                  style={{
                    ...btn('ghost'),
                    padding: '4px 8px',
                    fontSize: 11,
                    fontWeight: playbackRate === rate ? 700 : 400,
                    color: playbackRate === rate ? '#6c8fff' : '#555',
                    border: playbackRate === rate ? '1px solid #6c8fff' : '1px solid transparent',
                    borderRadius: 4,
                  }}
                  onClick={() => { setPlaybackRate(rate); if (videoRef.current) videoRef.current.playbackRate = rate }}
                >
                  {rate}x
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Selection status + clip form ── */}
      <div style={{ background: '#111', border: '1px solid #222', borderRadius: 10, marginTop: 16, padding: '14px 16px' }}>

        {/* Selection indicators */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: selRange ? 14 : 0 }}>
          <SelBadge label="Start" time={selStart ? selStart.start : null} color="#48c878" placeholder="click first word" />
          <span style={{ color: '#333', fontSize: 13 }}>→</span>
          <SelBadge label="End" time={selEnd ? selEnd.end : null} color="#6c8fff" placeholder="click last word" />
          {duration !== null && (
            <div style={{ marginLeft: 'auto', background: '#1a1f2e', borderRadius: 5, padding: '4px 12px', fontSize: 13, color: '#8af', fontWeight: 700, fontFamily: 'monospace', letterSpacing: 0.5 }}>
              {duration.toFixed(1)}s
            </div>
          )}
        </div>

        {/* Clip form — only visible once a range is selected */}
        {selRange && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 14 }}>
            <input
              autoFocus
              type="text"
              value={title}
              onChange={e => setTitle(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && title.trim()) addToQueue() }}
              placeholder="Title card text (required)…"
              style={{ background: '#1a1a2e', border: '1px solid #334', borderRadius: 5, color: '#fff', padding: '8px 10px', fontSize: 13, outline: 'none', width: '100%', boxSizing: 'border-box' }}
            />
            <input
              type="text"
              value={hookNote}
              onChange={e => setHookNote(e.target.value)}
              placeholder="Hook note (optional — why this clip is good)…"
              style={{ background: '#1a1a2e', border: '1px solid #2a2a3a', borderRadius: 5, color: '#ccc', padding: '8px 10px', fontSize: 13, outline: 'none', width: '100%', boxSizing: 'border-box' }}
            />
            <div style={{ display: 'flex', gap: 8 }}>
              <button style={btn()} onClick={addToQueue}>Add to Queue</button>
              <button style={{ ...btn('ghost'), color: '#666', fontSize: 12 }} onClick={clearSelection}>Cancel</button>
            </div>
          </div>
        )}

        {/* ── Clip queue ── */}
        <div style={{ fontSize: 11, fontWeight: 600, color: '#444', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8, marginTop: selRange ? 4 : 0, paddingTop: selRange ? 14 : 0, borderTop: selRange ? '1px solid #1a1a1a' : 'none' }}>
          Clip Queue ({queuedClips.length})
        </div>

        {queuedClips.length === 0 && !selRange && (
          <div style={{ color: '#3a3a3a', fontSize: 13, paddingBottom: 2 }}>
            Click a start word (green), then an end word (blue) in the transcript.
          </div>
        )}

        {queuedClips.map(clip => (
          <div
            key={clip.id}
            style={{ display: 'flex', alignItems: 'center', gap: 10, background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 6, padding: '9px 12px', marginBottom: 7, cursor: 'pointer' }}
            onClick={() => seekTo(clip.start)}
            title="Seek to clip start"
          >
            <div style={{ fontSize: 11, color: '#555', fontFamily: 'monospace', minWidth: 92, flexShrink: 0 }}>
              {toMMSS(clip.start)} → {toMMSS(clip.end)}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 13, fontWeight: 500, color: '#ddd', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {clip.title}
              </div>
              {clip.hook_note && (
                <div style={{ fontSize: 11, color: '#555', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {clip.hook_note}
                </div>
              )}
            </div>
            <div style={{ fontSize: 11, color: '#555', fontFamily: 'monospace', flexShrink: 0 }}>
              {(clip.end - clip.start).toFixed(1)}s
            </div>
            <button
              style={{ background: 'none', border: 'none', color: '#555', fontSize: 18, cursor: 'pointer', padding: '0 4px', lineHeight: 1, flexShrink: 0 }}
              onClick={e => { e.stopPropagation(); removeClip(clip.id) }}
              title="Remove"
            >
              ×
            </button>
          </div>
        ))}

        {queuedClips.length > 0 && (
          <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #1e1e1e' }}>
            {/* Style selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: '#555', textTransform: 'uppercase', letterSpacing: 1 }}>Style</span>
              {SUBTITLE_STYLES.map(st => (
                <button
                  key={st.value}
                  onClick={() => setSubtitleStyle(st.value)}
                  style={{
                    ...btn(subtitleStyle === st.value ? 'primary' : 'secondary'),
                    padding: '4px 12px',
                    fontSize: 12,
                    border: subtitleStyle === st.value ? '1px solid #6c8fff' : '1px solid #333',
                  }}
                >
                  {st.label}
                </button>
              ))}
            </div>
            {/* Cadrage selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: '#555', textTransform: 'uppercase', letterSpacing: 1 }}>Cadrage</span>
              {CROP_ZONES.map(cz => (
                <button
                  key={cz.value}
                  onClick={() => setCropZone(cz.value)}
                  style={{
                    ...btn(cropZone === cz.value ? 'primary' : 'secondary'),
                    padding: '4px 12px',
                    fontSize: 12,
                    border: cropZone === cz.value ? '1px solid #6c8fff' : '1px solid #333',
                  }}
                >
                  {cz.label}
                </button>
              ))}
            </div>
            <button
              style={{ ...btn('success'), padding: '10px 24px', fontSize: 14, opacity: rendering ? 0.6 : 1 }}
              onClick={renderAll}
              disabled={rendering}
            >
              {rendering ? 'Starting render…' : `Render All ${queuedClips.length} Clip${queuedClips.length !== 1 ? 's' : ''}`}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Small helper component ────────────────────────────────────────────────────

function SelBadge({ label, time, color, placeholder }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
      <div style={{
        width: 8, height: 8, borderRadius: 2, flexShrink: 0,
        background: time !== null ? color : '#2a2a2a',
        boxShadow: time !== null ? `0 0 6px ${color}88` : 'none',
        transition: 'all .2s',
      }} />
      <span style={{
        fontSize: 12,
        fontFamily: 'monospace',
        color: time !== null ? color : '#333',
        transition: 'color .2s',
      }}>
        {time !== null ? toMMSS(time) : placeholder}
      </span>
    </div>
  )
}
