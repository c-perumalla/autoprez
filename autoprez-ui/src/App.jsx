import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Mic, Play, Square, Settings2, Music, Clock } from 'lucide-react'
import './App.css'

export default function App() {
  const [songs, setSongs] = useState([])
  const [mics, setMics] = useState([])
  
  const [selectedSong, setSelectedSong] = useState('')
  const [selectedMic, setSelectedMic] = useState('')
  
  const [params, setParams] = useState({
    trigger_words: 5,
    min_match_pct: 60,
    min_time_on_line: 4.0,
    word_tolerance: 2,
    update_interval: 0.5
  })
  
  const [isRunning, setIsRunning] = useState(false)
  const [elapsedTime, setElapsedTime] = useState(0)
  
  // Lyric state
  const [lyrics, setLyrics] = useState([])
  const [currentIndex, setCurrentIndex] = useState(-1) // -1 means starting
  const [lastLatency, setLastLatency] = useState('')
  const [liveTranscript, setLiveTranscript] = useState('')
  
  const wsRef = useRef(null)
  const timerRef = useRef(null)

  useEffect(() => {
    // Fetch songs and mics
    fetch('http://localhost:9000/songs')
      .then(res => res.json())
      .then(data => {
        setSongs(data.songs)
        if (data.songs.length > 0) {
          setSelectedSong(data.songs[0].filename)
        }
      })
      .catch(err => console.error("Error fetching songs:", err))

    fetch('http://localhost:9000/mics')
      .then(res => res.json())
      .then(data => {
        setMics(data.mics)
        if (data.mics.length > 0) {
          // Try to select a default mic, or just leave it empty for system default
          setSelectedMic('') 
        }
      })
      .catch(err => console.error("Error fetching mics:", err))

    // Setup WebSocket
    connectWebSocket()

    return () => {
      if (wsRef.current) wsRef.current.close()
      clearInterval(timerRef.current)
    }
  }, [])
  
  // When song changes, update local lyrics array
  useEffect(() => {
    const song = songs.find(s => s.filename === selectedSong)
    if (song) {
      setLyrics(song.lyrics)
      setCurrentIndex(-1)
    }
  }, [selectedSong, songs])

  const connectWebSocket = () => {
    wsRef.current = new WebSocket('ws://localhost:9000/ws')
    
    wsRef.current.onopen = () => console.log('WS Connected')
    
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      console.log("WS Message:", data)
      
      if (data.action === 'transition_slide') {
        if (data.line_index !== undefined) {
          setCurrentIndex(data.line_index)
        } else {
          // Fallback if line_index is missing
          setCurrentIndex(prev => prev + 1)
        }
        
        if (data.latency_metric) {
          setLastLatency(data.latency_metric)
        }
        // Clear transcript on transition
        setLiveTranscript('')
      } else if (data.action === 'transcript_update') {
        setLiveTranscript(data.text)
      }
    }
    
    wsRef.current.onclose = () => {
      console.log('WS Disconnected, reconnecting...')
      setTimeout(connectWebSocket, 1000)
    }
  }

  const handleStart = async () => {
    try {
      const body = {
        song_file: selectedSong,
        params: params,
        mic_device: selectedMic ? parseInt(selectedMic) : null
      }
      
      await fetch('http://localhost:9000/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      
      setIsRunning(true)
      setCurrentIndex(-1) // Start before line 0
      setLastLatency('')
      setLiveTranscript('')
      setElapsedTime(0)
      
      timerRef.current = setInterval(() => {
        setElapsedTime(prev => prev + 1)
      }, 1000)
      
    } catch (err) {
      console.error("Failed to start:", err)
    }
  }

  const handleStop = async () => {
    try {
      await fetch('http://localhost:9000/stop', { method: 'POST' })
      setIsRunning(false)
      clearInterval(timerRef.current)
    } catch (err) {
      console.error("Failed to stop:", err)
    }
  }

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0')
    const s = (seconds % 60).toString().padStart(2, '0')
    return `${m}:${s}`
  }

  const handleParamChange = (name, value) => {
    setParams(prev => ({ ...prev, [name]: parseFloat(value) }))
  }

  // Get rolling window lines
  const prevLine = currentIndex > 0 ? lyrics[currentIndex - 1] : ''
  const currentLine = currentIndex >= 0 && currentIndex < lyrics.length ? lyrics[currentIndex] : (currentIndex === -1 ? 'Ready to start...' : 'End of song')
  const nextLine = currentIndex + 1 < lyrics.length ? lyrics[currentIndex + 1] : ''

  return (
    <div className="app-container">
      {/* Sidebar Controls */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h1>Autoprez Tester</h1>
          <p>Real-time lyric matcher evaluation</p>
        </div>

        <div className="control-group">
          <label><Music size={14} style={{display:'inline', verticalAlign:'bottom', marginRight:4}}/> Song</label>
          <select 
            value={selectedSong} 
            onChange={e => setSelectedSong(e.target.value)}
            disabled={isRunning}
          >
            {songs.map(song => (
              <option key={song.id} value={song.filename}>{song.title}</option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label><Mic size={14} style={{display:'inline', verticalAlign:'bottom', marginRight:4}}/> Microphone</label>
          <select 
            value={selectedMic} 
            onChange={e => setSelectedMic(e.target.value)}
            disabled={isRunning}
          >
            <option value="">System Default</option>
            {mics.map(mic => (
              <option key={mic.id} value={mic.id}>{mic.name}</option>
            ))}
          </select>
        </div>

        <div style={{height: 1, backgroundColor: 'var(--panel-border)', margin: '16px 0'}}></div>
        
        <div style={{display:'flex', alignItems:'center', gap: 6, marginBottom: 16}}>
          <Settings2 size={16} color="var(--text-secondary)"/>
          <span style={{fontSize: 14, fontWeight: 600, color: 'var(--text-secondary)'}}>Parameters</span>
        </div>

        <div className="control-group">
          <div className="range-header">
            <label>Trigger Words</label>
            <span className="range-value">{params.trigger_words}</span>
          </div>
          <input type="range" min="0" max="10" step="1" 
            value={params.trigger_words} onChange={e => handleParamChange('trigger_words', e.target.value)} disabled={isRunning}/>
        </div>

        <div className="control-group">
          <div className="range-header">
            <label>Min Match %</label>
            <span className="range-value">{params.min_match_pct}%</span>
          </div>
          <input type="range" min="0" max="100" step="5" 
            value={params.min_match_pct} onChange={e => handleParamChange('min_match_pct', e.target.value)} disabled={isRunning}/>
        </div>

        <div className="control-group">
          <div className="range-header">
            <label>Min Time On Line (s)</label>
            <span className="range-value">{params.min_time_on_line}s</span>
          </div>
          <input type="range" min="0" max="10" step="0.5" 
            value={params.min_time_on_line} onChange={e => handleParamChange('min_time_on_line', e.target.value)} disabled={isRunning}/>
        </div>

        <div className="control-group">
          <div className="range-header">
            <label>Word Tolerance</label>
            <span className="range-value">{params.word_tolerance}</span>
          </div>
          <input type="range" min="0" max="5" step="1" 
            value={params.word_tolerance} onChange={e => handleParamChange('word_tolerance', e.target.value)} disabled={isRunning}/>
        </div>

        <div className="button-group">
          {!isRunning ? (
            <button className="btn-start" onClick={handleStart}>
              <Play size={18} fill="currentColor" /> Start Matching
            </button>
          ) : (
            <button className="btn-stop" onClick={handleStop}>
              <Square size={18} fill="currentColor" /> Stop
            </button>
          )}
        </div>

        <div className="status-bar">
          <div className={`status-indicator ${isRunning ? 'active' : ''}`}></div>
          <span>{isRunning ? 'Listening...' : 'Idle'}</span>
          {isRunning && (
            <span style={{marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'monospace'}}>
              <Clock size={14} /> {formatTime(elapsedTime)}
            </span>
          )}
        </div>
      </div>

      {/* Main Lyric Display Area */}
      <div className="main-display">
        {selectedSong ? (
          <div className="lyric-window">
            <AnimatePresence mode="popLayout">
              {/* Previous Line */}
              {prevLine && (
                <motion.div 
                  key={`prev-${currentIndex}`}
                  className="lyric-line lyric-prev"
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 0.7, y: 0 }}
                  exit={{ opacity: 0, y: -40 }}
                  transition={{ duration: 0.4 }}
                >
                  {prevLine}
                </motion.div>
              )}

              {/* Current Line */}
              <motion.div 
                key={`curr-${currentIndex}`}
                className="lyric-line lyric-current glow"
                initial={{ opacity: 0, y: 40, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -40, scale: 1.05 }}
                transition={{ duration: 0.4, type: 'spring', stiffness: 100 }}
              >
                {currentLine}
              </motion.div>

              {/* Next Line */}
              {nextLine && (
                <motion.div 
                  key={`next-${currentIndex}`}
                  className="lyric-line lyric-next"
                  initial={{ opacity: 0, y: 40 }}
                  animate={{ opacity: 0.7, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.4 }}
                >
                  {nextLine}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ) : (
          <div className="lyric-window">
            <span className="placeholder-text">Select a song to begin</span>
          </div>
        )}

        {/* Debug Panel at bottom */}
        {isRunning && (
          <div className="debug-panel">
            <div className="debug-transcript">
              <span className="transcript-label">Live Hearing:</span> {liveTranscript || <span style={{opacity: 0.5}}>Waiting for audio...</span>}
            </div>
            <div className="debug-stats">
              <div className="debug-text">
                Index: {currentIndex} / {lyrics.length} | Params: {params.trigger_words} words, {params.min_match_pct}% match
              </div>
              {lastLatency && (
                <div className="debug-latency">
                  Latency: {lastLatency}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
