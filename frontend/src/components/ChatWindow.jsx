import { useState, useRef, useEffect } from 'react'
import MessageBubble from './MessageBubble'
import { Send, Loader2 } from 'lucide-react'

const SUGGESTIONS = [
  'What molecules have been detected in exoplanet atmospheres?',
  'How do gravitational waves reveal black hole mergers?',
  'What is the role of dark matter in galaxy formation?',
  'How does JWST observe the early universe?',
]

export default function ChatWindow() {
  const [messages, setMessages] = useState([])
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const bottomRef               = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send(question) {
    const q = question || input.trim()
    if (!q || loading) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: q }])
    setLoading(true)
    try {
      const res  = await fetch('/api/query', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ question: q, top_k: 5, use_rerank: true }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, {
        role:    'assistant',
        content: data.answer,
        sources: data.sources,
        model:   data.model,
      }])
    } catch {
      setMessages(prev => [...prev, {
        role:    'assistant',
        content: 'Something went wrong. Make sure the backend is running.',
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-6 px-4">
            <h1 className="text-3xl font-semibold text-white">Space RAG</h1>
            <p className="text-gray-400 text-sm">Ask anything about space research papers</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-2xl">
              {SUGGESTIONS.map(s => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="text-left text-sm text-gray-300 bg-input border border-border
                             rounded-xl px-4 py-3 hover:border-accent transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto w-full">
            {messages.map((m, i) => <MessageBubble key={i} message={m} />)}
            {loading && (
              <div className="flex gap-4 px-4 py-6 bg-surface">
                <div className="w-8 h-8 rounded-full bg-input flex items-center justify-center">
                  <Loader2 size={16} className="text-accent animate-spin" />
                </div>
                <div className="flex items-center text-gray-400 text-sm">Thinking...</div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-4 border-t border-border">
        <div className="max-w-3xl mx-auto flex gap-3 items-end bg-input border border-border
                        rounded-2xl px-4 py-3 focus-within:border-accent transition-colors">
          <textarea
            rows={1}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
            placeholder="Ask about space research..."
            className="flex-1 bg-transparent text-sm text-gray-100 placeholder-gray-500
                       resize-none outline-none max-h-40"
          />
          <button
            onClick={() => send()}
            disabled={!input.trim() || loading}
            className="flex-shrink-0 p-1.5 rounded-lg bg-accent text-white
                       disabled:opacity-30 hover:opacity-90 transition-opacity"
          >
            <Send size={16} />
          </button>
        </div>
        <p className="text-center text-xs text-gray-600 mt-2">
          Answers grounded in ingested arXiv papers
        </p>
      </div>
    </div>
  )
}
