import ReactMarkdown from 'react-markdown'
import SourceCard from './SourceCard'
import { Telescope, User } from 'lucide-react'

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-4 px-4 py-6 ${isUser ? '' : 'bg-surface'}`}>
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
                      bg-input text-sm">
        {isUser
          ? <User size={16} className="text-gray-300" />
          : <Telescope size={16} className="text-accent" />
        }
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0 space-y-3">
        <div className="prose prose-invert prose-sm max-w-none text-gray-100 leading-relaxed">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>

        {/* Sources */}
        {message.sources?.length > 0 && (
          <div className="mt-4">
            <p className="text-xs text-gray-500 mb-2 uppercase tracking-wide">Sources</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {message.sources.map((s) => (
                <SourceCard key={s.chunk_id} source={s} />
              ))}
            </div>
          </div>
        )}

        {/* Model badge */}
        {message.model && (
          <p className="text-xs text-gray-600">{message.model}</p>
        )}
      </div>
    </div>
  )
}
