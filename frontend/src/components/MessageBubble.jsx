import ReactMarkdown from 'react-markdown'
import { Telescope, User } from 'lucide-react'

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-4 px-4 py-6 ${isUser ? '' : 'bg-surface'}`}>
      <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-input">
        {isUser
          ? <User size={16} className="text-gray-300" />
          : <Telescope size={16} className="text-accent" />
        }
      </div>

      <div className="flex-1 min-w-0 space-y-1">
        <div className="prose prose-invert prose-sm max-w-none text-gray-100 leading-relaxed">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
        {message.model && (
          <p className="text-xs text-gray-600 pt-1">{message.model}</p>
        )}
      </div>
    </div>
  )
}
