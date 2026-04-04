import { useState } from 'react'
import ChatWindow from './components/ChatWindow'
import { Telescope, Plus, MessageSquare, Trash2 } from 'lucide-react'

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2)
}

function newChat() {
  return { id: generateId(), title: 'New Chat', messages: [] }
}

export default function App() {
  const [chats,         setChats]         = useState([newChat()])
  const [activeChatId,  setActiveChatId]  = useState(chats[0].id)

  const activeChat = chats.find(c => c.id === activeChatId)

  function createChat() {
    const chat = newChat()
    setChats(prev => [chat, ...prev])
    setActiveChatId(chat.id)
  }

  function deleteChat(id) {
    setChats(prev => {
      const remaining = prev.filter(c => c.id !== id)
      if (remaining.length === 0) {
        const fresh = newChat()
        setActiveChatId(fresh.id)
        return [fresh]
      }
      if (activeChatId === id) setActiveChatId(remaining[0].id)
      return remaining
    })
  }

  function updateChat(id, messages) {
    setChats(prev => prev.map(c => {
      if (c.id !== id) return c
      // use first user message as title
      const firstUser = messages.find(m => m.role === 'user')
      const title = firstUser
        ? firstUser.content.slice(0, 40) + (firstUser.content.length > 40 ? '…' : '')
        : 'New Chat'
      return { ...c, messages, title }
    }))
  }

  return (
    <div className="flex h-screen bg-sidebar">
      {/* Sidebar */}
      <aside className="w-64 flex flex-col bg-sidebar border-r border-border">
        {/* Logo */}
        <div className="flex items-center gap-2 px-4 py-4 border-b border-border">
          <Telescope size={20} className="text-accent" />
          <span className="font-semibold text-white">AstroRAG</span>
        </div>

        {/* New chat button */}
        <div className="p-3">
          <button
            onClick={createChat}
            className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm
                       text-gray-300 border border-border hover:bg-input transition-colors"
          >
            <Plus size={15} /> New Chat
          </button>
        </div>

        {/* Chat list */}
        <div className="flex-1 overflow-y-auto px-2 space-y-0.5">
          {chats.map(chat => (
            <div
              key={chat.id}
              onClick={() => setActiveChatId(chat.id)}
              className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer
                          text-sm transition-colors ${
                activeChatId === chat.id
                  ? 'bg-input text-white'
                  : 'text-gray-400 hover:bg-input hover:text-white'
              }`}
            >
              <MessageSquare size={14} className="flex-shrink-0" />
              <span className="flex-1 truncate">{chat.title}</span>
              <button
                onClick={e => { e.stopPropagation(); deleteChat(chat.id) }}
                className="opacity-0 group-hover:opacity-100 text-gray-500
                           hover:text-red-400 transition-all"
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>

      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {activeChat && (
          <ChatWindow
            key={activeChatId}
            messages={activeChat.messages}
            onUpdate={msgs => updateChat(activeChatId, msgs)}
          />
        )}
      </main>
    </div>
  )
}
