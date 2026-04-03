import ChatWindow from './components/ChatWindow'
import { Telescope } from 'lucide-react'

export default function App() {
  return (
    <div className="flex h-screen bg-sidebar">
      {/* Sidebar */}
      <aside className="w-64 flex flex-col bg-sidebar border-r border-border p-4">
        <div className="flex items-center gap-2 px-2 py-3 mb-4">
          <Telescope size={22} className="text-accent" />
          <span className="font-semibold text-lg text-white">Space RAG</span>
        </div>
        <div className="mt-auto text-xs text-gray-600 px-2">
          Powered by arXiv · Pinecone · OpenRouter
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <ChatWindow />
      </main>
    </div>
  )
}
