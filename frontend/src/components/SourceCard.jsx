export default function SourceCard({ source }) {
  const url = `https://arxiv.org/abs/${source.arxiv_id}`

  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="block bg-input border border-border rounded-lg p-3 hover:border-accent
                 transition-colors group"
    >
      <p className="text-xs font-medium text-gray-200 line-clamp-2 group-hover:text-accent
                    transition-colors leading-snug">
        {source.title}
      </p>
      <div className="flex items-center justify-between mt-2">
        <span className="text-xs text-gray-500">{source.arxiv_id}</span>
        <span className="text-xs text-gray-600">
          score {source.score?.toFixed(3)}
        </span>
      </div>
    </a>
  )
}
