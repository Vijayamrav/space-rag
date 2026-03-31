"""
Debug script — checks what's in Pinecone and tests retrieval directly.
Run: python debug_pinecone.py
"""
from app.core.pinecone_client import get_client, get_index
from app.core.config import settings

# 1. Index stats
pc    = get_client()
index = get_index()
stats = index.describe_index_stats()
print(f"\n=== Pinecone Index Stats ===")
print(f"Index name     : {settings.pinecone_index_name}")
print(f"Total vectors  : {stats['total_vector_count']}")
print(f"Dimension      : {stats['dimension']}")

if stats['total_vector_count'] == 0:
    print("\n❌ Index is EMPTY — ingest hasn't worked yet.")
else:
    print(f"\n✅ Index has {stats['total_vector_count']} vectors")

    # 2. Test retrieval directly
    from app.services.retrieval_service import retrieve
    query   = "exoplanet atmospheric conditions"
    results = retrieve(query, top_k=3, use_rerank=False)

    print(f"\n=== Retrieval Test: '{query}' ===")
    print(f"Results returned: {len(results)}")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r['title']}")
        print(f"     arxiv_id : {r['arxiv_id']}")
        print(f"     score    : {r['score']:.4f}")
        print(f"     text     : {r['text'][:200]}")
