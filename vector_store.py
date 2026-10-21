"""Explicit small-data cosine scan for local tests; PostgreSQL uses pgvector."""
import math


def cosine(left, right):
    if len(left) != len(right):
        raise ValueError('vector dimension mismatch')
    return sum(a*b for a, b in zip(left, right)) / (math.sqrt(sum(a*a for a in left))*math.sqrt(sum(b*b for b in right)))


def local_search(rows, vector, model_id, threshold, limit):
    result = []
    for row in rows:
        if row.get('vector_status') == 'ready' and row.get('embedding_model') == model_id and row.get('embedding_content_hash') == row['content_hash']:
            score = cosine(row['embedding'], vector)
            if score >= threshold:
                result.append(row | {'score': score})
    return sorted(result, key=lambda r: (-r['score'], r['id']))[:limit]
