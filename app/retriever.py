import hashlib

import numpy as np

from .config import CACHE_DIR, EMBED_MODEL, get_client
from .ingest import Chunk, load_corpus

BATCH = 64


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or load_corpus()
        self.matrix = self._load_or_build()

    def _embed(self, texts: list[str]) -> np.ndarray:
        client = get_client()
        out = []
        for i in range(0, len(texts), BATCH):
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + BATCH])
            out.extend(d.embedding for d in resp.data)
        m = np.array(out, dtype=np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def _load_or_build(self) -> np.ndarray:
        texts = [c.embed_text for c in self.chunks]
        key = hashlib.sha256((EMBED_MODEL + "\x00".join(texts)).encode()).hexdigest()[:16]
        path = CACHE_DIR / f"emb_{key}.npy"
        if path.exists():
            return np.load(path)
        matrix = self._embed(texts)
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(path, matrix)
        return matrix

    def search(self, query: str, k: int = 5, drug: str | None = None,
               source_type: str | None = None) -> list[tuple[Chunk, float]]:
        q = self._embed([query])[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)
        results = []
        for idx in order:
            c = self.chunks[idx]
            if drug and c.drug != drug.lower():
                continue
            if source_type and c.source_type != source_type:
                continue
            results.append((c, float(scores[idx])))
            if len(results) == k:
                break
        return results


if __name__ == "__main__":
    r = Retriever()
    for query, drug in [("contraindications", "warfarin"),
                        ("renal impairment eGFR threshold", "metformin"),
                        ("bleeding risk with SSRIs", None)]:
        print(f"\n== {query} (drug={drug})")
        for chunk, score in r.search(query, k=3, drug=drug):
            print(f"  {score:.3f}  {chunk.file} > {chunk.section}")