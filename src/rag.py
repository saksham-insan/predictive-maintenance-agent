"""
RAG (Retrieval-Augmented Generation) Module for Predictive Maintenance Domain Knowledge.

Design Rationale:
-----------------
For a specialized corpus of 5 authoritative engineering failure mode documents,
a TF-IDF vectorizer (sklearn.feature_extraction.text.TfidfVectorizer) paired with
cosine similarity is optimal:
1. Zero vector DB overhead / zero network latency (sub-millisecond retrieval in-memory).
2. Deterministic exact-term and n-gram keyword matching on domain engineering units
   (e.g., "min*Nm", "200 min", "TWF", "OSF", "PWF", "HDF", "RNF", "8.6 K").
3. Completely local and self-contained without needing embedding model weights or paid APIs.
"""

import os
import glob
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KB_DIR = os.path.join(PROJECT_ROOT, "src", "knowledge_base")


@dataclass
class RetrievedDoc:
    doc_id: str
    filename: str
    title: str
    score: float
    content: str


class FailureModeKnowledgeBase:
    """
    In-memory TF-IDF knowledge base indexing industrial failure mode engineering docs.
    """

    def __init__(self, kb_dir: str = DEFAULT_KB_DIR):
        self.kb_dir = kb_dir
        self.documents: List[dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self):
        md_files = glob.glob(os.path.join(self.kb_dir, "*.md"))
        if not md_files:
            raise FileNotFoundError(f"No markdown documents found in knowledge base directory '{self.kb_dir}'")

        docs = []
        for file_path in sorted(md_files):
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract title from the first heading line
            lines = content.strip().splitlines()
            title = lines[0].replace("#", "").strip() if lines else filename

            docs.append({
                "doc_id": os.path.splitext(filename)[0],
                "filename": filename,
                "title": title,
                "content": content
            })

        self.documents = docs
        corpus_texts = [d["content"] for d in self.documents]

        # Use unigram and bigram tokenization with sublinear term frequency scaling
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w\.\-\*]+\b"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query: str, top_k: int = 2, min_score: float = 0.03) -> List[RetrievedDoc]:
        """
        Retrieves the top-k most relevant documents for the given text query.
        """
        if not query or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        ranked_indices = np.argsort(sims)[::-1]

        results = []
        for idx in ranked_indices:
            score = float(sims[idx])
            if score < min_score:
                continue
            doc = self.documents[idx]
            results.append(RetrievedDoc(
                doc_id=doc["doc_id"],
                filename=doc["filename"],
                title=doc["title"],
                score=round(score, 4),
                content=doc["content"]
            ))
            if len(results) >= top_k:
                break

        return results

    def retrieve_context_string(self, query: str, top_k: int = 2, min_score: float = 0.03) -> str:
        """
        Returns retrieved document contexts formatted with clean source headers,
        ready to be injected directly into LLM prompts.
        """
        docs = self.retrieve(query, top_k=top_k, min_score=min_score)
        if not docs:
            return "No matching domain knowledge base records found."

        sections = []
        for doc in docs:
            header = f"--- SOURCE: {doc.filename} (relevance={doc.score:.2f}) ---"
            sections.append(f"{header}\n{doc.content}\n")

        return "\n".join(sections).strip()


# Singleton instance
_KB_SINGLETON: Optional[FailureModeKnowledgeBase] = None


def get_knowledge_base() -> FailureModeKnowledgeBase:
    """
    Singleton accessor for FailureModeKnowledgeBase.
    """
    global _KB_SINGLETON
    if _KB_SINGLETON is None:
        _KB_SINGLETON = FailureModeKnowledgeBase()
    return _KB_SINGLETON


if __name__ == "__main__":
    kb = get_knowledge_base()
    test_query = "tool wear 215 min torque 55 Nm rising, Type M"
    results = kb.retrieve(test_query, top_k=2)
    print(f"Query: {test_query}\n")
    for r in results:
        print(f"-> {r.filename} (Score: {r.score:.4f}, Title: {r.title})")

    context = kb.retrieve_context_string(test_query, top_k=2)
    print("\nFormatted Context Output:\n" + "=" * 60)
    print(context[:400] + "...\n" + "=" * 60)
