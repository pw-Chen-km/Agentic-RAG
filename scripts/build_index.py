#!/usr/bin/env python3
"""
Build sentence-level embedding index for semantic search.

Usage:
    python scripts/build_index.py \
        --chunks data/chunks.json \
        --output data/index \
        --model sentence-transformers/all-MiniLM-L6-v2
"""

import os
import json
import re
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'[.!?\n]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def load_chunks(chunks_file: str) -> List[Dict[str, Any]]:
    """Load chunks from file."""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if data and isinstance(data[0], dict):
        return data
    
    chunks = []
    for item in data:
        if isinstance(item, str):
            parts = item.split(':', 1)
            if len(parts) == 2:
                chunks.append({'id': parts[0], 'text': parts[1]})
    return chunks


def build_index(
    chunks_file: str,
    output_dir: str,
    model_name: str,
    device: str = None,
    batch_size: int = 32,
    query_prompt_name: str = None,
):
    """Build sentence-level embedding index."""
    from sentence_transformers import SentenceTransformer
    
    # Load chunks
    print(f"Loading chunks from: {chunks_file}")
    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks")
    
    # Create chunk lookup
    chunk_lookup = {c['id']: c for c in chunks}
    
    # Extract sentences
    print("Extracting sentences...")
    sentences = []
    sentence_to_chunk = []
    
    for chunk in tqdm(chunks, desc="Processing chunks"):
        chunk_sentences = split_sentences(chunk['text'])
        for sent in chunk_sentences:
            sentences.append(sent)
            sentence_to_chunk.append(chunk['id'])
    
    print(f"Total sentences: {len(sentences)}")
    
    # Load model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    if query_prompt_name and query_prompt_name not in model.prompts:
        raise ValueError("Embedding model has no requested query prompt")
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Encode sentences
    print("Encoding sentences...")
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        prompt=""
    )
    
    # Save index
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    index_file = output_path / "sentence_index.pkl"
    index_data = {
        'sentences': sentences,
        'embeddings': embeddings,
        'sentence_to_chunk': sentence_to_chunk,
        'chunks': chunk_lookup,
        'model_name': model_name,
        'query_prompt_name': query_prompt_name,
        'document_prompt': '',
    }
    
    print(f"Saving index to: {index_file}")
    with open(index_file, 'wb') as f:
        pickle.dump(index_data, f)
    
    print(f"Index built successfully!")
    print(f"  - Chunks: {len(chunks)}")
    print(f"  - Sentences: {len(sentences)}")
    print(f"  - Embedding dim: {embeddings.shape[1]}")


def main():
    parser = argparse.ArgumentParser(description="Build semantic search index")
    parser.add_argument("--chunks", "-c", required=True, help="Path to chunks.json")
    parser.add_argument("--output", "-o", required=True, help="Output directory for index")
    parser.add_argument("--model", "-m", default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Embedding model name or path")
    parser.add_argument("--device", "-d", default=None, help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--query-prompt-name", default=None)
    
    args = parser.parse_args()
    
    build_index(
        chunks_file=args.chunks,
        output_dir=args.output,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        query_prompt_name=args.query_prompt_name,
    )


if __name__ == "__main__":
    main()
