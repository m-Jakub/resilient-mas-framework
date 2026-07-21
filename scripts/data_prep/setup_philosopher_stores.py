"""
Setup script for creating RAG vector stores for all philosophers.

This script dynamically loads philosopher configurations from config/settings.py
and creates Chroma vector stores for their ethical texts.

Supports both PDF and TXT files automatically.

Usage:
    python setup_philosopher_stores.py --all
    python setup_philosopher_stores.py --philosopher plato
    python setup_philosopher_stores.py --philosopher mill
    
Required:
    - Source files in data/<philosopher>/ directories (PDF or TXT)
    - GOOGLE_API_KEY in .env
    - config/settings.py with PHILOSOPHERS dictionary
"""

import argparse
from pathlib import Path
import sys
from pathlib import Path

# Ensure repository root is on sys.path for absolute imports to resolve config
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    PyPDFLoader = None  # PDF support optional
from dotenv import load_dotenv
import os

load_dotenv()

# Import philosopher configuration dynamically from config
from config.settings import PHILOSOPHERS

def get_philosopher_config(philosopher_key: str):
    """
    Get configuration for a philosopher from settings.
    
    Args:
        philosopher_key: Key from PHILOSOPHERS dict (e.g., 'mill', 'kant', 'plato')
        
    Returns:
        Dict with source_dir, store_dir, collection_name
    """
    if philosopher_key not in PHILOSOPHERS:
        return None
    
    philosopher_data = PHILOSOPHERS[philosopher_key]
    
    # Build paths
    source_dir = Path("data") / philosopher_key
    collection_name = philosopher_data.get("collection_name")
    store_dir = Path("data/chroma") / collection_name if collection_name else None
    
    return {
        "name": philosopher_data.get("name"),
        "source_dir": source_dir,
        "store_dir": store_dir,
        "collection_name": collection_name
    }


def discover_source_files(source_dir: Path):
    """
    Auto-discover PDF and TXT files in philosopher's directory.
    
    Args:
        source_dir: Path to philosopher's source directory
        
    Returns:
        List of (filepath, file_type) tuples
    """
    files = []
    
    # Find PDFs
    for pdf_file in source_dir.glob("*.pdf"):
        files.append((pdf_file, "pdf"))
    
    # Find TXT files
    for txt_file in source_dir.glob("*.txt"):
        files.append((txt_file, "txt"))
    
    return files


def load_philosopher_texts(philosopher_key: str):
    """
    Load texts for a specific philosopher (auto-detects PDF or TXT).
    
    Args:
        philosopher_key: Key from PHILOSOPHERS dict (e.g., 'plato', 'mill')
        
    Returns:
        List of loaded documents
    """
    config = get_philosopher_config(philosopher_key)
    
    if not config:
        print(f"Error: Unknown philosopher '{philosopher_key}'")
        return []
    
    source_dir = config["source_dir"]
    philosopher_name = config["name"]
    
    print(f"\n{'='*70}")
    print(f"Loading texts for: {philosopher_name.upper()} ({philosopher_key})")
    print(f"{'='*70}")
    print(f"Source directory: {source_dir}")
    
    if not source_dir.exists():
        print(f"  ✗ Directory not found: {source_dir}")
        return []
    
    documents = []
    files = discover_source_files(source_dir)
    
    if not files:
        print(f"  ✗ No PDF or TXT files found in {source_dir}")
        return []
    
    print(f"Found {len(files)} source files")
    
    for filepath, file_type in files:
        try:
            if file_type == "pdf":
                loader = PyPDFLoader(str(filepath))
            else:  # txt
                loader = TextLoader(str(filepath), encoding='utf-8')
            
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata['philosopher'] = philosopher_key
                doc.metadata['source_file'] = filepath.name
            
            print(f"  ✓ Loaded {filepath.name} ({len(docs)} documents) [{file_type.upper()}]")
            documents.extend(docs)
        except Exception as e:
            print(f"  ✗ Failed to load {filepath.name}: {e}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks suitable for embeddings.
    """
    print(f"\nSplitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"  ✓ Created {len(split_docs)} chunks")
    
    return split_docs


def create_vector_store(documents, persist_dir: Path, collection_name: str):
    """
    Create Chroma vector store for philosopher.
    
    Args:
        documents: List of document chunks
        persist_dir: Directory to persist vector store
        collection_name: Name of the collection (e.g., 'mill_store')
    """
    print(f"\nCreating vector store at: {persist_dir}")
    print(f"Collection name: {collection_name}")
    
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize embeddings (matching philosopher_agents.py)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",  # 1024 dimensions - matches PhilosopherAgent
        model_kwargs={'device': device}
    )
    
    # Create vector store
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=collection_name
    )
    
    print(f"  ✓ Vector store created with {len(documents)} chunks")
    
    return vectorstore


def test_vector_store(vectorstore, philosopher_key: str):
    """
    Test the created vector store with sample queries.
    """
    # Philosopher-specific test queries
    test_queries = {
        "plato": [
            "What is the Form of the Good?",
            "How does Socrates define justice in the Republic?",
            "What is the tripartite soul?"
        ],
        "aristotle": [
            "What is eudaimonia?",
            "How does Aristotle define virtue?",
            "What is phronesis or practical wisdom?"
        ],
        "bentham": [
            "What is the principle of utility?",
            "How does the hedonic calculus work?",
            "What is the greatest happiness principle?"
        ],
        "nietzsche": [
            "What is master morality?",
            "What does Nietzsche mean by the Übermensch?",
            "What is the genealogy of morals?"
        ],
        "st_augustine": [
            "What is the City of God?",
            "What role does grace play in Augustine's ethics?",
            "How does Augustine view human nature after the Fall?"
        ],
        "mill": [
            "What is the difference between higher and lower pleasures?",
            "What is the harm principle?",
            "How does Mill refine Bentham's utilitarianism?"
        ],
        "kant": [
            "What is the categorical imperative?",
            "Why must we treat persons as ends in themselves?",
            "What is the relationship between duty and morality?"
        ],
        "aquinas": [
            "What is natural law?",
            "How do the cardinal virtues relate to the theological virtues?",
            "What is beatitude?"
        ]
    }
    
    queries = test_queries.get(philosopher_key, ["What is virtue?", "What is the good life?"])
    
    config = get_philosopher_config(philosopher_key)
    philosopher_name = config["name"] if config else philosopher_key
    
    print(f"\n{'='*70}")
    print(f"TESTING VECTOR STORE: {philosopher_name.upper()}")
    print(f"{'='*70}")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        try:
            docs = retriever.get_relevant_documents(query)
            
            for i, doc in enumerate(docs, 1):
                preview = doc.page_content[:150].replace('\n', ' ')
                source = doc.metadata.get('source_file', 'unknown')
                print(f"{i}. [{source}] {preview}...")
        except Exception as e:
            print(f"✗ Query failed: {e}")


def setup_philosopher(philosopher_key: str, chunk_size: int, chunk_overlap: int):
    """
    Setup vector store for a single philosopher.
    """
    config = get_philosopher_config(philosopher_key)
    
    if not config:
        print(f"Error: Unknown philosopher '{philosopher_key}'")
        print(f"Available: {', '.join(get_available_philosophers())}")
        return False
    
    source_dir = config["source_dir"]
    store_dir = config["store_dir"]
    collection_name = config["collection_name"]
    
    if not store_dir or not collection_name:
        print(f"Error: Philosopher '{philosopher_key}' missing collection_name in config")
        return False
    
    # Verify source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return False
    
    # Load documents
    documents = load_philosopher_texts(philosopher_key)
    
    if not documents:
        print(f"\n✗ No documents loaded for {philosopher_key}")
        return False
    
    # Split documents
    split_docs = split_documents(documents, chunk_size, chunk_overlap)
    
    # Create vector store
    vectorstore = create_vector_store(split_docs, store_dir, collection_name)
    
    # Test vector store
    test_vector_store(vectorstore, philosopher_key)
    
    print(f"\n{'='*70}")
    print(f"✓ SETUP COMPLETE: {config['name'].upper()}")
    print(f"{'='*70}")
    print(f"Vector store: {store_dir}")
    print(f"Collection: {collection_name}")
    
    return True


def get_available_philosophers():
    """Get list of philosopher keys that have collection_name configured."""
    available = []
    for key, data in PHILOSOPHERS.items():
        if data.get("collection_name"):
            available.append(key)
    return available


def main():
    parser = argparse.ArgumentParser(
        description="Setup philosopher vector stores for RAG (dynamically loaded from config)"
    )
    
    available_philosophers = get_available_philosophers()
    
    parser.add_argument(
        "--philosopher",
        type=str,
        choices=available_philosophers,
        help="Setup store for specific philosopher"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Setup stores for all philosophers"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks for embedding"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between text chunks"
    )
    
    args = parser.parse_args()
    
    # Verify API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment")
        print("Please set it in .env file")
        return
    
    print("="*70)
    print("PHILOSOPHER VECTOR STORES SETUP")
    print("="*70)
    print(f"Loaded {len(available_philosophers)} philosophers from config/settings.py")
    print(f"Available: {', '.join(available_philosophers)}")
    print("="*70)
    
    if args.all:
        # Setup all philosophers
        print(f"\nSetting up stores for ALL philosophers...")
        
        success_count = 0
        for philosopher_key in available_philosophers:
            if setup_philosopher(philosopher_key, args.chunk_size, args.chunk_overlap):
                success_count += 1
        
        print("\n" + "="*70)
        print(f"BATCH SETUP COMPLETE: {success_count}/{len(available_philosophers)} succeeded")
        print("="*70)
        
    elif args.philosopher:
        # Setup single philosopher
        setup_philosopher(args.philosopher, args.chunk_size, args.chunk_overlap)
        
    else:
        print("Error: Must specify --philosopher <name> or --all")
        print(f"Available philosophers: {', '.join(available_philosophers)}")
        return
    
    print("\nNext steps:")
    print("1. Verify stores in data/chroma/")
    print("2. Test integration: python tests/test_id_rag_integration.py")
    print("3. Run debate: python tests/demo_team_debate.py")


if __name__ == "__main__":
    main()
