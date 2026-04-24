
# rag.py - ChromaDB context storage and retrieval for support tickets

import os
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import pandas as pd

# Constants
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../chroma_db")
COLLECTION_NAME = "support_tickets"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=DefaultEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"}
)

def index_tickets_from_csv(csv_path):
    """
    Index support tickets from a CSV file into ChromaDB.
    CSV must have columns: tweet_id, text, priority, author_id
    """
    df_tickets = pd.read_csv(csv_path)
    df_tickets = df_tickets[df_tickets['text'].notna() & (df_tickets['text'].astype(str).str.strip() != '')].reset_index(drop=True)
    ids = df_tickets['tweet_id'].astype(str).tolist()
    documents = df_tickets['text'].tolist()
    metadatas = [
        {"priority": int(p), "author": str(a)}
        for p, a in zip(df_tickets['priority'], df_tickets['author_id'])
    ]
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
    print(f"✅ Successfully indexed {collection.count()} tickets into ChromaDB.")

def retrieve_support_context(query, n_results=3):
    """
    Retrieve relevant support ticket context for a given query.
    Returns a formatted string and raw results.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    context_list = []
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        priority_label = "Urgent" if meta['priority'] == 1 else "Normal"
        context_list.append(f"Related Ticket [{priority_label}]: {doc}")
    return "\n---\n".join(context_list), results

# If run as a script, allow indexing and testing
if __name__ == "__main__":
    # Example: index tickets from a CSV file
    csv_path = os.path.join(os.path.dirname(__file__), "../be/dataset_extracted/twcs/twcs.csv")
    if os.path.exists(csv_path):
        index_tickets_from_csv(csv_path)
    else:
        print(f"CSV not found at {csv_path}. Skipping indexing.")

    # Example query
    query = "My order is 3 days late and no one is answering the phone!!"
    context, raw_data = retrieve_support_context(query)
    print("🔍 USER QUERY:", query)
    print("\n📚 RETRIEVED CONTEXT:\n", context)