import ollama
import textwrap
import pickle
import os

# Load the dataset
dataset = []
with open('data/SukanyaSamriddhiAccountSchemeRule.pdf-temp.txt', 'r', encoding='utf-8') as file:
    dataset = [line.strip() for line in file.readlines() if line.strip()]
    print(f'Loaded {len(dataset)} entries')

# Define constants and variables
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
MAX_CHUNK_SIZE = 1000  # Maximum size for each chunk (adjustable)
VECTOR_DB = []

# Function to add a chunk to the vector database
def add_chunk_to_database(chunk):
    if len(chunk) > MAX_CHUNK_SIZE:
        sub_chunks = textwrap.wrap(chunk, width=MAX_CHUNK_SIZE)
        for sub_chunk in sub_chunks:
            try:
                embedding = ollama.embed(model=EMBEDDING_MODEL, input=sub_chunk)['embeddings'][0]
                VECTOR_DB.append((sub_chunk, embedding))
            except Exception as e:
                print(f"Failed to embed sub-chunk: {e}")
    else:
        try:
            embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            VECTOR_DB.append((chunk, embedding))
        except Exception as e:
            print(f"Failed to embed chunk: {e}")

# Process each chunk and add to VECTOR_DB
for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')

# Save VECTOR_DB to disk
with open('vectordbs/SSAS_vector_db.pkl', 'wb') as f:
    pickle.dump(VECTOR_DB, f)

print(f'VECTOR_DB saved to vector_db.pkl with {len(VECTOR_DB)} entries')
