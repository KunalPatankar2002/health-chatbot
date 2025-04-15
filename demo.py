import ollama
import textwrap

# Load the dataset

dataset = []
with open('pdfs/Janani suraksha yojana.pdf_sections.txt', 'r', encoding='utf-8') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')



# Implement the retrieval system

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'mistral:7b-instruct-q4_K_M'  # If using Ollama's Mistral 7B

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
VECTOR_DB = []

MAX_CHUNK_SIZE = 1000  # You can tweak this

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

for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f'Added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]



# Chatbot

input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
  print(f' - (similarity: {similarity:.2f}) {chunk}')

instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''
# print(instruction_prompt)

stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

# print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)