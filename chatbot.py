# import ollama
# import pickle
# import os

# # Load the saved VECTOR_DB
# VECTOR_DB = []
# if os.path.exists('vectordbs/JSY_vector_db.pkl'):
#     with open('vectordbs/JSY_vector_db.pkl', 'rb') as f:
#         VECTOR_DB = pickle.load(f)
#     print(f'Loaded VECTOR_DB with {len(VECTOR_DB)} entries')
# else:
#     print('No saved VECTOR_DB found. Please run build_vector_db.py first.')

# # Define constants and functions for retrieval
# EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
# LANGUAGE_MODEL = 'mistral:7b-instruct-q4_K_M'

# def cosine_similarity(a, b):
#     dot_product = sum([x * y for x, y in zip(a, b)])
#     norm_a = sum([x ** 2 for x in a]) ** 0.5
#     norm_b = sum([x ** 2 for x in b]) ** 0.5
#     return dot_product / (norm_a * norm_b)

# def retrieve(query, top_n=3):
#     query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
#     similarities = []
#     for chunk, embedding in VECTOR_DB:
#         similarity = cosine_similarity(query_embedding, embedding)
#         similarities.append((chunk, similarity))
#     similarities.sort(key=lambda x: x[1], reverse=True)
#     return similarities[:top_n]

# # Chatbot logic
# input_query = input('Ask me a question: ')
# retrieved_knowledge = retrieve(input_query)

# # print('Retrieved knowledge:')
# # for chunk, similarity in retrieved_knowledge:
# #     print(f' - (similarity: {similarity:.2f}) {chunk}')

# instruction_prompt = f'''You are a helpful chatbot.
# Use only the following pieces of context to answer the question. Don't make up any new information:
# {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
# '''

# # Query the language model for a response
# stream = ollama.chat(
#     model=LANGUAGE_MODEL,
#     messages=[{'role': 'system', 'content': instruction_prompt},
#               {'role': 'user', 'content': input_query}],
#     stream=True,
# )

# # Print the chatbot response in real-time
# print('Chatbot response:')
# for chunk in stream:
#     print(chunk['message']['content'], end='', flush=True)

# # After generating the chatbot response
# generated_response = ''.join([chunk['message']['content'] for chunk in stream])
# print("Generated Response:")
# print(generated_response)

# # Check if the response is directly linked to retrieved knowledge
# relevant_chunks = [chunk for chunk, similarity in retrieved_knowledge]
# hallucinated = True
# hallucinated_info = []  # To store any hallucinated information

# # Check if generated response contains relevant chunks
# for chunk in relevant_chunks:
#     if chunk in generated_response:
#         hallucinated = False
#         break

# # If hallucinated, extract and print the hallucinated info
# if hallucinated:
#     print("Potential hallucination detected: Response not supported by context.")
    
#     # Split generated response into sentences for easier assessment
#     sentences = generated_response.split(". ")
    
#     for sentence in sentences:
#         # Check if the sentence is not supported by any relevant chunk
#         if not any(relevant_chunk in sentence for relevant_chunk in relevant_chunks):
#             hallucinated_info.append(sentence)
    
#     print("Hallucinated Information:")
#     for info in hallucinated_info:
#         print(f" - {info}")

# else:
#     print("Response seems grounded in the retrieved knowledge.")

#######################################################################################################################################



import ollama
import pickle
import os

# Define constants
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'mistral:7b-instruct-q4_K_M'
VDB_PATH = 'vectordbs'

# Define your topic descriptions (can be expanded)
TOPIC_PROFILES = {
    'JSY': 'Janani Suraksha Yojana (JSY) is a safe motherhood program under the National Health Mission in India. It aims to reduce maternal and neonatal mortality by promoting institutional deliveries among poor and rural women. JSY provides cash incentives to eligible pregnant women and community health workers (ASHAs) for facilitating antenatal care, delivery in health facilities, and postnatal support. The scheme targets women from Below Poverty Line (BPL) households, especially in low-performing states, and includes SC/ST women regardless of age or parity.',
    'SSAS': 'INTRODUCTION: Minimum deposit ₹ 250/- Maximum deposit ₹ 1.5 Lakh in a financial year. Account can be opened in the name of a girl child till she attains the age of 10 years. Only one account can be opened in the name of a girl child. Account can be opened in Post offices and in authorised banks. Withdrawal shall be allowed for the purpose of higher education of the Account holder to meet education expenses. The account can be prematurely closed in case of marriage of girl child after her attaining the age of 18 years. The account can be transferred anywhere in India from one Post office/Bank to another. The account shall mature on completion of a period of 21 years from the date of opening of account. Deposit qualifies for deduction under Sec.80-C of I.T.Act.',

}

# Load topic embeddings once
print("Embedding topic profiles...")
topic_embeddings = {
    topic: ollama.embed(model=EMBEDDING_MODEL, input=desc)['embeddings'][0]
    for topic, desc in TOPIC_PROFILES.items()
}
print("Topic profiles embedded.\n")

# Cosine similarity
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot / (norm_a * norm_b)

# Topic detection
def detect_topic_by_embedding(query, threshold=0.60):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = {
        topic: cosine_similarity(query_embedding, topic_embedding)
        for topic, topic_embedding in topic_embeddings.items()
    }
    best_topic = max(similarities, key=similarities.get)
    best_score = similarities[best_topic]
    print(f"Detected topic: {best_topic} (similarity: {best_score:.2f})")

    if best_score >= threshold:
        return best_topic
    else:
        print("Low confidence in topic detection. You may want to specify the topic manually.")
        return None

# Load vector DB
def load_vector_db(topic_name):
    path = os.path.join(VDB_PATH, f'{topic_name}_vector_db.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            vector_db = pickle.load(f)
        print(f'Loaded vector DB: {topic_name} ({len(vector_db)} entries)')
        return vector_db
    else:
        raise FileNotFoundError(f'No vector DB found for topic: {topic_name}')

# Retrieve top matching chunks
def retrieve(query, vector_db, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in vector_db:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# --- Main logic ---
input_query = input('Ask me a question: ')
topic = detect_topic_by_embedding(input_query)

if topic is None:
    exit("Topic detection failed. Please provide a more specific query.")

VECTOR_DB = load_vector_db(topic)
retrieved_knowledge = retrieve(input_query, VECTOR_DB)

# Build context
instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information. Dont make any information up:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''

# Query the model
stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[{'role': 'system', 'content': instruction_prompt},
              {'role': 'user', 'content': input_query}],
    stream=True,
)

print('\nChatbot response:')
generated_response_parts = []
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
    generated_response_parts.append(chunk['message']['content'])

generated_response = ''.join(generated_response_parts)

# print("\n\nGenerated Response:")
# print(generated_response)
