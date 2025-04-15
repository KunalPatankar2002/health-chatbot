
from flask import Flask, request, jsonify


import ollama
import pickle
import os


app = Flask(__name__)



# Define constants
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'mistral:7b-instruct-q4_K_M'
VDB_PATH = 'vectordbs'

# Define your topic descriptions (can be expanded)
TOPIC_PROFILES = {
    'JSY': 'JananiSurakshaYojana (JSY) is a safe motherhood intervention under the National Health Mission. It is being implemented with the objective of reducing maternal and neonatal mortality by promoting institutional delivery among poor pregnant women. The scheme, launched on 12 April 2005 by the Hon\'ble Prime Minister, is under implementation in all states and Union Territories (UTs), with a special focus on Low Performing States (LPS). JSY is a centrally sponsored scheme, which integrates cash assistance with delivery and post-delivery care. The Yojana has identified Accredited Social Health Activist (ASHA) as an effective link between the government and pregnant women.  Important Features of JSY The scheme focuses on poor pregnant woman with a special dispensation for states that have low institutional delivery rates, namely, the states of Uttar Pradesh, Uttarakhand, Bihar, Jharkhand, Madhya Pradesh, Chhattisgarh, Assam, Rajasthan, Orissa, and Jammu and Kashmir. While these states have been named Low Performing States (LPS), the remaining states have been named High Performing states (HPS).',
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
        print("⚠️ Low confidence in topic detection. You may want to specify the topic manually.")
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



def getResponse(input_query):
   
    # --- Main logic ---
    topic = detect_topic_by_embedding(input_query)

    if topic is None:
        return "Topic detection failed. Please provide a more specific query."

    VECTOR_DB = load_vector_db(topic)
    retrieved_knowledge = retrieve(input_query, VECTOR_DB)

    # Build context
    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. Don't make up any new information:
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

    return generated_response


@app.route('/chat/<query>', methods=['GET'])
def chat(query):

    # data = request.get_json()
    
    try:
  
        output = getResponse(query)
        response = {
            'response': output
        }
        return jsonify(response)
    except ValueError:
        return jsonify({'Error': 'Invalid input'}), 400


if __name__ == '__main__':
    app.run()

### # To run the server, use the command: python REST.py
### # To expose the port to the internet: ngrok http --url=sharp-thankful-anchovy.ngrok-free.app 5000
### # 5000 is the port number on local