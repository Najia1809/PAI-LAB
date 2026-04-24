from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

app = Flask(__name__)


df = pd.read_csv('hotel_qna.csv')
embeddings = np.load('hotel_embeddings.npy')


faiss_index = faiss.read_index('faiss_index.index')


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_answer(query):
    query_embedding = model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding), 1)
    answer = df["answer"].iloc[indices[0][0]]
    return answer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = get_answer(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)