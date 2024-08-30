from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd

# Load the pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the FAISS index
index = faiss.read_index('course_embeddings.index')

# Load the course data (assuming the course_data list is already populated)
course_data = [
    {"title": "Course 1", "description": "Description for course 1"},
    {"title": "Course 2", "description": "Description for course 2"}
]
df = pd.DataFrame(course_data)

app = Flask(__name__)
api = Api(app)

class Chatbot(Resource):
    def post(self):
        user_input = request.get_json().get("input")
        
        # Encode user input
        inputs = tokenizer.encode_plus(
            user_input,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
        user_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        # Search for similar embeddings
        D, I = index.search(user_embedding, k=5)  # Retrieve top 5 results

        # Get the top-N similar results
        results = []
        for i in range(len(I[0])):
            results.append({"text": df.iloc[I[0][i]]["description"], "score": float(D[0][i])})
        
        return jsonify(results)

api.add_resource(Chatbot, '/chat')

if __name__ == '__main__':
    app.run(debug=True)
