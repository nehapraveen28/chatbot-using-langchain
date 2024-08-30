import pandas as pd
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np
import torch

# Load the pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example course_data (replace with actual data)
course_data = [
    {"title": "Course 1", "description": "Description for course 1"},
    {"title": "Course 2", "description": "Description for course 2"}
]

# Create a pandas DataFrame to store the data
df = pd.DataFrame(course_data)

# Create embeddings for each row in the DataFrame
embeddings = []
for index, row in df.iterrows():
    inputs = tokenizer.encode_plus(
        row["description"],
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())

# Flatten the list of embeddings into a 2D NumPy array
embeddings = np.vstack(embeddings)

# Create a FAISS index
dimension = embeddings.shape[1]  # The size of the embedding vectors
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(embeddings)

# Optionally save the index to a file
faiss.write_index(index, 'course_embeddings.index')
