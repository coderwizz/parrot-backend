import json
import numpy as np
import torch
import os
from gensim.models import KeyedVectors
from flask import Flask, request, jsonify
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Initialize the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Load pre-trained emoji embeddings
with open('emoji_embeddings.json', 'r') as f:
    emoji_embeddings = json.load(f)

# Define the file path for the Word2Vec model
word2vec_model_path = 'word2vec-small.model'

# Load the pre-trained Word2Vec model globally
word2vec_model = KeyedVectors.load(word2vec_model_path)

def load_keywords_from_excel(excel_file='key_words_vocab.xlsx'):
    try:
        df = pd.read_excel(excel_file, engine='openpyxl')
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return []

    # Assuming the 'pleasure' column contains the emoji words
    if 'pleasure' not in df.columns:
        print("Column 'pleasure' not found.")
        return []

    # Extract the words as a list from the 'pleasure' column
    keywords = df['pleasure'].dropna().tolist()
    keywords = [keyword.strip() for keyword in keywords if isinstance(keyword, str)]

    return keywords

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.route('/pythonEmojiMatcher', methods=['POST'])
def python_emoji_matcher():
    # Check if the image is in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Get the image from the request
    image_file = request.files['image']

    # Convert image to PIL format
    image = Image.open(image_file.stream)

    # Get the embedding for the image
    image_embedding = get_image_embedding(image)

    # Find the best matching word
    word = find_best_matching_word(image_embedding)

    # Get the Word2Vec embedding of the word
    word_embedding = get_word2vec_embedding(word)

    # Find the closest emoji based on word embedding
    best_emoji = find_closest_emoji(word_embedding)

    return jsonify({'bestEmoji': best_emoji})

def get_image_embedding(image):
    # Preprocess the image for CLIP
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        # Extract image features
        image_features = model.get_image_features(**inputs)
    return image_features[0].numpy()

def find_best_matching_word(image_embedding):
    candidate_words = load_keywords_from_excel()  # Load the emoji keywords list dynamically
    batch_size = 64  # You can adjust the batch size as per your memory constraints
    best_word = ''
    best_score = -float('inf')

    # Process the candidate words in batches
    for i in range(0, len(candidate_words), batch_size):
        batch_words = candidate_words[i:i+batch_size]
        
        # Preprocess the batch of words for CLIP
        inputs = processor(text=batch_words, return_tensors="pt", padding=True)
        with torch.no_grad():
            word_features = model.get_text_features(**inputs)

        # Calculate cosine similarity for each word in the batch
        for idx, word in enumerate(batch_words):
            score = cosine_similarity(word_features[idx].numpy(), image_embedding)
            if score > best_score:
                best_score = score
                best_word = word

    return best_word

def get_word2vec_embedding(word):
    try:
        return word2vec_model[word]
    except KeyError:
        return np.zeros(word2vec_model.vector_size)

def find_closest_emoji(word_embedding):
    best_emoji = ''
    best_score = -float('inf')

    for emoji, emoji_embedding in emoji_embeddings.items():
        score = cosine_similarity(word_embedding, emoji_embedding)
        if score > best_score:
            best_score = score
            best_emoji = emoji

    return best_emoji

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)