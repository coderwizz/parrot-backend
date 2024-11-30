import os
import json
from flask import Flask, request, jsonify, url_for
import replicate
import numpy as np
import pandas as pd
from flask_cors import CORS
import tempfile
import shutil

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all domains and routes
CORS(app, supports_credentials=True)

# Load the list of keywords from Excel
def load_keywords_from_excel(excel_file='key_words_vocab.xlsx'):
    try:
        df = pd.read_excel(excel_file, engine='openpyxl')
    except Exception as e:
        return []

    if 'pleasure' not in df.columns:
        return []

    keywords = df['pleasure'].dropna().tolist()
    keywords = [keyword.strip() for keyword in keywords if isinstance(keyword, str)]
    return keywords

# Load emoji embeddings from a JSON file
def load_emoji_embeddings(json_file='emoji_embeddings.json'):
    try:
        with open(json_file, 'r') as file:
            emoji_embeddings = json.load(file)
    except Exception as e:
        emoji_embeddings = {}
    return emoji_embeddings

# Load keyword embeddings from the JSON file
def load_keyword_embeddings(json_file='keyword_embeddings.json'):
    try:
        with open(json_file, 'r') as file:
            keyword_embeddings = json.load(file)
    except Exception as e:
        keyword_embeddings = {}
    return keyword_embeddings

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.route('/api/EmojiMatcher', methods=['POST', 'OPTIONS'])
def python_emoji_matcher():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({"message": "CORS preflight successful"})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 200

    # Process POST request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_path = save_image(image_file)  # Save the image to a temporary directory
    image_uri = url_for('static', filename=f'image/{os.path.basename(image_path)}', _external=True)

    # Load necessary data
    keywords = load_keywords_from_excel()  # Load the keywords from Excel
    emoji_embeddings = load_emoji_embeddings()  # Load emoji embeddings from JSON
    keyword_embeddings = load_keyword_embeddings()  # Load keyword embeddings from JSON

    image_embedding = get_image_embedding(image_uri, keywords)  # Pass keywords here
    word = find_best_matching_word(image_embedding, keywords)  # Pass keywords here
    word_embedding = get_keyword_embedding(word, keyword_embeddings)  # Pass keyword embeddings here
    best_emoji = find_closest_emoji(word_embedding, emoji_embeddings)  # Pass emoji embeddings here

    # Delete the image after processing
    os.remove(image_path)

    return jsonify({
        'bestEmoji': best_emoji,
        'imageUri': image_uri  # Return the URI for the image
    })

def save_image(image_file):
    """Save the uploaded image to a temporary writable directory and return the file path."""
    # Use a temporary directory for saving images
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    image_path = os.path.join(temp_dir, image_file.filename)

    os.makedirs(os.path.dirname(image_path), exist_ok=True)  # Ensure the directory exists
    image_file.save(image_path)
    return image_path

def get_image_embedding(image_uri, keywords):
    """Get the image embedding using Replicate's CLIP model."""
    keywords_string = " | ".join(keywords)  # Use the passed keywords list
    
    # Debugging: Print the image URI and text (keywords_string)
    print(f"Image URI: {image_uri}")
    print(f"Text (keywords): {keywords_string}")
    
    input_data = {
        "input": {
            "image": image_uri,  # Pass the public URI of the image
            "text": keywords_string  # Use the concatenated keyword string
        }
    }

    # Call Replicate model for the image embedding
    try:
        output = replicate.run(
            "cjwbw/clip-vit-large-patch14:566ab1f111e526640c5154e712d4d54961414278f89d36590f1425badc763ecb", 
            input=input_data
        )
        return np.array(output)
    except Exception as e:
        print(f"Error getting image embedding: {e}")
        return np.zeros(512)  # Return a zero vector if error occurs

def find_best_matching_word(image_embedding, keywords):
    """Find the best matching word from the keywords based on the image embedding."""
    best_index = np.argmax(image_embedding)
    best_word = keywords[best_index]
    return best_word

def get_keyword_embedding(word, keyword_embeddings):
    """Get the embedding for a given keyword."""
    try:
        return np.array(keyword_embeddings.get(word, np.zeros(300)))
    except Exception as e:
        return np.zeros(300)

def find_closest_emoji(word_embedding, emoji_embeddings):
    """Find the closest emoji for the given word embedding."""
    best_emoji = ''
    best_score = -float('inf')

    for emoji, emoji_embedding in emoji_embeddings.items():
        score = cosine_similarity(word_embedding, emoji_embedding)
        if score > best_score:
            best_score = score
            best_emoji = emoji

    return best_emoji

# Preflight handling for all routes
@app.before_request
def handle_options_request():
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS preflight successful"})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 200

if __name__ == '__main__':
    # Load data at startup
    keywords = load_keywords_from_excel()
    emoji_embeddings = load_emoji_embeddings()  # Load emoji embeddings from JSON
    keyword_embeddings = load_keyword_embeddings()  # Load keyword embeddings from JSON
