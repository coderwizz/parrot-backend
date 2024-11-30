import os
import json
from flask import Flask, request, jsonify, url_for
import replicate
import numpy as np
import pandas as pd
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all domains and routes
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Global variables
keywords = []
emoji_embeddings = {}
keyword_embeddings = {}

# Load the list of keywords from Excel
def load_keywords_from_excel(excel_file='key_words_vocab.xlsx'):
    global keywords
    try:
        df = pd.read_excel(excel_file, engine='openpyxl')
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return []

    if 'pleasure' not in df.columns:
        print("Column 'pleasure' not found.")
        return []

    keywords = df['pleasure'].dropna().tolist()
    keywords = [keyword.strip() for keyword in keywords if isinstance(keyword, str)]

# Load emoji embeddings from a JSON file
def load_emoji_embeddings(json_file='emoji_embeddings.json'):
    global emoji_embeddings
    try:
        with open(json_file, 'r') as file:
            emoji_embeddings = json.load(file)
    except Exception as e:
        print(f"Error loading emoji embeddings from JSON file: {e}")

# Load keyword embeddings from the JSON file
def load_keyword_embeddings(json_file='keyword_embeddings.json'):
    global keyword_embeddings
    try:
        with open(json_file, 'r') as file:
            keyword_embeddings = json.load(file)
    except Exception as e:
        print(f"Error loading keyword embeddings from JSON file: {e}")

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.route('/api/pythonEmojiMatcher', methods=['POST'])
def python_emoji_matcher():
    # Check for uploaded image
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    
    # Save the image in static/image directory
    image_path = save_image(image_file)

    # Get the public URI for the saved image
    image_uri = url_for('static', filename=f'image/{image_file.filename}', _external=True)

    # Get the image embedding using Replicate's CLIP model
    image_embedding = get_image_embedding(image_uri)  # Pass the public URI

    # Find the best matching word from the 2800 keywords list
    word = find_best_matching_word(image_embedding)

    # Get the embedding of the word from the loaded keyword embeddings
    word_embedding = get_keyword_embedding(word)

    # Find the closest emoji based on word embedding
    best_emoji = find_closest_emoji(word_embedding)

    # Delete the image after processing
    os.remove(image_path)

    return jsonify({
        'bestEmoji': best_emoji,
        'imageUri': image_uri  # Return the URI for the image
    })

def save_image(image_file):
    """Save the uploaded image to a temporary directory and return the file path."""
    image_path = os.path.join('static', 'image', image_file.filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)  # Ensure the directory exists
    image_file.save(image_path)
    return image_path

def get_image_embedding(image_uri):
    """Get the image embedding using Replicate's CLIP model."""
    keywords_string = " | ".join(keywords)  # Assuming 'keywords' is globally loaded

    input_data = {
        "input": {
            "image": image_uri,  # Pass the public URI of the image
            "text": keywords_string  # Use the concatenated keyword string
        }
    }

    output = replicate.run(
        "cjwbw/clip-vit-large-patch14:566ab1f111e526640c5154e712d4d54961414278f89d36590f1425badc763ecb", 
        input=input_data
    )
    return np.array(output)

def find_best_matching_word(image_embedding):
    """Find the best matching word from the keywords based on the image embedding."""
    global keywords
    best_index = np.argmax(image_embedding)
    best_word = keywords[best_index]
    return best_word

def get_keyword_embedding(word):
    """Get the embedding for a given keyword."""
    try:
        return np.array(keyword_embeddings.get(word, np.zeros(300)))
    except Exception as e:
        print(f"Error fetching embedding for word '{word}': {e}")
        return np.zeros(300)

def find_closest_emoji(word_embedding):
    """Find the closest emoji for the given word embedding."""
    best_emoji = ''
    best_score = -float('inf')

    for emoji, emoji_embedding in emoji_embeddings.items():
        score = cosine_similarity(word_embedding, emoji_embedding)
        if score > best_score:
            best_score = score
            best_emoji = emoji

    return best_emoji

if __name__ == '__main__':
    # Load data at startup
    load_keywords_from_excel()
    load_emoji_embeddings()  # Load emoji embeddings from JSON
    load_keyword_embeddings()  # Load keyword embeddings from JSON

    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=3000)
