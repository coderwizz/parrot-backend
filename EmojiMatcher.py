import os
import json
from flask import Flask, request, jsonify
import replicate
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Set the static folder for images
app.config['UPLOAD_FOLDER'] = 'static/images'

# Global variables
keywords = []
emoji_embeddings = {}
keyword_embeddings = {}  # This will store the keyword embeddings from the JSON file

# Load the list of keywords from Excel
def load_keywords_from_excel(excel_file='key_words_vocab.xlsx'):
    global keywords  # Use the global variable
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
    global emoji_embeddings  # Access the global emoji_embeddings variable
    try:
        with open(json_file, 'r') as file:
            emoji_embeddings = json.load(file)
    except Exception as e:
        print(f"Error loading emoji embeddings from JSON file: {e}")

# Load keyword embeddings from the JSON file
def load_keyword_embeddings(json_file='keyword_embeddings.json'):
    global keyword_embeddings  # Access the global keyword_embeddings variable
    try:
        with open(json_file, 'r') as file:
            keyword_embeddings = json.load(file)
    except Exception as e:
        print(f"Error loading keyword embeddings from JSON file: {e}")

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.route('/pythonEmojiMatcher', methods=['POST'])
def python_emoji_matcher():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_url = save_image(image_file)  # Save the image and get the URL

    # Get the image embedding using Replicate's CLIP model
    image_embedding = get_image_embedding(image_url)

    # Clean up: Delete the image after it has been processed
    delete_image(image_url)

    # Find the best matching word from the 2800 keywords list
    word = find_best_matching_word(image_embedding)

    # Get the embedding of the word from the loaded keyword embeddings
    word_embedding = get_keyword_embedding(word)

    # Find the closest emoji based on word embedding
    best_emoji = find_closest_emoji(word_embedding)

    return jsonify({'bestEmoji': best_emoji})

def save_image(image_file):
    # Ensure the directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Save the image with a unique filename
    image_filename = f"{str(np.random.randint(1000000))}.jpg"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

    # Save the image to disk
    image_file.save(image_path)

    # Construct the URL for deployment (e.g., on vercel.sample.com)
    image_url = f"https://vercel.sample.com/{app.config['UPLOAD_FOLDER']}/{image_filename}"
    
    return image_url

def delete_image(image_url):
    # Extract the image filename from the URL
    image_filename = image_url.split('/')[-1]
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    
    # Delete the image file from the server
    if os.path.exists(image_path):
        os.remove(image_path)

def get_image_embedding(image_url):
    # Create a string of keywords separated by " | "
    keywords_string = " | ".join(keywords)  # Assuming 'keywords' is globally loaded

    input_data = {
        "input": {
            "image": image_url,
            "text": keywords_string  # Use the concatenated keyword string
        }
    }
    # Get the image embedding via Replicate
    output = replicate.run(
        "cjwbw/clip-vit-large-patch14:566ab1f111e526640c5154e712d4d54961414278f89d36590f1425badc763ecb", 
        input=input_data
    )
    return np.array(output)

def find_best_matching_word(image_embedding):
    global keywords  # Access global keywords list
    
    # Find the index of the highest value in the image_embedding array
    best_index = np.argmax(image_embedding)  # Get index of highest value
    
    # Return the keyword corresponding to that index
    best_word = keywords[best_index]
    
    return best_word

# Update this function to get embeddings from the loaded JSON data
def get_keyword_embedding(word):
    try:
        return np.array(keyword_embeddings.get(word, np.zeros(300)))  # Assuming vector size of 300
    except Exception as e:
        print(f"Error fetching embedding for word '{word}': {e}")
        return np.zeros(300)  # Default zero vector if not found

def find_closest_emoji(word_embedding):
    best_emoji = ''
    best_score = -float('inf')

    # Iterate through the loaded emoji embeddings
    for emoji, emoji_embedding in emoji_embeddings.items():
        score = cosine_similarity(word_embedding, emoji_embedding)
        if score > best_score:
            best_score = score
            best_emoji = emoji

    return best_emoji

if __name__ == '__main__':
    # Load the keywords and emoji embeddings once when the app starts
    load_keywords_from_excel()
    load_emoji_embeddings()  # Load emoji embeddings from JSON
    load_keyword_embeddings()  # Load keyword embeddings from JSON

    app.run(debug=True, host='0.0.0.0', port=3000)
