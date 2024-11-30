import os
import json
import requests
from flask import Flask, request, jsonify
import replicate
import numpy as np
import pandas as pd
from flask_cors import CORS
import tempfile
import time

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all domains and routes
CORS(app, supports_credentials=True)

# Load the Replicate API token from the environment
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')

# Initialize Replicate client with the API token
replicate.Client(api_token=REPLICATE_API_TOKEN)

# Imgur API Client ID (use your own generated client ID)
IMGUR_CLIENT_ID = os.environ.get('IMGUR_CLIENT_ID')

# Load the list of keywords from Excel
def load_keywords_from_excel(excel_file='key_words_vocab.xlsx'):
    try:
        df = pd.read_excel(excel_file, engine='openpyxl')
    except Exception as e:
        print(f"Error loading keywords from Excel: {e}")
        return []

    if 'pleasure' not in df.columns:
        print("No 'pleasure' column found in the Excel file")
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
        print(f"Error loading emoji embeddings: {e}")
        emoji_embeddings = {}
    return emoji_embeddings

# Load keyword embeddings from the JSON file
def load_keyword_embeddings(json_file='keyword_embeddings.json'):
    try:
        with open(json_file, 'r') as file:
            keyword_embeddings = json.load(file)
    except Exception as e:
        print(f"Error loading keyword embeddings: {e}")
        keyword_embeddings = {}
    return keyword_embeddings

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Prevent division by zero
    if norm1 == 0 or norm2 == 0:
        return 0  # or you can return np.nan, depending on your needs

    return np.dot(vec1, vec2) / (norm1 * norm2)

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
    
    # Upload the image to Imgur and get the public URL
    image_uri = upload_image_to_imgur(image_path)

    if not image_uri:
        return jsonify({'error': 'Failed to upload image to Imgur'}), 400

    # Load necessary data
    keywords = load_keywords_from_excel()  # Load the keywords from Excel
    keywords = np.random.choice(keywords, size=2803, replace=False) # choose a random batch
    emoji_embeddings = load_emoji_embeddings()  # Load emoji embeddings from JSON
    keyword_embeddings = load_keyword_embeddings()  # Load keyword embeddings from JSON

    # Get the image embedding using Replicate and keywords
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
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    image_path = os.path.join(temp_dir, image_file.filename)

    os.makedirs(os.path.dirname(image_path), exist_ok=True)  # Ensure the directory exists
    image_file.save(image_path)
    return image_path

def upload_image_to_imgur(image_path):
    """Upload the image to Imgur and convert the link to use Rimgo for direct access."""
    url = "https://api.imgur.com/3/image"
    headers = {
        'Authorization': f'Client-ID {IMGUR_CLIENT_ID}'
    }

    try:
        with open(image_path, 'rb') as image_file:
            # Upload the image file
            files = {'image': image_file}
            response = requests.post(url, headers=headers, files=files)

        # Parse the response
        response_data = response.json()

        if response.status_code == 200:
            # Extract the Imgur ID and create the Rimgo link
            imgur_link = response_data['data']['link']
            imgur_id = imgur_link.split('/')[-1]  # Get the image ID
            rimgo_link = f"https://rimgo.pussthecat.org/{imgur_id}"
            return rimgo_link
        else:
            error_message = response_data.get('data', {}).get('error', 'Unknown error')
            print(f"Imgur upload failed: {error_message}")
            return None
    except Exception as e:
        print(f"Error uploading image to Imgur: {e}")
        return None
        
def get_image_embedding(image_uri, keywords):
    """Get the image embedding using Replicate's CLIP model."""
    # Randomly select 200 keywords from the list
    selected_keywords = np.random.choice(keywords, size=200, replace=False)
    
    keywords_string = " | ".join(selected_keywords)  # Concatenate keywords string
    
    input_data = {
        "text": keywords_string,  # Use the concatenated keywords
        "image": image_uri  # Pass the public URI of the image
    }

    try:
        # Request the image embedding
        prediction = replicate.run(
            "cjwbw/clip-vit-large-patch14:566ab1f111e526640c5154e712d4d54961414278f89d36590f1425badc763ecb", 
            input=input_data
        )

        # Print the prediction to the console/log
        print("Prediction from Replicate:", prediction)
        
        # Check if the prediction contains valid output
        if isinstance(prediction, list) and len(prediction) > 0:
            return np.array(prediction)  
        elif isinstance(prediction, dict):
            return np.array(prediction.get('output', np.zeros(512)))  # Default to zero vector if no output key

    except Exception as e:
        print(f"Error getting image embedding from Replicate: {e}")
        return np.zeros(512)  # Return a zero vector in case of error     

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
        print(f"Error getting keyword embedding: {e}")
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
