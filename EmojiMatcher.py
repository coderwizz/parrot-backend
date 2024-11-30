def get_image_embedding(image_uri, keywords):
    """Get the image embedding using Replicate's CLIP model."""
    # Randomly select 200 keywords from the list
    selected_keywords = np.random.choice(keywords, size=200, replace=False)

    # Join the selected keywords into a single string
    keywords_string = " | ".join(selected_keywords)  # Use the random 200 keywords
    
    input_data = {
        "text": keywords_string,  # Use the concatenated keyword string
        "image": image_uri  # Pass the public URI of the image
    }

    try:
        # Request the image embedding
        output = replicate.run(
            "cjwbw/clip-vit-large-patch14:566ab1f111e526640c5154e712d4d54961414278f89d36590f1425badc763ecb", 
            input=input_data
        )

        # Poll for completion and return once done
        while output.get("status") != "succeeded":
            time.sleep(5)  # Wait for 5 seconds before checking again
            output = replicate.get(output["id"])

        return np.array(output["output"])
    
    except Exception as e:
        print(f"Error getting image embedding from Replicate: {e}")
        return np.zeros(512)  # Return a zero vector if error occurs
