import requests

def query_ollama(model_name, prompt):
    """
    Function to interact with an Ollama model.
    """
    url = "http://localhost:11434/api/chat"  # Default endpoint for Ollama
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP issues
        data = response.json()
        return data.get("response", "No response from model.")
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

# Example usage
if __name__ == "__main__":
    model_name = "mistral"  # Cambia el modelo a "mistral"
    prompt = "What are the main benefits of using Mistral for LLMs?"
    
    print("==== Querying Ollama ====")
    response = query_ollama(model_name, prompt)
    print("Model Response:", response)