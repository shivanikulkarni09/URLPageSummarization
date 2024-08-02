import requests
from bs4 import BeautifulSoup
from flask import Flask, json,request,jsonify
from threading import Thread
from summarizer import Summarizer
import numpy as np
from scipy.spatial.distance import cosine
from InstructorEmbedding import INSTRUCTOR

app = Flask(__name__)

# Initialize the status and results storage
status = {"status":"not started"}
processed_urls = set()
results = []

# API to take a list of URLs and start the processing in the background
@app.route('/api/start_processing', methods=['POST'])
def get_urls():
    try:
        urls = request.json.get("urls",[]) 
        thread = Thread(target=extract_urls_data,args=([urls]))    
        thread.start()
        return jsonify({"msg":"success"}),200
    except Exception as e:
        return jsonify({"error":str(e)}),500

# API to get the results or the status of the processing
@app.route('/api/results', methods=['GET'])
def get_results():
    try:
        if status["status"] == "completed":
            return jsonify({"results":results,"cosine distance matrix":(status["matrix"])}),200
        else:
            return jsonify({"status":status["status"]}),200
    except Exception as e:
        return jsonify({"error":str(e)}),500

# Report API that returns a paginated list of all URLs 
@app.route('/api/report', methods=['POST'])
def get_report():
    try:
        page = int(request.json.get('page', 1))
        page_size = int(request.json.get('page_size', 10))
        start = (page - 1) * page_size
        end = start + page_size
        return jsonify(results[start:end]),200
    except Exception as e:
        return jsonify({"error":str(e)}),500

# Function to scrape data from a URL and create vector embeddings
def crawl_url_and_vectorize(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    title = soup.find('h1').text if soup.find('h1') else 'No title'
    paragraphs = soup.find_all('p')
    text = " ".join([p.get_text() for p in paragraphs]) 
    
    # Generate embeddings for the extracted text
    embeddings = create_vector_embeddings(text)
    links = [a['href'] for a in soup.find_all('a', href=True)]
    
    # Summarize the page content
    summarizer = Summarizer()
    summary = summarizer(text, min_length=50, max_length=150)
    result = {
        "url": url,
        "page title": title,
        "page summary": summary,
        "links": links
    }  
    return result, embeddings

# Function to process all URLs and store results
def extract_urls_data(urls):
    global status, results, processed_urls
    status["status"] = "In progress"
    try:
        results = []
        vector_embeddings = []
        processed_urls = set()
        for url in urls:
            if url not in processed_urls:
                processed_urls.add(url)
                result,embeddings = crawl_url_and_vectorize(url)
                results.append(result)
                vector_embeddings.append(embeddings)
        
        # Calculate the cosine distance matrix for the embeddings
        matrix = cosine_distance_matrix(vector_embeddings)
        matrix_array = json.dumps(matrix.tolist())
        status["status"] = "completed"
        status["matrix"] = matrix_array

    except Exception as e:
        status["status"] = f"error - {str(e)}"

# Function to generate vector embeddings for a given text using INSTRUCTOR model
def create_vector_embeddings(text):
    model = INSTRUCTOR('hkunlp/instructor-large')
    embeddings = model.encode(text)
    return embeddings

# Function to create a cosine distance matrix from a list of vectors
def cosine_distance_matrix(vectors):
    size = len(vectors)
    matrix = np.zeros((size, size))

    # Compute the cosine distance between each pair of vectors
    for i in range(size):
        for j in range(size):
            matrix[i][j] = cosine(vectors[i], vectors[j])
    return matrix


if __name__ == '__main__':
    app.run('0.0.0.0',debug=True)
