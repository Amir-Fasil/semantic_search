from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Load product data
products = pd.read_csv("dataset/products.csv")  # Use relative paths here, ensuring dataset is in the correct location

# Create embeddings for all product descriptions
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
words = products.apply(lambda row: f'{row["name"]} is an {row["main_category"]} product under the sub-category {row["sub_category"]}. it has a rating of {row["ratings"]} star, {row["no_of_ratings"]} reviews, and it is priced between {row["discount_price"]} and {row["actual_price"]}', axis=1)
numpy_data = words.to_numpy()

# Function to precompute and save embeddings
def precompute_cache(data):
    """This function precomputes the data embedding and saves it."""
    pre_encode_data = model.encode(data, convert_to_tensor=True)
    np.save("cached_embeddings", pre_encode_data)

# Load precomputed embeddings from cache
def load_cache():
    """This function loads the precomputed embedded data."""
    return np.load(os.path.join(BASE_DIR, "dataset/cached_embeddings.npy"))

data_embeddings = load_cache()

# Function to get top 3 results from semantic search
def get_top_result(query):
    """This function returns the top three results of the semantic search."""
    query_embedding = model.encode(query)
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    data_norms = data_embeddings / np.linalg.norm(data_embeddings, axis=1, keepdims=True)
    cos_scores = np.dot(data_norms, query_norm)
    score_results = np.argsort(cos_scores)[::-1][:3]
    top_results = [numpy_data[idx] for idx in score_results]
    return top_results

# Function to filter product names based on category
def get_names(row, target):
    """Returns product names matching the category."""
    if row['sub_category'] == target:
        return row['name']

def get_category(target):
    """Filters products by category."""
    new_df = products[products['sub_category'] == target]  # Direct filtering by category
    return new_df['name'].to_numpy()  # Return only product names


@app.route('/', methods=['GET', 'POST'])
def index():
    categories = products['sub_category'].unique()
    results = []

    # Check if the form was submitted with a query
    if request.method == 'POST':
        query = request.form.get('query')
        category = request.form.get('category')
        print(f"Query: {query}, Category: {category}")  # Debugging log

        # Perform the semantic search if a query is provided
        if query:
            results = get_top_result(query)
            print(f"Top Results: {results}")  # Debugging log

        # If a category is selected, filter products by category
        if category:
            filtered_names = get_category(category)
            print(f"Filtered Names: {filtered_names}")  # Debugging log
            
            # If semantic search results exist, filter them by the selected category
            if filtered_names.size > 0:
                # Check if each result contains any of the filtered product names from the selected category
                results = [result for result in results if any(name in result for name in filtered_names)]
                print(f"Category Filtered Results: {results}")  # Debugging log

            # If no query was entered, show all products from the selected category
            elif not query:
                results = filtered_names
                print(f"Category Filtered Results (No Query): {results}")  # Debugging log

    return render_template('index.html', results=results, categories=categories)

# Run Flask app
if __name__ == '__main__':
    port = 5000
    app.run(host='127.0.0.1', port=port)  # Default behavior
