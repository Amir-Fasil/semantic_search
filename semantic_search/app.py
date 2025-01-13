from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from pyngrok import ngrok

app = Flask(__name__)

# Dummy products DataFrame for the example
products = pd.read_csv("dataset/products.csv") ### correct this so you don't need to specify specific geenral path

# Create embeddings for all product descriptions
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
words = products.apply(lambda row: f'{row["name"]} is an {row["main_category"]} product under the sub-category {row["sub_category"]}. it has a rating of {row["ratings"]} star, {row["no_of_ratings"]} reviews, and it is priced between {row["discount_price"]} and {row["actual_price"]}', axis=1)
numpy_data = words.to_numpy()
data_embeddings = model.encode(numpy_data, convert_to_tensor=True)

# Function to get top 3 results from semantic search
def get_top_result(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, data_embeddings)
    score_results = cos_scores.argsort(descending=True)
    top_results = []
    for idx in score_results[0][:3]:
        top_results.append(numpy_data[idx])
    return top_results

# Function to filter product names based on category
def get_names(row, target):
    if row['sub_category'] == target:
        return row['name']

def get_category(target):
    new_df = products.apply(lambda row: get_names(row, target), axis=1)
    new_df = new_df.dropna()
    return new_df.to_numpy()

@app.route('/', methods=['GET', 'POST'])
def index():
    categories = products['sub_category'].unique()  # Get unique categories
    results = []

    if request.method == 'POST':
        query = request.form['query']
        category = request.form['category']

        # Get top results based on semantic similarity
        results = get_top_result(query)

        # Filter by the selected category
        if category:
            filtered_names = get_category(category)
            results = [result for result in results if any(name in result for name in filtered_names)]

    return render_template('index.html', results=results, categories=categories)

# Run Flask app using ngrok
if __name__ == '__main__':
    port = 5000

    # Open a ngrok tunnel to the Flask app
    public_url = ngrok.connect(port)
    print(f" * ngrok tunnel \"{public_url}\" -> http://127.0.0.1:{port}")

    app.run(port=port)
