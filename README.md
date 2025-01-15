# Product Semantic Search with Category Filter

This project implements a semantic search engine that allows users to search for product names based on a query. 
The system uses Sentence-BERT embeddings for generating semantic representations of product descriptions and offers a filtering option to narrow down the search based on product categories.

# Features:
Semantic Search: Users can search for products based on a textual query. 
The system returns the top 3 most semantically similar products.

Category Filter: Users can filter search results by selecting a product category from a dropdown list,
showing all the product names in that category.

# Installation and Running the Code Locally
1. Clone the repository
2. Create and activate a virtual environment
3. install dependencies
     pip install flask sentence-transformers pandas numpy
4. Run the Flask application
     python app.py

5. Example Searches
        -> Air conditioners that have 5.0 rating
        -> backs packs for school
        -> Air conditioners that has lower price

