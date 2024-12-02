
# Pinecone AI Search Toolkit

This repository demonstrates various implementations of advanced search and retrieval techniques using Pinecone, OpenAI models, and diverse datasets.

---
# Table of Contents

- [Semantic Search](#semantic-search)
- [Simple Retrieval-Augmented Generation (RAG)](#simple-retrieval-augmented-generation-rag)
- [Recommender Systems](#recommender-systems)
- [Hybrid Search](#hybrid-search)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

---
Below is an overview of each module and its functionality:

## Semantic Search

This module performs **semantic search** on the [Quora dataset](https://quoradata.quora.com/) by embedding textual data and leveraging Pinecone for efficient similarity matching. The embeddings are generated using the `SentenceTransformer` model, and the Pinecone index is configured with cosine similarity for search.

### Key Features:
- Embedding textual data using **SentenceTransformer**:
  ```python
  model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
  ```
- Creating and managing a Pinecone index:
  ```python
  pinecone.create_index(
      name=INDEX_NAME, 
      dimension=model.get_sentence_embedding_dimension(), 
      metric='cosine',
      spec=ServerlessSpec(cloud='aws', region='us-west-2')
  )
  ```
- Efficient search and retrieval of semantically similar content.

---

## Simple Retrieval-Augmented Generation (RAG)

This module demonstrates a simple **RAG pipeline** using Wikipedia data. It embeds text data, searches for relevant articles, and generates content by prompting the GPT-3.5-turbo-instruct model. The process involves contextual retrieval and generation.

### Workflow:
1. Embed articles using OpenAI’s `text-embedding-ada-002`:
   ```python
   def get_embeddings(articles, model="text-embedding-ada-002"):
       return openai_client.embeddings.create(input=articles, model=model)
   ```
2. Retrieve relevant content from Pinecone:
   ```python
   res = index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)
   contexts = [x['metadata']['text'] for x in res['matches']]
   ```
3. Construct a context-driven prompt and generate the article:
   ```python
   prompt = (
       "Answer the question based on the context below.\n\n" +
       "Context:\n" + "\n\n---\n\n".join(contexts) + 
       f"\n\nQuestion: {query}\nAnswer:"
   )
   res = openai_client.completions.create(
       model="gpt-3.5-turbo-instruct", 
       prompt=prompt,
       temperature=0, max_tokens=636
   )
   ```

---

## Recommender Systems

This module implements a **content-based recommendation system** using a dataset of news article titles. The titles are embedded and indexed, and relevant titles are suggested based on user input (e.g., a keyword or name).

### Key Steps:
- Create and initialize a Pinecone index for storing embeddings:
  ```python
  pinecone.create_index(
      name=INDEX_NAME, 
      dimension=1536, 
      metric='cosine', 
      spec=ServerlessSpec(cloud='aws', region='us-west-2')
  )
  ```
- Embed article titles:
  ```python
  def embed(embeddings, title, prepped, embed_num):
      for embedding in embeddings.data:
          prepped.append({
              'id': str(embed_num), 
              'values': embedding.embedding, 
              'metadata': {'title': title}
          })
  ```
- Query the index for recommendations:
  ```python
  reco = get_recommendations(articles_index, 'obama', top_k=100)
  for r in reco.matches:
      print(f'{r.score} : {r.metadata["title"]}')
  ```

---

## Hybrid Search

This module combines **dense vector search** and **sparse search** to perform a hybrid search that retrieves products based on their descriptions while displaying their images. The dataset used is ["ashraq/fashion-product-images-small"](https://huggingface.co/datasets/ashraq/fashion-product-images-small).

### Key Features:
- **Dense Vector Creation**: Embeddings are generated using the `CLIP` model:
  ```python
  model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)
  dense_vec = model.encode([metadata['productDisplayName'][0]])
  ```

- **Hybrid Search Scaling**: Combines sparse (BM25) and dense embeddings using a scaling function. This allows fine-tuning the balance between semantic similarity and keyword-based search by adjusting the `alpha` parameter:
  ```python
  def hybrid_scale(dense, sparse, alpha: float):
      """Hybrid vector scaling using a convex combination"""
      hsparse = {'indices': sparse['indices'], 'values': [v * (1 - alpha) for v in sparse['values']]}
      hdense = [v * alpha for v in dense]
      return hdense, hsparse
  ```

- **Product Search**: Users can search for products like "dark blue French connection jeans for men," and the system retrieves relevant items along with their images. Results are ranked using a hybrid scoring mechanism.

### Visualization:
The retrieved products are displayed with their corresponding images to enhance user experience:
```python
imgs = [images[int(r["id"])] for r in result["matches"]]
display_result(imgs)
```

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/alirezafarzipour/Pinecone-AI-Search-Toolkit.git
   cd Pinecone-AI-Search-Toolkit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys:
   - Add your **Pinecone API Key** and **OpenAI API Key** to the environment variables or a `.env` file.

---

## Usage

Each module is provided as a Jupyter Notebook (`.ipynb`). You can run these notebooks locally using **Jupyter Notebook** or in the cloud using **Google Colab**:

1. **Jupyter Notebook**:
   - Install Jupyter Notebook if not already installed:
     ```bash
     pip install notebook
     ```
   - Navigate to the directory of the desired module and run:
     ```bash
     jupyter notebook
     ```
   - Open the corresponding notebook file (e.g., `semantic_search.ipynb`) and execute the cells step by step.

2. **Google Colab**:
   - Upload the notebook to Google Colab:
     - Go to [Google Colab](https://colab.research.google.com/).
     - Click on `File > Upload Notebook` and upload the desired `.ipynb` file.
   - Install required dependencies in the first cell and proceed with the execution.

This structure allows for a user-friendly experience while experimenting with the modules.

---

## Contributing

Contributions are welcome! Please feel free to submit a pull request or create an issue for feedback or feature requests.



