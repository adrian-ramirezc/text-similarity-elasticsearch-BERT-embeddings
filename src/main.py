import json
import time

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from sentence_transformers import SentenceTransformer


##### INDEXING #####


def index_data():
    print("Creating the 'articles' index.")
    client.indices.delete(index=INDEX_NAME, ignore=[404])

    with open(INDEX_FILE) as index_file:
        source = index_file.read().strip()
        client.indices.create(index=INDEX_NAME, body=source)

    with open(DATA_FILE) as data_file:
        docs = json.load(data_file)
        index_batch(docs)
        print("Indexed {} documents.".format(len(docs)))

    client.indices.refresh(index=INDEX_NAME)
    print("Done indexing.")


def index_batch(docs):
    titles = [doc["title"] for doc in docs]
    title_vectors = embed_text(titles)

    requests = []
    for i, doc in enumerate(docs):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = INDEX_NAME
        request["title_vector"] = title_vectors[i]
        requests.append(request)
    bulk(client, requests)


##### SEARCHING #####


def run_query_loop():
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            return


def handle_query():
    query = input("Enter query: ")

    embedding_start = time.time()
    query_vector = embed_text([query])[0]
    embedding_time = time.time() - embedding_start

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['title_vector']) + 1.0",
                "params": {"query_vector": query_vector},
            },
        }
    }

    search_start = time.time()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["title"]},
        },
    )
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    print("search time: {:.2f} ms".format(search_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"])
        print()


##### EMBEDDING #####


def embed_text(text):
    vectors = embeddings_model.encode(text)
    return [vector.tolist() for vector in vectors]


##### MAIN SCRIPT #####

if __name__ == "__main__":
    INDEX_NAME = "articles"
    INDEX_FILE = "data/articles/index.json"
    DATA_FILE = "data/articles/articles.json"
    SEARCH_SIZE = 5

    client = Elasticsearch()
    embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")

    index_data()
    run_query_loop()
    print("Done.")
