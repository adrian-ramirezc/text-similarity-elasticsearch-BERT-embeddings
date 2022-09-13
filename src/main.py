import json
import time

from typing import List

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from sentence_transformers import SentenceTransformer


INDEX_NAME = "articles"
INDEX_FILE = "data/articles/index.json"
DATA_FILE = "data/articles/articles.json"
SEARCH_SIZE = 5

class TextSimilaritySearch:
    def __init__(self, client: Elasticsearch, embedding_model: SentenceTransformer):
        self.client = client
        self.index_name = INDEX_NAME
        self.index_file = INDEX_FILE
        self.data_file = DATA_FILE
        self.search_size = SEARCH_SIZE
        self.embedding_model = embedding_model
        self.docs = None


    def index_data(self):
        print(f"Creating {self.index_name} index ...")
        client.indices.delete(index=self.index_name, ignore=[404])

        with open(self.index_file) as f:
            source = f.read().strip()
            client.indices.create(index=self.index_name, body=source)

        with open(self.data_file) as f:
            self.docs = json.load(f)
            self.index_documents()
            print(f"Indexed {len(self.docs)} documents.")

        client.indices.refresh(index=self.index_name)
        print("Done indexing")


    def index_documents(self):
        titles = [doc["title"] for doc in self.docs]
        descriptions = [doc["description"] for doc in self.docs]
        sections = [doc["section"] for doc in self.docs]
        keywords = [doc["keywords"] for doc in self.docs]
        topics = [doc["topic"] for doc in self.docs]

        title_vectors = self.embed_text(titles)
        descriptions_vectors = self.embed_text(descriptions)
        sections_vectors = self.embed_text(sections)
        keywords_vectors = self.embed_text(keywords)
        topics_vectors = self.embed_text(topics)

        requests = []
        for i, doc in enumerate(self.docs):
            request = doc
            request["_op_type"] = "index"
            request["_index"] = self.index_name
            request["title_vector"] = title_vectors[i]
            request["description_vector"] = descriptions_vectors[i]
            request["section_vector"] = sections_vectors[i]
            request["keywords_vector"] = keywords_vectors[i]
            request["topic_vector"] = topics_vectors[i]
            requests.append(request)
        bulk(self.client, requests)


    def run_query_loop(self):
        while True:
            try:
                self.handle_query()
            except KeyboardInterrupt:
                return


    def handle_query(self):
        query = input("Enter query: ")

        embedding_start = time.time()
        query_vector = self.embed_text(query)
        embedding_time = time.time() - embedding_start
    
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source":   """   params.title_w * cosineSimilarity(params.query_vector, doc['title_vector']) 
                                    + params.description_w * cosineSimilarity(params.query_vector, doc['description_vector']) 
                                    + params.keywords_w * cosineSimilarity(params.query_vector, doc['keywords_vector']) 
                                    + params.section_w * cosineSimilarity(params.query_vector, doc['section_vector']) 
                                    + params.topic_w * cosineSimilarity(params.query_vector, doc['topic_vector'])
                                    + 5.0
                                """,
                    "params": {
                        "query_vector": query_vector, 
                        "title_w": 1.0,
                        "description_w": 1.0,
                        "keywords_w": 0.8,
                        "section_w": 0.8,
                        "topic_w": 0.8,
                        },
                },
            }
        }

        search_start = time.time() 
        response = self.client.search(
            index=self.index_name,
            body={
                "size": self.search_size,
                "query": script_query,
                "_source": {"includes": ["title", "section", "keywords", "topic"]},
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

    def embed_text(self, text):
        vectors = self.embedding_model.encode(text)
        return [vector.tolist() for vector in vectors]


if __name__ == "__main__":
    client = Elasticsearch()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    text_similarity = TextSimilaritySearch(client, embedding_model)
    text_similarity.index_data()
    text_similarity.run_query_loop()
