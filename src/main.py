from importlib.resources import path
import json
import time

from bs4 import BeautifulSoup

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from sentence_transformers import SentenceTransformer

CLIENT = "weka_con"
INDEX_NAME = "articles"
INDEX_FILE = f"data/{CLIENT}/index.json"
DATA_FILE = f"data/{CLIENT}/articles.json"
SEARCH_SIZE = 10

class TextSimilaritySearch:
    def __init__(self, client: Elasticsearch, embedding_model: SentenceTransformer):
        self.client = client
        self.index_name = INDEX_NAME
        self.index_file = INDEX_FILE
        self.data_file = DATA_FILE
        self.search_size = SEARCH_SIZE
        self.embedding_model = embedding_model
        self.docs = self.load_documents()

    def load_documents(self):
        with open(self.data_file) as f:
            return json.load(f)
    
    def create_index(self):
        with open(self.index_file) as f:
            self.client.indices.delete(index=self.index_name, ignore=[404])
            self.client.indices.create(index=self.index_name, body=f.read().strip())
    
    def index_data(self):
        
        print(f"Creating {self.index_name} index ...")
        self.create_index()

        print(f"Indexing {len(self.docs)} documents ...")
        self.index_documents()

        print("Refreshing index ...")
        client.indices.refresh(index=self.index_name)
        
        print("Indexing finished")
    
    @staticmethod
    def clean_content(text:str):
        if text:
            return BeautifulSoup(text, "html.parser").text
        else:
            return "empty"

    @staticmethod
    def group_by_id(ids, vectors):
        kw_vector_list = []
        grouped_kw_vectors = []
        previous_id = ids[0]
        last_kw_vector = []
        for id, vector in zip(ids, vectors):
            if id == previous_id:
                kw_vector_list.append({"vector": vector})
                last_kw_vector = kw_vector_list
            else:
                grouped_kw_vectors.append(kw_vector_list)
                kw_vector_list = [{"vector": vector}]
                previous_id = id
        grouped_kw_vectors.append(last_kw_vector)
        return grouped_kw_vectors
    
    def index_documents(self):
        titles = [doc["title"] for doc in self.docs]
        descriptions = [doc["description"] for doc in self.docs]
        contents = [self.clean_content(doc["content"]) for doc in self.docs]
        keywords_with_id = [(doc["id"], keyword) for doc in self.docs for keyword in doc["keywords"].split(",") if keyword]
        topics_with_id = [(doc["id"], topic) for doc in self.docs for topic in doc["topic"].split(",") if topic]

        ids_keywords = [id for id,keyword in keywords_with_id]
        keywords = [keyword for id,keyword in keywords_with_id]

        ids_topics = [id for id,topic in topics_with_id]
        topics = [topic for id,topic in topics_with_id]

        print("Computing BERT embeddings ...")
        title_vectors = self.embed_text(titles)
        descriptions_vectors = self.embed_text(descriptions)
        contents_vectors = self.embed_text(contents)
        keywords_vectors_ = self.embed_text(keywords)
        topics_vectors_ = self.embed_text(topics)

        keywords_vectors = self.group_by_id(ids_keywords, keywords_vectors_)
        topics_vectors = self.group_by_id(ids_topics, topics_vectors_)

        requests = []
        for i, doc in enumerate(self.docs):
            request = doc
            request["_op_type"] = "index"
            request["_index"] = self.index_name
            request["content"] = self.clean_content(doc["content"]) 
            request["title_vector"] = title_vectors[i]
            request["description_vector"] = descriptions_vectors[i]
            request["content_vector"] = contents_vectors[i]
            request["keywords_vector"] = keywords_vectors[i]
            request["topic_vector"] = topics_vectors[i]
            requests.append(request)
        
        print("Indexing documents ...")
        bulk(self.client, requests)


    def run_query_loop(self):
        while True:
            try:
                self.handle_query()
            except KeyboardInterrupt:
                return


    def handle_query(self):
        query = input("Enter query: ")

        selection_list = query.split(",")

        embedding_start = time.time()
        query_vector_list = self.embed_text(selection_list)
        embedding_time = time.time() - embedding_start

        total_search_time = 0 

        for query_vector in query_vector_list:
            my_query = {
                "bool":{
                    "should": [
                        {"nested": {
                            "path": "keywords_vector",
                            "score_mode": "max",
                            "query":{
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source":   """  
                                                        doc['keywords_vector.vector'].size() == 0 ? 0: 1.0 + cosineSimilarity(params.query_vector, 'keywords_vector.vector')
                                                    """,
                                        "params": {
                                            "query_vector": query_vector, 
                                            },
                                    },
                                }
                            }
                        }},
                        {"nested": {
                            "path": "topic_vector",
                            "score_mode": "max",
                            "query":{
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source":   """
                                                    doc['topic_vector.vector'].size() == 0 ? 0: 1.0 +  cosineSimilarity(params.query_vector, 'topic_vector.vector')
                                                    """,
                                        "params": {
                                            "query_vector": query_vector, 
                                            },
                                    },
                                }
                            }
                        }}
                        ,{"script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source":   """  
                                                cosineSimilarity(params.query_vector, 'title_vector') + 
                                                cosineSimilarity(params.query_vector, 'description_vector') +
                                                cosineSimilarity(params.query_vector, 'content_vector') +
                                                3.0
                                            """,
                                "params": {
                                    "query_vector": query_vector, 
                                    }
                            }                    
                        }}
                    ] 
                }           
            }

            search_start = time.time() 
            response = self.client.search(
                index=self.index_name,
                body={
                    "size": self.search_size,
                    "query": my_query,
                    "_source": {"includes": ["title", "description", "content", "link", "keywords", "topic"]},
                },
            )
            search_time = time.time() - search_start

            total_search_time += search_time

            # print()
            # print("{} total hits.".format(response["hits"]["total"]["value"]))
            # print("search time: {:.2f} ms".format(search_time * 1000))
            # for hit in response["hits"]["hits"]:
            #     print(f'score: {hit["_score"] - 5.0}') # Substracting the added number above.
            #     for field, text in hit["_source"].items():
            #         print(f"{field}: {text[:150]}")
            #     print()

        print(f"Search time: {1000*total_search_time:.2f} ms")
        print(f"Embedding time: {1000*embedding_time:.2f} ms")
        print(f"Total time: {1000*(total_search_time + embedding_time):.2f} ms")

            

    def embed_text(self, text):
        vectors = self.embedding_model.encode(text, show_progress_bar=True)
        return [vector.tolist() for vector in vectors]


if __name__ == "__main__":
    client = Elasticsearch()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    text_similarity = TextSimilaritySearch(client, embedding_model)
    text_similarity.index_data()
    text_similarity.run_query_loop()


"""
Anrufer identifizieren, Caller ID, Werbeanrufe, Nummer blockieren, Spam, iPhone, Android, angebot, handy, laptop, tablet, smartwatch, over-ear-kopfhörer, bestenliste, bluetooth, anc, kabellos, top 10, kaufberatung
kaufempfehlung, Gaming, home connect, Heimvernetzung, Lautsprecher
Sonos, Dolby Atmos, smartwatch, bestenliste, fitnessuhr
tracker, test, vergleich, JBL, Sound
Kopfhörer, Lautsprecher, Boombox, CES 2022, Las Vegas
In-Ear-Kopfhörer, Over-Ear, Sony Xperia Pro-I, Smartphone, Test
Mobilfunk-Netzbetreiber, O2, Telekom, Vodafone, Vergleich
Tarife, Technik, Service, samsung galaxy s21 fe, realme
realme gt 2, pro, smartphone, mittelklasse, release
specs, preis, günstiges smartphonerealme, realme gt 2, pro
smartphone, mittelklasse, release, specs, preis
günstiges smartphone, acer, chrombeook, Notebook, CES 2022
Release, Specs, Preis, Windows, MediaMarket
"""