
import requests
import sys
import hashlib

from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from remote_embedding import RemoteEmbedding


class Qdrant_DB():

    def __init__(self, qdrant_url, embedding_url):

        self.qdrant_url = qdrant_url
        self.embedding_url = embedding_url

        self.store_dict = {}

        if not self.check_qdrant_health(qdrant_url):
            print("Qdrant host is not reachable")
            sys.exit(1)

        self.qdrant_client = QdrantClient(url=qdrant_url)

        self.init()


    def check_qdrant_health(self, url):

        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Qdrant health check failed: {e}")
            return False


    def get_collection_name(self, embed_model):

        return embed_model.replace("/", "_")


    def get_model_store(self, embed_model):

        model_store_dict = self.store_dict.get(embed_model, None)
        if not model_store_dict:
            return False, f"Embedding model '{embed_model}' not loaded."

        return True, model_store_dict


    def list_models(self):

        remote_embed = RemoteEmbedding(endpoint=self.embedding_url)
        model_list = remote_embed.list_models()
        return model_list


    def init(self):

        remote_embed = RemoteEmbedding(endpoint=self.embedding_url)

        if not remote_embed.check_health():
            sys.exit(1)

        model_size = remote_embed.get_vector_sizes()

        for model_name, vector_size in model_size.items():

            print(f"Initializing embedding model '{model_name}'...")

            collection_name = self.get_collection_name(model_name)

            if not self.qdrant_client.collection_exists(collection_name):
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )

            embedding = RemoteEmbedding(endpoint=f"{self.embedding_url}/embed", model=model_name)

            store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
                embedding=embedding,
            )

            self.store_dict[model_name] = {
                "collection_name": collection_name,
                "qdrant_client": self.qdrant_client,
                "qdrant_store": store
            }


    def add_documents(self, embed_model, chunks, batch_size=100):

        status, output = self.get_model_store(embed_model)
        if not status:
            return False, output

        model_store_dict = output

        collection_name = model_store_dict.get("collection_name")
        qdrant_client = model_store_dict.get("qdrant_client")
        qdrant_store = model_store_dict.get("qdrant_store")

        #########

        valid_docs = []
        ids_to_add = []

        for doc in chunks:

            if not isinstance(doc, Document):
                return False, f"Malformed entry: {str(doc)}"

            content = doc.page_content
            point_id = hashlib.md5(content.encode()).hexdigest()

            # Check if point_id already exists
            existing = qdrant_client.retrieve(collection_name=collection_name, ids=[point_id])

            if existing:
                print(f"Skipped duplicate: '{content[:60]}'")
                continue

            valid_docs.append(doc)
            ids_to_add.append(point_id)

        try:
            for i in range(0, len(valid_docs), batch_size):
                batch_docs = valid_docs[i:i + batch_size]
                batch_ids = ids_to_add[i:i + batch_size]
                qdrant_store.add_documents(batch_docs, ids=batch_ids)
        except Exception as e:
            return False, str(e)

        return True, None


    def get_retriever(self, embed_model, search_type="similarity", k=6):
        """
        search_type defines the type of search that the Retriever should perform:

            "similarity"                 : Returns the top-k documents most similar to the query.
            "mmr"                        : Returns relevant and diverse documents to reduce redundancy.
            "similarity_score_threshold" : Returns only documents with a similarity score above a given threshold.
        """

        status, output = self.get_model_store(embed_model)
        if not status:
            return False, output

        model_store_dict = output

        qdrant_store = model_store_dict.get("qdrant_store")

        retriever = qdrant_store.as_retriever(search_type=search_type, k=k)

        return True, retriever
