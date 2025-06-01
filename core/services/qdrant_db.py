
import sys
import requests
import hashlib

from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, FilterSelector

from services.remote_embedding import RemoteEmbedding


class Qdrant_DB():

    def __init__(self, qdrant_url, embedding_url):

        self.qdrant_url = qdrant_url
        self.embedding_url = embedding_url

        if not self.check_qdrant_health(qdrant_url):
            print("Qdrant host is not reachable")
            sys.exit(1)

        self.qdrant_client = QdrantClient(url=qdrant_url)

        remote_embed = RemoteEmbedding(endpoint=self.embedding_url)
        if not remote_embed.check_health():
            print("Remote embedding server is not reachable")
            sys.exit(1)
        print("Remote embedding server is reachable.")

        self.model_list = remote_embed.list_models()
        self.vector_size_map = remote_embed.get_vector_sizes()
        self.max_tokens_map = remote_embed.get_max_tokens()


    def check_qdrant_health(self, url):

        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Qdrant health check failed: {e}")
            return False


    def list_models(self):

        return self.model_list


    def get_model_vector_size(self):

        return self.vector_size_map


    def get_model_max_token(self):

        return self.max_tokens_map


    def list_collections(self):

        collections = self.qdrant_client.get_collections().collections

        return [
            c.name for c in collections
        ]


    def get_collection_name(self, collection_name):

        return collection_name.replace("/", "_")


    def create_collection(self, embed_model, collection_name):

        vector_size = self.vector_size_map.get(embed_model, None)
        if not vector_size:
            return False, f"Cannot get vector size of embed_model '{embed_model}'"

        collection_name = self.get_collection_name(collection_name)

        if not self.qdrant_client.collection_exists(collection_name):

            try:

                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )

            except Exception as e:
                return False, str(e)

        return True, None


    def get_store(self, embed_model, collection_name, create_collection=True):

        if not self.qdrant_client.collection_exists(collection_name):

            if not create_collection:
                return False, f"collection '{collection_name}' does not exist."

            status, output = self.create_collection(embed_model, collection_name)
            if not status:
                return False, output

        embedding = RemoteEmbedding(endpoint=f"{self.embedding_url}/embed", model=embed_model)

        store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=embedding,
        )

        return True, store


    def add_documents(self, embed_model, collection_name, chunks, batch_size=100):

        status, output = self.get_store(embed_model, collection_name)
        if not status:
            return False, output

        qdrant_store = output

        #########

        valid_docs = []
        ids_to_add = []

        for doc in chunks:

            if not isinstance(doc, Document):
                return False, f"Malformed entry: {str(doc)}"

            content = doc.page_content
            point_id = hashlib.md5(content.encode()).hexdigest()

            # Check if point_id already exists
            try:
                existing = self.qdrant_client.retrieve(collection_name=collection_name, ids=[point_id])
            except Exception as e:
                return False, {str(e)}

            if existing:
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


    def get_retriever(self, embed_model, collection_name, search_type="similarity", k=6):
        """
        search_type defines the type of search that the Retriever should perform:

            "similarity"                 : Returns the top-k documents most similar to the query.
            "mmr"                        : Returns relevant and diverse documents to reduce redundancy.
            "similarity_score_threshold" : Returns only documents with a similarity score above a given threshold.
        """

        status, output = self.get_store(embed_model, collection_name, create_collection=False)
        if not status:
            return False, output

        qdrant_store = output

        retriever = qdrant_store.as_retriever(search_type=search_type, k=k)

        return True, retriever


    def delete_points_by_filter(self, collection_name, filter_dict):

        conditions = [
            FieldCondition(
                key=k,
                match=MatchValue(value=v)
            )
            for k, v in filter_dict.items()
        ]

        deletion_filter = Filter(must=conditions)

        return self.qdrant_client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=deletion_filter)
        )
