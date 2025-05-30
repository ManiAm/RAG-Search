
import requests
import sys
import hashlib

from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

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

        self.embed_model_info = remote_embed.get_vector_sizes()


    def check_qdrant_health(self, url):

        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Qdrant health check failed: {e}")
            return False


    def get_collection_name(self, collection_name):

        return collection_name.replace("/", "_")


    def list_models(self):

        remote_embed = RemoteEmbedding(endpoint=self.embedding_url)
        model_list = remote_embed.list_models()
        return model_list


    def get_store(self, embed_model, collection_name, create_collection=True):

        vector_size = self.embed_model_info.get(embed_model, None)
        if not vector_size:
            return False, f"Cannot get vector size of embed_model '{embed_model}'"

        collection_name = self.get_collection_name(collection_name)

        if not self.qdrant_client.collection_exists(collection_name):

            if create_collection:

                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )

            else:

                return False, f"collection '{collection_name}' does not exist."

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
