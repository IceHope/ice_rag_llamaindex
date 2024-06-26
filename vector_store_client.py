from typing import List

from Tools.scripts.dutree import display
from gradio import Markdown
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from pydantic import PrivateAttr, Field
from llama_index.core import PromptTemplate

from files_load import get_nodes_from_files
from prompt.prompt_utils import get_rag_query_prompt


class VectorStoreClient:
    _vector_store: BasePydanticVectorStore = PrivateAttr()

    _chat_engine: CondenseQuestionChatEngine = PrivateAttr()

    _similarity_top_k: int = Field(2, description="返回相似度最高的前K个文档")

    def _init_chroma_db(self, collection_name: str) -> None:
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore

        chroma_client = chromadb.EphemeralClient()
        chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

        self._vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    def __init__(self, collection_name: str) -> None:
        self._init_chroma_db(collection_name)

    def add_files(self, files: List[str]) -> None:
        # 获取节点
        nodes = get_nodes_from_files(files)
        print("---add_files---len(nodes)= ", len(nodes))
        # 索引  灌库
        vector_store_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=StorageContext.from_defaults(
                vector_store=self._vector_store
            ),
            show_progress=True,
        )
        print("---VectorStoreIndex---end---")

        self._chat_engine = self._get_chat_engine(vector_store_index)

    def _get_chat_engine(self, index: VectorStoreIndex) -> CondenseQuestionChatEngine:
        # 单次查询
        query_engine = RetrieverQueryEngine(
            index.as_retriever(_similarity_top_k=self._similarity_top_k),
        )
        # 定义prompt
        rag_query_prompt = PromptTemplate(get_rag_query_prompt())
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": rag_query_prompt}
        )
        # 对话引擎
        return CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
        )

    def stream_chat(self, query: str):
        response_gen = self._chat_engine.stream_chat(query).response_gen
        return response_gen

    def chat(self, query: str) -> str:
        response = self._chat_engine.chat(query)
        return response
