from dotenv import load_dotenv
from llama_index.core import Settings
import os

load_dotenv()


class RagInit:
    def __init__(self) -> None:
        self._init_dashscope()
        # self._init_OpenAI()

    def _init_OpenAI(self):
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding

        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_BASE_URL")

        Settings.llm = OpenAI(
            api_key=api_key,
            api_base=api_base,
            temperature=0.7,
            model="gpt-3.5-turbo",
        )

        Settings.embed_model = OpenAIEmbedding(
            api_key=api_key,
            api_base=api_base,
            model="text-embedding-3-small",
            dimensions=512,
        )

    def _init_dashscope(self):
        from llama_index.llms.dashscope import DashScope
        from llama_index.embeddings.dashscope import DashScopeEmbedding

        api_key = os.getenv("DASHSCOPE_API_KEY")

        Settings.llm = DashScope(
            api_key=api_key,
            temperature=0.7,
        )

        Settings.embed_model = DashScopeEmbedding(dashscope_api_key=api_key)
