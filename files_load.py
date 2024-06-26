from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter


def get_nodes_from_files(files: List[str]):
    documents = SimpleDirectoryReader(
        input_files=files,
        file_extractor={".pdf": PyMuPDFReader()},
    ).load_data()

    node_parser = SentenceSplitter(chunk_size=400, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    return nodes
