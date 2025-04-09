from glob import glob

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import NotebookLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml


def load_markdown_documents(root_path: str):
    file_paths = glob(f"{root_path}/**/*.md", recursive=True)
    docs = []
    for file_path in file_paths:
        loader = UnstructuredMarkdownLoader(file_path)
        docs.extend(loader.load())
    return docs


def load_ipynb_documents(root_path: str):
    file_paths = glob(f"{root_path}/**/*.ipynb", recursive=True)
    docs = []
    for file_path in file_paths:
        loader = NotebookLoader(file_path)
        docs.extend(loader.load())
    return docs


def load_all_transcripts(root_path: str):
    lecture_files = glob(f"{root_path}/**/_lecture.yml", recursive=True)
    docs = []
    for lecture_file in lecture_files:
        with open(lecture_file, "r") as f:
            lecture = yaml.safe_load(f)
        for block in lecture["content_blocks"]:
            if block["type"] == "video":
                video_id = block.get("youtube_video_id", None)
                if video_id is not None:
                    doc = load_from_youtube_transcript(video_id)
                    for d in doc:
                        d.metadata["title"] = block["title"]
                    docs.extend(doc)
                    print(f"Loaded transcript for {block['title']}")
    return docs


def split_documents(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    return text_splitter.split_documents(docs)


def load_from_youtube_transcript(video_id: str):
    loader = YoutubeLoader.from_youtube_url(
        f"https://www.youtube.com/watch?v={video_id}", add_video_info=False
    )
    return loader.load()
