import os
import shutil
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# cloning the repo
local_repo_path = 'repo'

if os.path.exists(local_repo_path):
    shutil.rmtree(local_repo_path)

repo_url = 'https://github.com/aichagh/Criticove'
repo = Repo.clone_from(repo_url, local_repo_path)

# loading the files in the main folder
test_folder_path = 'repo/app/src/main/java/com/criticove'

loader = GenericLoader.from_filesystem(
            test_folder_path,
            suffixes=['.kt'],
            glob='**/*',
            parser=LanguageParser(
                language=Language.KOTLIN,
                parser_threshold=50
                )
            )

docs=loader.load()

# creating the chunks
code_splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size=3000,
        chunk_overlap=300,
        language=Language.KOTLIN)

chunks = code_splitter.split_documents(docs)

# creating the embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# making a persistent directory
perma_dir = "./perma_dir"

if os.path.exists(perma_dir):
    shutil.rmtree(perma_dir)

vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=perma_dir
        )

print("Indexed elements:", vector_db._collection.count())
