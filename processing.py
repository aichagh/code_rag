import os
import shutil
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

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

# creating a header for each document so the system knows the file's
# path (and therefore name)
for doc in docs:
    file_path = doc.metadata.get("source", "unknown")
    doc.page_content = f"FILE PATH: {file_path}\n\n{doc.page_content}"

# creating a file tree so the system knows the overall structure
# of the project
def file_tree(root_dir, exclude={'.git', 'node_modules', 'dist'}):
    tree = []
    for root, dirs, files in os.walk(root_dir):
        # extract the directories
        dirs[:] = [d for d in dirs if d not in exclude]
        
        # get the "level" (how deep we are in the tree)
        # the deeper we are, the more indented
        level = root.replace(root_dir, "").count(os.sep)
        indent = '  ' * level
        tree.append(f"{indent}-folder- {os.path.basename(root)}/")
        
        # indexing the files in the folder
        for f in files:
            tree.append(f"{indent}  -file- {f}")
    return "\n".join(tree)

file_tree_data = file_tree('repo')

file_tree_doc = Document(
        page_content=file_tree_data,
        metadata={"source": "project_structure",
                  "content_type": "file_tree"}
        )

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

vector_db.add_documents([file_tree_doc])
