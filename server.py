from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain import OpenAI, LLMChain
from langchain.prompts import Prompt
from flask import Flask, request, jsonify
import os

index = faiss.read_index("training.index")

with open("faiss.pkl", "rb") as f:
  store = pickle.load(f)

store.index = index

with open("training/master.txt", "r") as f:
  promptTemplate = f.read()

prompt = Prompt(template=promptTemplate, input_variables=["history", "context", "question"])

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
  return "API Online"

app.run(host="0.0.0.0", port=3000)