import argparse
import faiss
import os
import pickle

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import VectorDBQAWithSourcesChain

parser = argparse.ArgumentParser(description='Paepper.com Q&A')
parser.add_argument('question', type=str, help='Your question for Paepper.com')
args = parser.parse_args()

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

chain = VectorDBQAWithSourcesChain.from_llm(
        llm=ChatOpenAI(temperature=0.5, verbose=True,model_name='gpt-3.5-turbo'), vectorstore=store, verbose=True)
result = chain({"question": args.question})

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
