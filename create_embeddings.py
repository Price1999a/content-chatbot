import argparse
import pickle
import requests
import xmltodict

from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import MarkdownTextSplitter


def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Embedding website content')
    # parser.add_argument('-s', '--sitemap', type=str, required=False,
    #         help='URL to your sitemap.xml', default='https://www.paepper.com/sitemap.xml')
    # parser.add_argument('-f', '--filter', type=str, required=False,
    #         help='Text which needs to be included in all URLs which should be considered',
    #         default='https://www.paepper.com/blog/posts')
    # args = parser.parse_args()
    #
    # r = requests.get(args.sitemap)
    # xml = r.text
    # raw = xmltodict.parse(xml)

    souffle_urls = ['https://souffle-lang.github.io/program',
            'https://souffle-lang.github.io/relations',
            'https://souffle-lang.github.io/types',
            'https://souffle-lang.github.io/facts',
            'https://souffle-lang.github.io/rules',
            'https://souffle-lang.github.io/constraints',
            'https://souffle-lang.github.io/arguments',
            'https://souffle-lang.github.io/aggregates',
            'https://souffle-lang.github.io/directives',
            'https://souffle-lang.github.io/directives',
            'https://souffle-lang.github.io/components',
            'https://souffle-lang.github.io/choice',
            'https://souffle-lang.github.io/subsumption',
            'https://souffle-lang.github.io/functors',
            'https://souffle-lang.github.io/pragmas',
            'https://souffle-lang.github.io/tutorial',
            'https://souffle-lang.github.io/examples'
            ]
    ddlog_files = [
        './ddlog-md/language_reference.md',
        './ddlog-md/tutorial.md'
    ]
    pages = []
    for file in ddlog_files:
        pages.append({'text': open(file,'r').read(),'source': file})
    for url in souffle_urls:
        pages.append({'text': extract_text_from(url), 'source': url})


    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    markdown_splitter = MarkdownTextSplitter(chunk_size=1500)
    docs, metadatas = [], []
    for page in pages:
        if(page['source'].startswith('https://')):
            splits = text_splitter.split_text(page['text'])
        else:
            splits = markdown_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")

    store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)
