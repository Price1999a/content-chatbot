import requests
from bs4 import BeautifulSoup
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma


def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)

if __name__ == '__main__':
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
    documents = []
    for page in pages:
        metadata = {"source": page['source']}
        if(page['source'].startswith('https://')):
            splits = text_splitter.split_text(page['text'])
        else:
            splits = markdown_splitter.split_text(page['text'])
        for split in splits:
            documents.append(Document(page_content = split,metadata = metadata))
        print(f"Split {page['source']} into {len(splits)} chunks")
    persist_directory = 'db'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
    print("done")