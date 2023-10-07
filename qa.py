from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

global retriever


def load_embedding():
    embedding = OpenAIEmbeddings()
    global retriever
    vectordb = Chroma(persist_directory='db', embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})


def prompt(query):
    prompt_template = """请注意：请尝试基于给出的材料来解决问题，运用有信心的简单推理，但是不要使用曾经记忆的信息：
    Context: {context}
    Context: {context}
    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    docs = retriever.get_relevant_documents(query)

    # 基于docs来prompt，返回你想要的内容
    chain = load_qa_chain(ChatOpenAI(temperature=0.5), chain_type="stuff", prompt=PROMPT, verbose=True)
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=False)
    return result['output_text']


if __name__ == "__main__":
    # load embedding
    load_embedding()
    # 循环输入查询，直到输入 "exit" 测试
    while True:
        query = input("Enter query (or 'exit' to quit): ")
        if query == 'exit':
            print('exit')
            break
        elif query.startswith("file://"):
            # 在file://的这种情况下，我们需要从文件中读取对应的内容 目前多行输入可以通过这种方式来处理。
            print(query)
            query = open(query[7:]).read()
        print("Query:" + query + '\nAnswer:' + prompt(query) + '\n')
