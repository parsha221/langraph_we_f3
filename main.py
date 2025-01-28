import os
from PyPDF2 import PdfReader
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
import gradio as gr
from typing import TypedDict
 
 
# Define the TypedDicts for State Management
class InputState(TypedDict):
    files: list
    user_question: str
 
 
class OutputState(TypedDict):
    response: str
    matched_file: str
 
 
class OverallState(TypedDict):
    text_chunks: list
    chunk_sources: dict
    vector_store: Chroma
    user_question: str
    response: str
    matched_file: str
 
 
class PrivateState(TypedDict):
    match: list
 
 
# Define the StateGraph class
class StateGraph:
    def __init__(self, overall_state_type, input, output):
        self.nodes = {}
        self.edges = {}
        self.overall_state_type = overall_state_type
        self.input_type = input
        self.output_type = output
 
    def add_node(self, name, func):
        self.nodes[name] = func
 
    def add_edge(self, from_node, to_node):
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)
 
    def compile(self):
        return self
 
    def invoke(self, input_state):
        state = input_state
        overall_state = None
        current_node = START
        while current_node in self.edges:
            next_node = self.edges[current_node][0]
            if next_node not in self.nodes:
                raise ValueError(f"Node function for {next_node} not defined.")
            if next_node == "node_3":
                state = self.nodes[next_node](state, overall_state)
            else:
                state = self.nodes[next_node](state)
            if current_node == START:
                overall_state = state
            current_node = next_node
 
        # Return final state once the graph is completed
        return state
 
 
START = "START"
 
 
# Node Definitions
def node_1(state: InputState) -> OverallState:
    text_chunks = []
    chunk_sources = {}
 
    for file in state["files"]:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=30000,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        text_chunks.extend(chunks)
 
        for chunk in chunks:
            chunk_sources[chunk] = file  # Use plain chunk as key
 
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_texts(text_chunks, embeddings)
 
    return {
        "text_chunks": text_chunks,
        "chunk_sources": chunk_sources,
        "vector_store": vector_store,
        "user_question": state["user_question"],
        "response": "",
        "matched_file": "",
    }
 
 
def node_2(state: OverallState) -> PrivateState:
    match = state["vector_store"].similarity_search(state["user_question"])
    return {"match": match}
 
 
def node_3(state: PrivateState, overall_state: OverallState) -> OutputState:
    llm = OllamaLLM(model="llama3.1")
    chain = load_qa_chain(llm, chain_type="stuff")
 
    # Extract the plain text content of the matched chunks
    matched_chunks = [doc.page_content for doc in state["match"]]
 
    # Generate the response
    response = chain.run(input_documents=state["match"], question=overall_state["user_question"])
 
    # Find the source file for the first matched chunk
    first_matched_chunk = matched_chunks[0] if matched_chunks else ""
    matched_file = overall_state["chunk_sources"].get(first_matched_chunk, "Unknown file")
 
    return {
        "response": response,
        "matched_file": matched_file,
    }
 
 
# Build the StateGraph
builder = StateGraph(OverallState, input=InputState, output=OutputState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
 
graph = builder.compile()
 
 
# Define the Gradio Interface
def gradio_interface(file1, file2, file3, user_question):
    files = [file1, file2, file3]
    files = [file for file in files if file is not None]
    if len(files) > 0:
        input_state = {"files": files, "user_question": user_question}
        try:
            output_state = graph.invoke(input_state)
            return output_state
        except Exception as e:
            return {"error": str(e)}
    return "Please upload one or more files and enter a question."
 
 
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.components.File(type="filepath", label="Upload PDF file 1"),
        gr.components.File(type="filepath", label="Upload PDF file 2"),
        gr.components.File(type="filepath", label="Upload PDF file 3"),
        gr.components.Textbox(label="Type Your Question Here"),
    ],
    outputs="json",
    title="Chatbot with PDF Support",
    description="Upload one or more PDF files and ask questions about their content.",
)
 
iface.launch(share=True)
