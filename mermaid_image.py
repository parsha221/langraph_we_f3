import os
from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
 
class State(TypedDict):
    state: str
 
class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret
 
    def __call__(self, state: State) -> Any:
        print(f"Adding {self._value} to {state['state']}")
        return {"state": [self._value]}
 
 
# Add nodes
builder = StateGraph(State)
 
# Initialize each node with node_secret
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))
 
# Flow (add edges after nodes are defined)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", "c")
builder.add_edge("c", "d")
builder.add_edge("d", END)
 
# Compile graph and display it
graph = builder.compile()
 
# Define path in Downloads folder
graph_directory = r"C:\Users\verizon\Pictures\New folder"
graph_filename = "graph_output.png"
graph_image_path = os.path.join(graph_directory, graph_filename)
 
# Save graph as PNG file
graph_image = graph.get_graph().draw_mermaid_png()
with open(graph_image_path, "wb") as f:
    f.write(graph_image)
print(f"Graph saved at: {graph_image_path}")
 
# Open the image automatically (Windows only)
if os.path.exists(graph_image_path):
    os.startfile(graph_image_path)
else:
    print(f"Error: file not found at {graph_image_path}")
 
graph.invoke({"state": []})
