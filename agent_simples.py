# %%
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display


# %%
llm = ChatGroq(model_name="qwen/qwen3-32b", max_retries=2)


# %%
def multiply(a: int, b: int) -> int:
    """Multiplica a e b.

    Args:
        a: primeiro int
        b: segundo int
    """
    return a * b


# Essa função será uma ferramenta
def add(a: int, b: int) -> int:
    """Soma a e b.

    Args:
        a: primeiro int
        b: segundo int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Divide a por b.

    Args:
        a: primeiro int
        b: segundo int
    """
    return a / b


# %%
tools = [add, divide, multiply]
llm_with_tools = llm.bind_tools(tools)

# %%
# Mensagem de sistema
sys_msg = SystemMessage(
    content="Você é um assistente solicito e gentil que faz contas."
)


# Nó
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# %%
# Criando o grafo
builder = StateGraph(MessagesState)

# Adicionando nós
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Adicionando arestas: essas determinam como o fluxo de controle se move
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # Se a última mensagem (resultado) do assistente é uma chamada de ferramenta -> tools_condition roteia para ferramentas
    # Se a última mensagem (resultado) do assistente não é uma chamada de ferramenta -> tools_condition roteia para END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# Mostrando o grafo
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

# %%
messages = [
    HumanMessage(
        content="Realize a soma de 27 + 43, depois multiplique por 5 e ao final divida por 3"
    )
]
messages = react_graph.invoke({"messages": messages})
# %%
for m in messages["messages"]:
    m.pretty_print()
# %%
