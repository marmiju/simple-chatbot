import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent import conn, workflow

st.set_page_config(
    page_title="Simple Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def new_thread_id() -> str:
    return str(uuid.uuid4())


def to_text(content) -> str:
    """Message content can be a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return ""


def get_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def load_history(thread_id: str):
    """Read saved messages for this thread from the SQLite checkpointer."""
    snapshot = workflow.get_state(get_config(thread_id))
    return snapshot.values.get("messages", []) if snapshot.values else []


def get_saved_threads() -> list[str]:
    """Fetch all distinct thread IDs saved in SQLite (newest activity first)."""
    try:
        cursor = conn.execute(
            "SELECT thread_id FROM checkpoints GROUP BY thread_id ORDER BY MAX(rowid) DESC"
        )
        return [row[0] for row in cursor.fetchall() if row[0]]
    except Exception:
        return []


def get_thread_title(thread_id: str) -> str:
    """Use the first user message as the conversation title, or fall back to short UUID."""
    messages = load_history(thread_id)
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = to_text(msg.content).strip()
            if text:
                return text[:32] + ("..." if len(text) > 32 else "")
    return f"New chat ({thread_id[:8]})"


def stream_reply(user_input: str, thread_id: str):
    """Yield text chunks from the chat node as they are generated."""
    for chunk, metadata in workflow.stream(
        {"messages": [HumanMessage(content=user_input)], "status": "running"},
        config=get_config(thread_id),
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "chat":
            text = to_text(chunk.content)
            if text:
                yield text


# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------

if "threads" not in st.session_state:
    saved = get_saved_threads()
    st.session_state.threads = saved if saved else [new_thread_id()]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = st.session_state.threads[0]


# --------------------------------------------------
# Sidebar: All Conversations List
# --------------------------------------------------

with st.sidebar:
    st.header("💬 Conversations")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        fresh_id = new_thread_id()
        st.session_state.threads.insert(0, fresh_id)
        st.session_state.thread_id = fresh_id
        st.rerun()

    st.divider()

    # Merge any saved threads from DB with in-memory threads while preserving order
    for tid in get_saved_threads():
        if tid not in st.session_state.threads:
            st.session_state.threads.append(tid)

    for tid in st.session_state.threads:
        is_active = tid == st.session_state.thread_id
        title = get_thread_title(tid)
        if st.button(
            f"🗨️ {title}",
            key=f"thread_btn_{tid}",
            use_container_width=True,
            type="secondary" if not is_active else "primary",
        ):
            st.session_state.thread_id = tid
            st.rerun()

    st.divider()
    st.caption("Conversations are saved automatically in `memory_graph.db`.")

thread_id = st.session_state.thread_id


# --------------------------------------------------
# Main chat area
# --------------------------------------------------

st.title("🤖 Simple Chatbot")
st.caption(f"Thread ID: `{thread_id}`")

for message in load_history(thread_id):
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage): 
        role = "assistant"
    else:
        continue
    text = to_text(message.content)
    if text:
        with st.chat_message(role):
            st.markdown(text)

if prompt := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            st.write_stream(stream_reply(prompt, thread_id))
            st.rerun()
        except Exception as error:
            st.error(f"{type(error).__name__}: {error}")
