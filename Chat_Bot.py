import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from PIL import Image
import base64
from io import BytesIO

# -------------------
# Config
# -------------------
OLLAMA_MODEL = "phi3"  # use a small, efficient model

st.set_page_config(page_title="AI Shopping Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Shopping Assistant")

# -------------------
# Logo Handling
# -------------------
def logo_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

try:
    logo = Image.open("image-generator.jpeg")  # <-- put your robot logo here
    st.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{logo_to_base64(logo)}" width="500"/>
    </div>
    """,
    unsafe_allow_html=True
)


except FileNotFoundError:
    st.warning("Logo image not found. Place 'recommender.png' in the 'assets' folder.")

st.markdown("""
Welcome! 👋 I’m your AI shopping buddy.  
I can chat casually with you, and when you’re ready just ask things like:  
- *"Which is better, X or Y?"*  
- *"Can you recommend a laptop?"*  
- *"Compare iPhone and Samsung."*  

When you’re done, click **End Session & Get Summary** to see a recap of our chat 😃
""")

st.info("Ensure the Ollama server is running for the assistant to work.")
st.divider()

# -------------------
# LLM Setup
# -------------------
@st.cache_resource
def load_llm():
    try:
        return OllamaLLM(model=OLLAMA_MODEL)
    except Exception as e:
        st.error(f"Failed to connect to Ollama model '{OLLAMA_MODEL}'. Error: {e}")
        return None

llm = load_llm()

# Prompt template for chat
recommender_prompt = ChatPromptTemplate.from_template("""
You are a friendly AI shopping assistant for an e-commerce platform. 
- You greet and interact naturally like a helpful chatbot. 
- You can answer small talk casually (e.g., "hi", "how are you").
- You only recommend products if the user asks explicitly, 
  such as with phrases like: "what is better", "can you recommend", 
  "compare these", or "which one should I buy".
- When recommending, be concise and provide 2–3 options with short descriptions.
Conversation so far:
{chat_history}

User: {user_input}
Assistant:""")

# Prompt template for summary
summary_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant summarizing a shopping session.  
Write a short, friendly summary for the user.  

Rules:
1. Start with: "This is an AI-generated summary of your shopping session."  
2. Highlight the main topics/products discussed.  
3. Mention any comparisons or recommendations made.  
4. Keep it conversational, not too formal.  
5. End by inviting the user to return anytime.  

Conversation History:
{chat_history}

Summary:
""")

# -------------------
# Session State
# -------------------
if "assistant_memory" not in st.session_state:
    st.session_state.assistant_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="text"
    )

if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []

if "session_summary" not in st.session_state:
    st.session_state.session_summary = None

# -------------------
# Chains
# -------------------
@st.cache_resource
def get_conversation_chain(_llm, _prompt, _memory):
    if _llm is None:
        return None
    return LLMChain(llm=_llm, prompt=_prompt, memory=_memory, verbose=False)

@st.cache_resource
def get_summary_chain(_llm, _prompt):
    if _llm is None:
        return None
    return LLMChain(llm=_llm, prompt=_prompt, verbose=False)

conversation_chain = get_conversation_chain(llm, recommender_prompt, st.session_state.assistant_memory)
summary_chain = get_summary_chain(llm, summary_prompt)

# -------------------
# Chat Loop
# -------------------

for msg in st.session_state.assistant_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if llm and conversation_chain and not st.session_state.session_summary:
    user_input = st.chat_input("Say something...")
    if user_input:
        st.session_state.assistant_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if any(keyword in user_input.lower() for keyword in ["which", "recommend", "better", "compare"]):
                        full_response = conversation_chain.invoke({"user_input": user_input})
                        reply = full_response["text"].strip()
                    else:
                        reply = "Hey there 👋 I'm here to chat. Ask me for recommendations anytime!"
                    st.markdown(reply)
                    st.session_state.assistant_messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Error communicating with the AI model: {e}")

# -------------------
# End Session & Summary
# -------------------
st.divider()
if not st.session_state.session_summary:
    if st.button("End Session & Get Summary"):
        if not st.session_state.assistant_memory.chat_memory.messages:
            st.warning("No conversation to summarize yet.")
        else:
            with st.spinner("Generating summary..."):
                history = st.session_state.assistant_memory.load_memory_variables({})["chat_history"]
                formatted_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
                try:
                    response = summary_chain.invoke({"chat_history": formatted_history})
                    st.session_state.session_summary = response["text"].strip()
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
else:
    st.subheader("📝 Session Summary")
    st.markdown(st.session_state.session_summary)

# -------------------
# Restart Session
# -------------------
if st.button("Start New Session (Clear Chat & Summary)"):
    st.session_state.assistant_messages = []
    if "assistant_memory" in st.session_state:
        st.session_state.assistant_memory.clear()
    st.session_state.session_summary = None
    st.rerun()