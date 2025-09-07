# =====================================================================================
# ALL-IN-ONE APP.PY (LANGCHAIN VERSION)
# This version uses the core LangChain library to avoid the crewAI bug.
# =====================================================================================

import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

# --- 1. SETUP ---
# Load environment variables from .env file
load_dotenv()

# --- 2. INITIALIZE MODELS AND TOOLS ---
# Initialize the Language Model (LLM)
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant", # <-- THIS IS THE NEW, CORRECT MODEL
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize the Tavily Search Tool
search_tool = TavilySearchResults(max_results=3)

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="AI Research Assistant", page_icon="🤖")

st.title("🤖 AI Research Assistant")
st.markdown("### Let your AI team research any topic for you!")
st.markdown("---")

topic = st.text_input(
    "What topic would you like to research?",
    placeholder="e.g., The Impact of AI on Climate Change"
)

if st.button("Start Research"):
    if topic:
        # --- 4. AGENTIC WORKFLOW ---
        # This workflow simulates the multi-agent system using LangChain
        
        # Agent 1: Searcher
        with st.spinner("Agent 1: Searching the web..."):
            search_results = search_tool.invoke(topic)
            st.success("Search complete!")

        # Agent 2: Summarizer
        with st.spinner("Agent 2: Summarizing the findings..."):
            summarizer_prompt = ChatPromptTemplate.from_template(
                """Summarize the following search results into a concise overview.
                Focus on the key facts, figures, and main points.
                
                Search Results:
                {results}"""
            )
            summarizer_chain = summarizer_prompt | llm
            summary = summarizer_chain.invoke({"results": search_results})
            st.success("Summarization complete!")
            
        # Agent 3: Critic (simulated)
        with st.spinner("Agent 3: Critiquing the summary..."):
            critic_prompt = ChatPromptTemplate.from_template(
                """Critically analyze the following summary. Identify potential biases,
                missing information, or logical fallacies. Provide a list of 2-3 bullet
                points for improvement.

                Summary:
                {summary}"""
            )
            critic_chain = critic_prompt | llm
            critique = critic_chain.invoke({"summary": summary.content})
            st.success("Critique complete!")

        # Agent 4: Report Generator
        with st.spinner("Agent 4: Generating the final report..."):
            report_prompt = ChatPromptTemplate.from_template(
                """Create a final, well-structured research report based on the provided
                summary and critical feedback. The report should have an introduction,
                a body discussing the key findings, and a conclusion. It should be at
                least 4 paragraphs long.

                Original Summary:
                {summary}

                Critical Feedback:
                {critique}"""
            )
            report_chain = report_prompt | llm
            final_report = report_chain.invoke({"summary": summary.content, "critique": critique.content})
            st.success("Report generation complete!")

        # --- 5. DISPLAY FINAL RESULT ---
        st.markdown("---")
        st.subheader("Final Research Report:")
        st.markdown(final_report.content)

    else:
        st.error("Please enter a topic to start the research.")