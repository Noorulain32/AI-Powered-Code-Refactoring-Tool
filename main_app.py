import streamlit as st
from model.refactor import CodeRefactor, auto_format
from utils.analyzer import run_flake8

st.set_page_config(page_title="AI Code Review & Refactor")

st.title("🤖 AI Code Review and Refactor Tool")
st.markdown("Enter your Python code below to get AI-powered suggestions and refactoring.")

user_code = st.text_area("Your Python Code", height=300)

if st.button("Analyze and Refactor"):
    if user_code.strip():
        st.subheader("🔍 Static Code Analysis (flake8)")
        st.code(run_flake8(user_code), language="bash")

        st.subheader("🧹 Auto-formatted Code (autopep8)")
        formatted = auto_format(user_code)
        st.code(formatted, language="python")

        st.subheader("🤖 AI-Refactored Code (CodeT5)")
        model = CodeRefactor()
        refactored = model.refactor(user_code)
        st.code(refactored, language="python")
    else:
        st.warning("Please paste some code first.")