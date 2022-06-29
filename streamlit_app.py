from ctypes import resize
from urllib.parse import unquote

import arabic_reshaper
import streamlit as st
from bidi.algorithm import get_display

from html_utils import ga
from utils import annotate_answer, get_results, shorten_text, get_offline_results

ga()

st.set_page_config(
    page_title="Arabic QA app",
    page_icon="📖",
    initial_sidebar_state="expanded"
    # layout="wide"
)
# footer()


rtl = lambda w: get_display(f"{arabic_reshaper.reshape(w)}")


_, col1, col3 ,col4= st.beta_columns([1,2,1,1])

with col1:
    st.image("is2alni_logo.png", width=200)
    st.title("إسألني أي شيء")

st.markdown(
    """
<style>
p, div, input, label {
  text-align: right;
}
</style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Info")
#st.sidebar.write("Made by [Zeyad Ahmed](https://www.linkedin.com/in/zeyadahmed1/)")
st.sidebar.image("AraELECTRA.png", width=150)
st.sidebar.write("Powered by [AraELECTRA-ASQuADv2.0](https://huggingface.co/ZeyadAhmed/AraElectra-ASQuADv2-QA)")
st.sidebar.write("Trained on [ASQuADv2.0](https://huggingface.co/datasets/ZeyadAhmed/ASQuADv2.0)")
st.sidebar.write(
    "Source Code [GitHub](https://github.com/zeyadahmed10/arabic-wikipedia-qa-streamlit)"
)
st.sidebar.write("\n")
n_answers = st.sidebar.slider(
    "Max. number of answers", min_value=1, max_value=10, value=2, step=1
)

question = st.text_input("", value="من هو جو بايدن؟")
if "؟" not in question:
    question += "؟"

st.markdown(
        """
    <style>
    checkbox {
    text-align: right;
    }
    </style>
        """,
        unsafe_allow_html=True,
)
offline_flag = st.checkbox(label = "ادخل قطعة بنفسك", value=False)
if offline_flag:
    doc = st.text_input(label = "")
run_query = st.button("أجب")
if run_query:
    # https://discuss.streamlit.io/t/showing-a-gif-while-st-spinner-runs/5084
    with st.spinner("... جاري البحث "):
        if offline_flag:
            results_dict = get_offline_results(question, doc)
        else:
            results_dict = get_results(question)

    if len(results_dict) > 0:
        st.write("## :الأجابات هي")
        for result in results_dict["results"][:n_answers]:
            annotate_answer(result)
            x = f"{(1-result['conf']):.2f}"
            new_title = f'<p style="font-family:sans-serif; color:#990000; font-size: 18px;">السؤال يمكن إجابته بنسبة: {x}</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            if not offline_flag:
                x = f"{(1-result['conf']):.2f}"
                new_title = f'<p style="font-family:sans-serif; color:#990000; font-size: 18px;">السؤال يمكن إجابته بنسبة: {x}</p>'
                st.markdown(new_title, unsafe_allow_html=True)
                f"[**المصدر**](<{result['link']}>)"
    else:
        st.write("## 😞 ليس لدي جواب")
