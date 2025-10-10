"""Streamlit entry-point for Project Cook.

The app focuses on providing a lightweight experience that works even when the
optional assets (model weights, dataset or translator service) are not
available.  It surfaces clear guidance to the user on how to supply those
assets so a freshly deployed environment can be configured quickly.
"""

from __future__ import annotations

import streamlit as st

from project_cook.interface import main as interface_main

MODEL_PATH = interface_main.MODEL_PATH
DATASET_PATH = interface_main.DATASET_PATH
INDEX_DIR = interface_main.INDEX_DIR
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
}


def _render_sidebar_status() -> None:
    """Display deployment diagnostics in the sidebar."""

    st.sidebar.header("Deployment status")
    if MODEL_PATH.exists():
        st.sidebar.success(f"Model located at: {MODEL_PATH}")
    else:
        st.sidebar.warning(
            "No model file detected. Upload your trained weights and set"
            " `COOK_MODEL_PATH` to point to them.")

    if DATASET_PATH.exists():
        st.sidebar.success(f"Dataset available at: {DATASET_PATH}")
        if INDEX_DIR.exists():
            st.sidebar.info(f"Existing search index found in: {INDEX_DIR}")
    else:
        st.sidebar.warning(
            "Recipe search is disabled because the dataset file is missing."
            " Provide a CSV and update `COOK_DATASET_PATH` to enable Whoosh.")

    st.sidebar.markdown(
        "Configure the application via environment variables or a `.streamlit/"
        "secrets.toml` file. Refer to the README for detailed instructions.")


def _prediction_section(language: str) -> None:
    """Render the section that allows users to classify an image."""

    st.subheader("Classify a dish")
    st.markdown(
        "Upload a picture of a meal to let the model guess which recipe it"
        " belongs to. Predictions are cached so subsequent requests are fast.")

    with st.form("prediction_form"):
        uploaded_file = st.file_uploader(
            "Select a food photo", type=["png", "jpg", "jpeg", "gif"],
            help="High quality images work best.")
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    if uploaded_file is None:
        st.warning("Please upload an image before requesting a prediction.")
        return

    try:
        result = interface_main.pred_streamlit(uploaded_file.getvalue(), language)
    except FileNotFoundError as exc:
        st.error(
            "The model file could not be found. Place your trained weights in"
            f" `{MODEL_PATH}` or point `COOK_MODEL_PATH` to the correct"
            " location before running predictions.")
        st.stop()
    except RuntimeError as exc:  # TensorFlow not installed, etc.
        st.error(str(exc))
        st.stop()

    predictions = result.get('prediction', [])
    translations = result.get('translation', [])

    if not predictions:
        st.info("The model did not return any predictions.")
        return

    st.success("Prediction complete!")
    for original, translated in zip(predictions, translations):
        if language != 'en':
            st.markdown(f"* {translated} _(original label: {original})_")
        else:
            st.markdown(f"* {original}")


def _render_recipe_hit(hit: dict) -> None:
    """Pretty print a single recipe search result."""

    st.markdown(f"#### {hit['title']}")
    st.caption(f"Source: {hit['source']}")
    st.write(hit['ingredients'])
    st.write(hit['directions'])
    if hit.get('link'):
        st.markdown(f"[Open recipe]({hit['link']})", unsafe_allow_html=False)


def _search_section(language: str) -> None:
    """Render the recipe search functionality."""

    st.subheader("Search recipes by ingredients")
    st.markdown(
        "Type a few ingredients or keywords to find matching recipes in the"
        " dataset. Results will be translated automatically when possible.")

    with st.form("search_form"):
        query = st.text_input("What ingredients do you have?", max_chars=120)
        submitted = st.form_submit_button("Search")

    if not submitted:
        return

    if not query.strip():
        st.warning("Enter at least one ingredient to start the search.")
        return

    results = interface_main.search_recipes(query, language)
    if not results:
        if not DATASET_PATH.exists():
            st.warning(
                "No dataset available – upload `full_dataset.csv` and update"
                " `COOK_DATASET_PATH` to enable recipe search.")
        else:
            st.info("No recipes matched your query. Try different ingredients.")
        return

    for hit in results[:20]:  # Keep the UI snappy by limiting the number of rows.
        _render_recipe_hit(hit)
        st.divider()


def main() -> None:
    """Application entry-point."""

    st.set_page_config(page_title="Project Cook", layout="wide")
    st.title("Project Cook")
    st.caption("Find the perfect recipe and classify dishes with a single app.")

    _render_sidebar_status()

    language = st.sidebar.selectbox(
        "Display language", options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=SUPPORTED_LANGUAGES.get)

    classify_tab, search_tab = st.tabs(["Classify", "Recipe search"])
    with classify_tab:
        _prediction_section(language)
    with search_tab:
        _search_section(language)


if __name__ == "__main__":
    main()
