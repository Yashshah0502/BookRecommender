import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()
# import os
#
# os.environ["ANONYMIZED_TELEMETRY"] = "false"
#
# import pandas as pd
# import numpy as np
# from dotenv import load_dotenv
#
# from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_chroma import Chroma
#
# import gradio as gr
#
# load_dotenv()
#
# books = pd.read_csv("books_with_emotions.csv")
# books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
# books["large_thumbnail"] = np.where(
#     books["large_thumbnail"].isna(),
#     "cover-not-found.jpg",
#     books["large_thumbnail"],
# )
#
# raw_documents = TextLoader("tagged_description.txt").load()
# text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)
# db_books = Chroma.from_documents(documents, OpenAIEmbeddings())
#
#
# def retrieve_semantic_recommendations(
#         query: str,
#         category: str = None,
#         tone: str = None,
#         initial_top_k: int = 50,
#         final_top_k: int = 16,
# ) -> pd.DataFrame:
#     recs = db_books.similarity_search(query, k=initial_top_k)
#     books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
#     book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)
#
#     if category != "All":
#         book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
#     else:
#         book_recs = book_recs.head(final_top_k)
#
#     if tone == "Happy":
#         book_recs.sort_values(by="joy", ascending=False, inplace=True)
#     elif tone == "Surprising":
#         book_recs.sort_values(by="surprise", ascending=False, inplace=True)
#     elif tone == "Angry":
#         book_recs.sort_values(by="anger", ascending=False, inplace=True)
#     elif tone == "Suspenseful":
#         book_recs.sort_values(by="fear", ascending=False, inplace=True)
#     elif tone == "Sad":
#         book_recs.sort_values(by="sadness", ascending=False, inplace=True)
#
#     return book_recs
#
#
# def recommend_books(query: str, category: str, tone: str):
#     if not query.strip():
#         return []
#
#     recommendations = retrieve_semantic_recommendations(query, category, tone)
#     results = []
#
#     for _, row in recommendations.iterrows():
#         description = row["description"]
#         truncated_desc_split = description.split()
#         truncated_description = " ".join(truncated_desc_split[:30]) + "..."
#
#         authors_split = row["authors"].split(";")
#         if len(authors_split) == 2:
#             authors_str = f"{authors_split[0]} and {authors_split[1]}"
#         elif len(authors_split) > 2:
#             authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
#         else:
#             authors_str = row["authors"]
#
#         caption = f"**{row['title']}** \n*by {authors_str}*\n\n{truncated_description}"
#         results.append((row["large_thumbnail"], caption))
#
#     return results
#
#
# categories = ["All"] + sorted(books["simple_categories"].unique())
# tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
#
# # Enhanced CSS with animations and modern styling
# enhanced_css = """
# <style>
#     /* Import modern font */
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
#
#     /* Global enhancements */
#     .gradio-container {
#         font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
#         max-width: 1400px !important;
#         margin: 0 auto !important;
#         padding: 1rem !important;
#     }
#
#     /* Enhanced buttons */
#     .btn-primary {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
#         border: none !important;
#         border-radius: 12px !important;
#         padding: 12px 32px !important;
#         font-weight: 600 !important;
#         font-size: 1.1rem !important;
#         transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
#         box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
#         text-transform: none !important;
#     }
#
#     .btn-primary:hover {
#         transform: translateY(-2px) !important;
#         box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4) !important;
#         filter: brightness(1.05) !important;
#     }
#
#     .btn-primary:active {
#         transform: translateY(0px) !important;
#         box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
#     }
#
#     /* Enhanced input fields */
#     .gr-textbox, .gr-dropdown {
#         border-radius: 12px !important;
#         border: 2px solid #e2e8f0 !important;
#         transition: all 0.3s ease !important;
#         font-size: 1rem !important;
#         background: white !important;
#     }
#
#     .gr-textbox:focus, .gr-dropdown:focus {
#         border-color: #667eea !important;
#         box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
#         outline: none !important;
#     }
#
#     /* Gallery enhancements */
#     .gr-gallery {
#         border-radius: 16px !important;
#         border: 1px solid #e2e8f0 !important;
#         background: white !important;
#         padding: 1rem !important;
#     }
#
#     .gr-gallery img {
#         border-radius: 8px !important;
#         transition: transform 0.3s ease !important;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
#     }
#
#     .gr-gallery img:hover {
#         transform: scale(1.05) !important;
#         box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
#     }
#
#     /* Label enhancements */
#     .gr-form label {
#         font-weight: 600 !important;
#         color: #1e293b !important;
#         font-size: 0.95rem !important;
#         margin-bottom: 0.5rem !important;
#     }
#
#     /* Loading animation */
#     .loading-spinner {
#         display: inline-block;
#         width: 20px;
#         height: 20px;
#         border: 3px solid #f3f4f6;
#         border-top: 3px solid #667eea;
#         border-radius: 50%;
#         animation: spin 1s linear infinite;
#     }
#
#     @keyframes spin {
#         0% { transform: rotate(0deg); }
#         100% { transform: rotate(360deg); }
#     }
#
#     /* Fade-in animation */
#     .fade-in {
#         animation: fadeIn 0.6s ease-in-out;
#     }
#
#     @keyframes fadeIn {
#         from { opacity: 0; transform: translateY(20px); }
#         to { opacity: 1; transform: translateY(0); }
#     }
#
#     /* Enhanced cards */
#     .recommendation-card {
#         background: white;
#         border-radius: 16px;
#         padding: 1.5rem;
#         margin: 1rem 0;
#         box-shadow: 0 4px 20px rgba(0,0,0,0.08);
#         border: 1px solid #f1f5f9;
#         transition: all 0.3s ease;
#     }
#
#     .recommendation-card:hover {
#         transform: translateY(-4px);
#         box-shadow: 0 12px 40px rgba(0,0,0,0.12);
#     }
#
#     /* Progress bar enhancement */
#     .progress-bar {
#         background: linear-gradient(90deg, #667eea, #764ba2);
#         border-radius: 10px;
#         height: 4px;
#         animation: pulse 2s ease-in-out infinite;
#     }
#
#     @keyframes pulse {
#         0%, 100% { opacity: 0.7; }
#         50% { opacity: 1; }
#     }
#
#     /* Responsive design */
#     @media (max-width: 768px) {
#         .gradio-container {
#             padding: 0.5rem !important;
#         }
#
#         .btn-primary {
#             padding: 10px 24px !important;
#             font-size: 1rem !important;
#         }
#     }
#
#     /* Dark mode support */
#     @media (prefers-color-scheme: dark) {
#         .gr-textbox, .gr-dropdown {
#             background: #1e293b !important;
#             border-color: #334155 !important;
#             color: white !important;
#         }
#
#         .gr-gallery {
#             background: #1e293b !important;
#             border-color: #334155 !important;
#         }
#     }
# </style>
# """
#
# # Enhanced interface with animations and modern design
# with gr.Blocks(title="AI Book Discovery", css=enhanced_css) as dashboard:
#     # Enhanced header with gradient and animations
#     gr.HTML("""
#     <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#                 padding: 4rem 2rem;
#                 border-radius: 20px;
#                 margin-bottom: 2rem;
#                 color: white;
#                 text-align: center;
#                 position: relative;
#                 overflow: hidden;
#                 box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);">
#         <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0;
#                     background: url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1000 1000\"><defs><radialGradient id=\"a\" cx=\"50%\" cy=\"50%\" r=\"50%\"><stop offset=\"0%\" stop-color=\"%23ffffff\" stop-opacity=\"0.1\"/><stop offset=\"100%\" stop-color=\"%23ffffff\" stop-opacity=\"0\"/></radialGradient></defs><circle cx=\"200\" cy=\"200\" r=\"100\" fill=\"url(%23a)\"/><circle cx=\"800\" cy=\"300\" r=\"150\" fill=\"url(%23a)\"/><circle cx=\"400\" cy=\"700\" r=\"120\" fill=\"url(%23a)\"/></svg>') no-repeat center center;
#                     background-size: cover;
#                     opacity: 0.1;"></div>
#         <div style="position: relative; z-index: 1;">
#             <h1 style="font-size: 3.5rem; margin-bottom: 1rem; font-weight: 700;
#                        text-shadow: 0 4px 8px rgba(0,0,0,0.2);
#                        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
#                        -webkit-background-clip: text;
#                        -webkit-text-fill-color: transparent;
#                        background-clip: text;">
#                 📚 AI Book Discovery
#             </h1>
#             <p style="font-size: 1.4rem; opacity: 0.95; margin: 0; font-weight: 300;
#                       text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
#                 Discover your next favorite book through intelligent semantic search
#             </p>
#             <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
#                 <div style="display: flex; align-items: center; gap: 0.5rem; opacity: 0.9;">
#                     <span style="font-size: 1.2rem;">🤖</span>
#                     <span style="font-size: 0.9rem;">AI-Powered</span>
#                 </div>
#                 <div style="display: flex; align-items: center; gap: 0.5rem; opacity: 0.9;">
#                     <span style="font-size: 1.2rem;">🎯</span>
#                     <span style="font-size: 0.9rem;">Personalized</span>
#                 </div>
#                 <div style="display: flex; align-items: center; gap: 0.5rem; opacity: 0.9;">
#                     <span style="font-size: 1.2rem;">⚡</span>
#                     <span style="font-size: 0.9rem;">Instant Results</span>
#                 </div>
#             </div>
#         </div>
#     </div>
#     """)
#
#     # Enhanced search section with better styling
#     gr.HTML("""
#     <div style="background: white; padding: 2rem; border-radius: 16px;
#                 box-shadow: 0 8px 32px rgba(0,0,0,0.08);
#                 border: 1px solid #f1f5f9; margin-bottom: 2rem;">
#         <h2 style="color: #1e293b; font-size: 1.8rem; font-weight: 600;
#                    margin-bottom: 1.5rem; text-align: center;">
#             🔍 Tell us what you're looking for
#         </h2>
#     </div>
#     """)
#
#     user_query = gr.Textbox(
#         label="📖 Book Description",
#         placeholder="e.g., A heartwarming story about friendship and overcoming challenges, set in a small town with memorable characters...",
#         lines=3,
#         max_lines=5
#     )
#
#     with gr.Row():
#         with gr.Column(scale=1):
#             category_dropdown = gr.Dropdown(
#                 choices=categories,
#                 label="📚 Category",
#                 value="All",
#                 interactive=True
#             )
#         with gr.Column(scale=1):
#             tone_dropdown = gr.Dropdown(
#                 choices=tones,
#                 label="🎭 Emotional Tone",
#                 value="All",
#                 interactive=True
#             )
#
#     submit_button = gr.Button(
#         "✨ Find My Perfect Books",
#         variant="primary",
#         size="lg"
#     )
#
#     # Enhanced example suggestions with interactive styling
#     gr.HTML("""
#     <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
#                 padding: 2rem; border-radius: 16px;
#                 border: 1px solid #e2e8f0; margin: 2rem 0;
#                 box-shadow: 0 4px 20px rgba(0,0,0,0.05);">
#         <h3 style="margin: 0 0 1.5rem 0; color: #1e293b; font-weight: 600;
#                    text-align: center; font-size: 1.3rem;">
#             💡 Need inspiration? Try these searches:
#         </h3>
#         <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
#                     gap: 1rem; margin-top: 1rem;">
#             <div style="background: white; padding: 1rem; border-radius: 12px;
#                         box-shadow: 0 2px 8px rgba(0,0,0,0.05);
#                         border: 1px solid #f1f5f9; transition: all 0.3s ease;
#                         cursor: pointer;"
#                  onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.1)'"
#                  onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.05)'">
#                 <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">🤝</div>
#                 <div style="font-weight: 500; color: #1e293b; margin-bottom: 0.5rem;">Friendship & Forgiveness</div>
#                 <div style="font-size: 0.9rem; color: #64748b;">A story about forgiveness and redemption</div>
#             </div>
#             <div style="background: white; padding: 1rem; border-radius: 12px;
#                         box-shadow: 0 2px 8px rgba(0,0,0,0.05);
#                         border: 1px solid #f1f5f9; transition: all 0.3s ease;
#                         cursor: pointer;"
#                  onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.1)'"
#                  onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.05)'">
#                 <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">🕵️</div>
#                 <div style="font-weight: 500; color: #1e293b; margin-bottom: 0.5rem;">Mystery & Suspense</div>
#                 <div style="font-size: 0.9rem; color: #64748b;">Mystery with unexpected twists</div>
#             </div>
#             <div style="background: white; padding: 1rem; border-radius: 12px;
#                         box-shadow: 0 2px 8px rgba(0,0,0,0.05);
#                         border: 1px solid #f1f5f9; transition: all 0.3s ease;
#                         cursor: pointer;"
#                  onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.1)'"
#                  onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.05)'">
#                 <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">🌱</div>
#                 <div style="font-weight: 500; color: #1e293b; margin-bottom: 0.5rem;">Coming of Age</div>
#                 <div style="font-size: 0.9rem; color: #64748b;">Coming-of-age story with friendship</div>
#             </div>
#             <div style="background: white; padding: 1rem; border-radius: 12px;
#                         box-shadow: 0 2px 8px rgba(0,0,0,0.05);
#                         border: 1px solid #f1f5f9; transition: all 0.3s ease;
#                         cursor: pointer;"
#                  onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.1)'"
#                  onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.05)'">
#                 <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">⚔️</div>
#                 <div style="font-weight: 500; color: #1e293b; margin-bottom: 0.5rem;">Historical Fiction</div>
#                 <div style="font-size: 0.9rem; color: #64748b;">Historical fiction set during wartime</div>
#             </div>
#         </div>
#     </div>
#     """)
#
#     # Enhanced results section
#     gr.HTML("""
#     <div style="text-align: center; margin: 3rem 0 2rem 0;
#                 padding: 2rem; background: white; border-radius: 16px;
#                 box-shadow: 0 8px 32px rgba(0,0,0,0.08);">
#         <h2 style="color: #1e293b; font-size: 2.2rem; font-weight: 700;
#                    margin-bottom: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
#                    background-clip: text;">
#             📖 Your Personalized Recommendations
#         </h2>
#         <p style="color: #64748b; font-size: 1.1rem; margin: 0; font-weight: 400;">
#             Curated just for you based on your preferences and reading mood
#         </p>
#     </div>
#     """)
#
#     output = gr.Gallery(
#         label="",
#         show_label=False,
#         columns=4,
#         rows=4,
#         height=700,
#         object_fit="contain",
#         preview=True
#     )
#
#     # Enhanced footer with social proof
#     gr.HTML("""
#     <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
#                 padding: 3rem 2rem; border-radius: 16px;
#                 text-align: center; margin-top: 3rem;
#                 border: 1px solid #e2e8f0;">
#         <div style="display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap;
#                     margin-bottom: 2rem;">
#             <div style="text-align: center;">
#                 <div style="font-size: 2rem; font-weight: 700; color: #667eea; margin-bottom: 0.5rem;">10,000+</div>
#                 <div style="color: #64748b; font-size: 0.9rem;">Books Analyzed</div>
#             </div>
#             <div style="text-align: center;">
#                 <div style="font-size: 2rem; font-weight: 700; color: #667eea; margin-bottom: 0.5rem;">AI-Powered</div>
#                 <div style="color: #64748b; font-size: 0.9rem;">Semantic Search</div>
#             </div>
#             <div style="text-align: center;">
#                 <div style="font-size: 2rem; font-weight: 700; color: #667eea; margin-bottom: 0.5rem;">Instant</div>
#                 <div style="color: #64748b; font-size: 0.9rem;">Results</div>
#             </div>
#         </div>
#         <p style="color: #64748b; margin: 0; font-size: 1rem;">
#             Powered by advanced AI semantic search • Find books that match your exact mood and interests
#         </p>
#     </div>
#     """)
#
#     # Event handlers
#     submit_button.click(
#         fn=recommend_books,
#         inputs=[user_query, category_dropdown, tone_dropdown],
#         outputs=output
#     )
#
#     user_query.submit(
#         fn=recommend_books,
#         inputs=[user_query, category_dropdown, tone_dropdown],
#         outputs=output
#     )
#
# if __name__ == "__main__":
#     dashboard.launch(
#         server_name="127.0.0.1",  # Changed from 0.0.0.0
#         server_port=7860
#     )