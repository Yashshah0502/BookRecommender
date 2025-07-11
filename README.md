# Book Recommender System
A modern, end-to-end semantic book recommendation engine leveraging large language models (LLMs), vector search, and interactive dashboards. This project helps users discover books tailored to their interests, emotional preferences, and genre, using advanced NLP and machine learning techniques.

## Project Overview
- **Content-based recommendations** using book descriptions and LLM embeddings.
- **Semantic search** for matching user queries to relevant books.
- **Zero-shot classification** to assign clean, meaningful categories (e.g., Fiction, Non-fiction, Children's Fiction).
- **Sentiment analysis** to filter or sort books by emotional tone (e.g., Joyful, Sad, Suspenseful).
- **Interactive Gradio dashboard** for a user-friendly, visual experience.

## Architecture
The system is built as a modular pipeline, integrating data cleaning, NLP, vector databases, and a web-based dashboard.

Flowchart diagram of the Book Recommender project architecture:
## Features
- **Data Cleaning & Preprocessing:** Handles missing values, normalizes categories, and ensures high-quality inputs.
- **Vector Embeddings:** Transforms book descriptions into high-dimensional vectors using LLMs.
- **Vector Search (Chroma):** Efficiently finds books similar to any user query.
- **Zero-shot Classification:** Automatically classifies books into broad, user-friendly categories.
- **Sentiment Analysis:** Detects the dominant emotion in each book description for mood-based filtering.
- **Gradio Dashboard:** Lets users search, filter by category, and sort by emotional tone, with book covers and summaries.

## Getting Started
### Prerequisites
- Python 3.8+
- API key for OpenAI (for embeddings)
- Kaggle account (for dataset access)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yashshah0502/BookRecommender.git
   cd BookRecommender
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your OpenAI API key:**
   - Create a `.env` file in the project root.
   - Add: `OPENAI_API_KEY=your_key_here`

4. **Download the dataset:**
   - Place the required book dataset CSV in the project directory, or use the provided scripts to download from Kaggle.

### Running the Project
- **Data Cleaning & Preprocessing:**
  - Run `data-exploration.ipynb` to clean and prepare the dataset.
- **Vector Search & Classification:**
  - Use `Vector-search.ipynb` and `text-classification.ipynb` to build embeddings and classify categories.
- **Sentiment Analysis:**
  - Run `sentiment-analysis.ipynb` to add emotion scores.
- **Dashboard:**
  - Launch the Gradio interface:
    ```bash
    python Gradio-Dashboard.py
    ```

## Usage
- Enter a book description or theme in the search box.
- Filter results by category (e.g., Fiction, Non-fiction).
- Sort recommendations by emotional tone (e.g., Happy, Sad, Suspenseful).
- View book covers, titles, authors, and summaries in a visual gallery.

## Repository Structure
| File/Folder               | Description                                      |
|---------------------------|--------------------------------------------------|
| `Books_Cleanned.csv`      | Cleaned book dataset                             |
| `books_with_categories.csv` | Dataset with classified categories             |
| `books_with_emotions.csv` | Dataset with sentiment scores                    |
| `tagged_description.txt`  | Text file for vector embedding                   |
| `data-exploration.ipynb`  | Data cleaning and EDA notebook                   |
| `Vector-search.ipynb`     | Embedding and vector search notebook             |
| `text-classification.ipynb` | Zero-shot classification notebook              |
| `sentiment-analysis.ipynb` | Sentiment analysis notebook                     |
| `Gradio-Dashboard.py`     | Main file to launch the dashboard                |
| `cover-not-found.jpg`     | Placeholder image for missing book covers        |

## Technologies Used
- **Python** (Jupyter, scripts)
- **Pandas, NumPy** (data processing)
- **Matplotlib, Seaborn** (visualization)
- **OpenAI, Hugging Face Transformers** (LLMs, embeddings, classification)
- **Chroma** (vector database)
- **Gradio** (interactive dashboard)
- **Kaggle** (dataset source)

## Customization
- Swap in different LLMs for embeddings or classification.
- Add more categories or emotion labels as needed.
- Extend the dashboard with user authentication or additional filters.

## Credits
- Inspired by modern NLP and recommendation system best practices.
- Built with open-source tools and datasets.
- See the repository for full acknowledgments.

## License
This project is open-source and available under the MIT License.

## Contact
For questions or contributions, please open an issue or pull request on the [GitHub repository](https://github.com/Yashshah0502/BookRecommender).

Enjoy discovering your next favorite book!
