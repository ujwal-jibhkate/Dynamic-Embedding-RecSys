# Dynamic Multimodal Recommendation System

![Python 3.9](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Research_Prototype-informational.svg)

**A novel recommendation engine that understands movies and users on a deeper level by fusing visual and textual data into dynamic, evolving embeddings.**

This repository contains the complete research prototype for a recommendation system designed to tackle the classic "cold-start" problem. Instead of relying on user interaction history alone, this system builds a rich "Digital DNA" for every movie and a "Living Taste Profile" for every user, enabling meaningful recommendations from the very first interaction.

---

## 📖 Table of Contents
* [The Big Idea](#-the-big-idea)
* [🌟 Key Features](#-key-features)
* [🏗️ System Architecture: A Deeper Dive](#️-system-architecture-a-deeper-dive)
* [🛠️ Tech Stack](#️-tech-stack)
* [🚀 Getting Started](#-getting-started)
* [🔬 Reproducing the Results](#-reproducing-the-results)
* [📈 Evaluation & Key Insights](#-evaluation--key-insights)
* [🔮 Future Work](#-future-work)
* [📜 License](#-license)
* [📧 Contact](#-contact)

---

## 💡 The Big Idea

How do you recommend a movie to a new user with no viewing history? This is the "cold-start" problem, a major hurdle for traditional recommendation systems. My hypothesis was that we could solve this by understanding the *intrinsic properties* of movies and users, not just their interactions.

This project explores a framework to create:
1.  A **fixed, multimodal embedding** for every movie, capturing its genre, visual tone, and creative style.
2.  A **dynamic, evolving embedding** for every user, which adapts in real-time as they interact with new content.

By modeling users and items in this sophisticated way, the system can make intelligent recommendations even with minimal data, effectively turning a "cold start" into a "warm start."

---

## 🌟 Key Features

-   🧠 **Multimodal Embeddings**: Creates a rich "Digital DNA" for each movie by fusing textual data (plots, cast, crew) and visual data (poster art) using **CLIP** and **Sentence-BERT**.
-   🎯 **Multi-Task Metric Learning**: Employs a **multi-triplet loss** function that forces the model to learn a semantically rich embedding space where movies are simultaneously close to others with the same genre, director, and actors.
-   👤 **Dynamic User Profiles**: A lightweight network with **residual connections** models the *delta* (change) in a user's taste, allowing their profile to evolve smoothly without forgetting core preferences.
-   ⚙️ **Production-Oriented Pipeline**: Implements a scalable **Two-Stage (Retrieval-then-Rank)** architecture.
    -   **Stage 1: Retrieval**: Uses **FAISS** (Facebook AI Similarity Search) for sub-second retrieval of hundreds of relevant candidates from a catalog of 44,000+ movies.
    -   **Stage 2: Ranking**: Uses **Maximal Marginal Relevance (MMR)** to re-rank candidates, striking an optimal balance between accuracy and diversity.
-   📊 **Interactive 3D Visualizations**: Leverages Plotly to create explorable 3D plots of the learned movie and user embedding spaces, offering clear insights into the model's behavior.

---

## 🏗️ System Architecture: A Deeper Dive

The system is designed as a modular pipeline, where the output of each stage serves as the input for the next.



**Phase 1: Crafting the Movie "Digital DNA" (Content Embedding)**
-   **Input**: Raw movie data (metadata, plots, poster images).
-   **Process**: Pre-trained Transformer models (CLIP for images, S-BERT for text) generate initial embeddings. A custom **PyTorch MLP** is then trained with the multi-triplet loss to fuse these into a single, fixed vector for each movie.
-   **Output**: A master embedding matrix where each row is a unique movie's semantic fingerprint.

**Phase 2: Modeling the User "Taste Profile" (Dynamic Embedding)**
-   **Input**: The movie embedding matrix and user interaction data.
-   **Process**: A user's profile is initialized by averaging the embeddings of their favorite movies. A dedicated network with residual connections is trained to predict the change to this profile after each new movie interaction.
-   **Output**: An evolving embedding vector for each user that represents their current tastes.

**Phase 3: The Recommendation Pipeline (Retrieval & Ranking)**
-   **Input**: A target user's "Taste Profile" vector.
-   **Process**:
    1.  **Candidate Retrieval**: FAISS performs an efficient similarity search to instantly retrieve the top 100 most relevant movies from the master embedding matrix.
    2.  **Fine-Grained Ranking**: A Two-Tower model projects the user and candidate movies into a shared latent space to calculate fine-tuned relevance scores.
    3.  **Re-Ranking for Diversity**: The MMR algorithm re-orders the top candidates to ensure the final list isn't monotonous, balancing relevance with novelty.
-   **Output**: A final, ordered list of top-10 movie recommendations.

---

## 🛠️ Tech Stack

-   **Core ML Framework**: PyTorch
-   **Embeddings & NLP**: Hugging Face Transformers (Sentence-BERT, CLIP)
-   **Similarity Search**: FAISS (Facebook AI)
-   **Data Manipulation**: Pandas, NumPy, Scikit-learn
-   **Visualization**: Plotly, Matplotlib, Seaborn
-   **Environment**: Jupyter Notebooks, Python 3.9+

---

## 🚀 Getting Started

Follow these steps to set up the environment and run the project on your local machine.

### Prerequisites
-   Python 3.9 or higher
-   `pip` and `venv`

### Installation
Clone the repository and install the dependencies in a virtual environment.

```bash
# 1. Clone the repository
git clone [https://github.com/ujwal-jibhkate/Dynamic-Embedding-RecSys.git](https://github.com/ujwal-jibhkate/Dynamic-Embedding-RecSys.git)
cd Dynamic-Embedding-RecSys

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# 3. Install all required packages
pip install -r requirements.txt
````

### Data Setup

The primary movie datasets are sourced from Kaggle. Please follow the instructions in the `data/README.md` file to download and place the required CSV files.

-----

## 🔬 Reproducing the Results

The project is organized into a sequence of Jupyter notebooks. To reproduce the entire pipeline from scratch, please run them in the specified order.

1.  **`notebooks/1_Content_Embedding_Generation.ipynb`**
      * *Goal*: Trains the fusion network and generates the fixed multimodal embeddings for all movies.
2.  **`notebooks/2_User_Embedding_Generation.ipynb`**
      * *Goal*: Trains the user profile updater network and generates the dynamic taste vectors for all users.
3.  **`notebooks/3_Ranking_and_Evaluation.ipynb`**
      * *Goal*: Trains the Two-Tower ranking model and conducts a comprehensive evaluation of the final recommendation pipeline.
4.  **`notebooks/4_Visualizations.ipynb`**
      * *Goal*: Generates the interactive 3D plots to explore the learned embedding spaces.

-----

## 📈 Evaluation & Key Insights

The system's primary goal is to balance **Relevance** (accuracy) and **Diversity** (novelty).

  - **Relevance Metrics**: Precision@k, Recall@k, **nDCG@k** (primary)
  - **Diversity Metrics**: **Intra-List Diversity (ILD)**, Catalog Coverage

The key to balancing these competing goals lies in tuning the `lambda` parameter of the MMR re-ranker.

### The Relevance vs. Diversity Trade-off

As we increase `lambda`, the system prioritizes relevance over diversity. The sweet spot was identified at `lambda = 0.5`, which provides a **significant boost in diversity** with only a minor drop in relevance.

| Lambda (λ) | NDCG@10 (Relevance) | ILD@10 (Diversity) | Coverage |
| :--------: | :-----------------: | :----------------: | :------: |
|   `0.7`    |      `0.0664`       |      `0.0261`      |  `2.17%` |
|   `0.5`    |      `0.0476`       |    **`0.0387`** | **`4.49%`** |
|   `0.3`    |      `0.0470`       |      `0.0393`      |  `3.03%` |

### Visual Proof of Concept

**1. The Learned Movie Embedding Space**
This 3D plot shows \~2000 movie embeddings, colored by genre. The clear formation of clusters (e.g., Action, Comedy) confirms that the multi-task model successfully learned a meaningful representation of movie similarity.

**2. A User's Evolving Taste Journey**
This plot traces a single user's embedding as it moves through the space with each new movie interaction, visually demonstrating the dynamic and adaptive nature of the user profiles.

-----

## 🔮 Future Work

  - **Explore More Modalities**: Incorporate audio embeddings from soundtracks or NLP analysis of subtitles.
  - **Advanced Sequential Models**: Replace the user updater FFN with a Transformer-based model (e.g., SASRec) to better capture the sequence and context of user interactions.
  - **End-to-End Training**: Combine the stages into a single, end-to-end trainable model for potentially better performance.
  - **Deployment & A/B Testing**: Deploy the model via a simple API (e.g., FastAPI) and conduct online A/B testing to measure real-world engagement metrics like click-through rate.

-----

## 📜 License

This project is licensed under the MIT License.

-----

## 📧 Contact

Ujwal Jibhkate - [LinkedIn](https://www.google.com/search?q=https-your-linkedin-url) - [Medium Article](https://medium.com/@ujwaljibhkate/from-clicks-to-connections-building-a-smarter-recommender-with-embeddings-cb0f8ca61aaf)

Project Link: [https://github.com/ujwal-jibhkate/Dynamic-Embedding-RecSys](https://github.com/ujwal-jibhkate/Dynamic-Embedding-RecSys)

