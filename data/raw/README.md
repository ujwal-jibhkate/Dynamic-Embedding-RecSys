# Raw Data

This directory is for the raw datasets required to run the project notebooks. Due to their size, these files are not committed to the repository. Please download them from Kaggle using the links below and place them in this folder.

---

### 1. The Movies Dataset

This dataset provides the essential user ratings, movie IDs, and timestamps needed to create user interaction histories.

* **Source:** [The Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
* **Files to Download:**
    * `ratings_small.csv`
    * `links_small.csv`
* **Action:** Place these two `.csv` files directly inside this `data/raw/` folder.

---

### 2. IMDB Multimodal Dataset

This dataset provides the core movie metadata, including titles, plot summaries, genres, and poster images.

* **Source:** [IMDB Multimodal Vision & NLP on Kaggle](https://www.kaggle.com/datasets/zulkarnainsaurav/imdb-multimodal-vision-and-nlp-genre-classification)
* **Files to Download:** Download the entire dataset archive (`.zip` file).
* **Action:** Unzip the archive. You will need the main CSV file (e.g., `IMDB_Movies.csv`) and the folder containing the poster images (e.g., `posters`). Place the CSV file and the poster image folder inside this `data/raw/` directory.

---

### Note on Kaggle API

For automated workflows, you can use the official [Kaggle API](https://www.kaggle.com/docs/api) to download these datasets directly from your terminal or notebook.
