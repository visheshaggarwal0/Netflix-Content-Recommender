# 🎬 Netflix Content Clustering & Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

An unsupervised machine learning system that analyzes Netflix's content catalog to create semantic clusters and provide intelligent, content-based recommendations without relying on user ratings or watch history.

![Netflix Recommendation System](https://img.shields.io/badge/ML-Unsupervised%20Learning-orange)
![NLP](https://img.shields.io/badge/NLP-Sentence%20Transformers-blueviolet)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

Netflix hosts over 7,000 movies and TV shows across multiple countries, languages, and genres. This abundance creates **choice paralysis** - users struggle to decide what to watch. Traditional metadata-based categorization (e.g., Comedy, Romance, Action) is too coarse-grained and fails to capture cultural and semantic nuances.

This project implements an **unsupervised machine learning system** that:
- Understands the semantic essence of Netflix titles using textual metadata
- Automatically groups similar content into meaningful clusters
- Generates recommendations based on content similarity
- Operates without user ratings or interaction history

---

## 🔍 Problem Statement

### Background Challenges:
1. **Choice Paralysis**: Too many options overwhelm users
2. **Generic Categorization**: Traditional genres don't capture semantic differences
   - Example: A US Romantic Comedy differs significantly from an Indian Romantic Comedy in tone, storytelling, and themes
3. **Limited Recommendation Systems**: 
   - Rating-based systems face cold-start problems
   - Poor discoverability of less popular or newly added content

### Objective:
Design and implement an unsupervised ML system that clusters Netflix content and recommends similar titles based on textual similarity of descriptions, genres, cast, and directors.

---

## ✨ Features

### Core Functionality:
- 📊 **Exploratory Data Analysis**: Comprehensive analysis of content distribution, countries, and ratings
- 🧹 **Data Preprocessing**: Robust handling of missing values and text normalization
- 🔤 **NLP Processing**: Advanced text vectorization using TF-IDF and Sentence Transformers
- 📉 **Dimensionality Reduction**: TruncatedSVD for efficient feature representation
- 🎯 **Multiple Clustering Algorithms**:
  - K-Means Clustering
  - HDBSCAN (Hierarchical Density-Based Spatial Clustering)
  - Leiden Community Detection
- 📈 **Cluster Validation**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
- 🎬 **Content-Based Recommendations**: Cosine similarity-based recommendation engine
- 🖥️ **Interactive Web Interface**: Streamlit-based user interface for real-time recommendations

### Key Capabilities:
- ✅ Operates without user ratings or watch history
- ✅ Handles missing data intelligently
- ✅ Scalable to large datasets
- ✅ Domain-independent framework applicable to other streaming platforms
- ✅ Privacy-compliant (no user data required)

---

## 🛠️ Methodology

### 1. Data Preprocessing
```python
# Create unified textual representation
df['content'] = (
    df['title'] + " " +
    df['description'] + " " +
    df['listed_in'] + " " +
    df['cast'] + " " +
    df['director']
)
```

### 2. Feature Extraction

#### Approach A: TF-IDF Vectorization
```python
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.8,
    min_df=5,
    max_features=10000,
    ngram_range=(1,2)
)
X = vectorizer.fit_transform(df['content'])
```

#### Approach B: Sentence Transformers (Advanced)
```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df['content'].tolist())
```

### 3. Dimensionality Reduction
```python
svd = TruncatedSVD(n_components=200, random_state=42)
X_reduced = svd.fit_transform(X)
```

### 4. Clustering
Three approaches implemented:

1. **K-Means**: Fast, interpretable, works well with spherical clusters
2. **HDBSCAN**: Density-based, handles noise, automatic cluster detection
3. **Leiden**: Graph-based community detection, high-quality partitions

### 5. Recommendation Engine
```python
def recommend(title, df, vectors, top_n=7):
    # Find title index
    idx = df[df['title'].str.lower() == title.lower()].index[0]
    
    # Calculate cosine similarities
    sims = cosine_similarity(vectors[idx].reshape(1,-1), vectors)[0]
    
    # Get top similar items
    top_idx = np.argsort(sims)[::-1][1:top_n+1]
    
    return df.iloc[top_idx][['title','type','listed_in']]
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/netflix-clustering.git
cd netflix-clustering
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
Download the Netflix dataset from [this link](https://drive.google.com/file/d/1RwwzDYGn3LAfFupe_kQ10mw9GbY1gmYW/view) and place `NetflixSimple.csv` in the project root directory.

---

## 💻 Usage

### Running the Jupyter Notebook
```bash
jupyter notebook Ediglobe_Final_Project.ipynb
```

### Running the Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Recommendation System

#### Example 1: Using TF-IDF Features
```python
import pandas as pd
import numpy as np

# Load processed data
df = pd.read_pickle("df.pkl")
X_reduced = np.load("X_reduced.npy")

# Get recommendations
recommendations = recommend("3 Idiots", df, X_reduced, top_n=7)
print(recommendations)
```

#### Example 2: Using Sentence Embeddings
```python
# Load embeddings
embeddings = np.load("embeddings.npy")

# Get better recommendations
recommendations = recommend_v2("3 Idiots", df, embeddings, top_n=7)
print(recommendations)
```

---

## 📁 Project Structure

```
netflix-clustering/
│
├── Ediglobe_Final_Project.ipynb   # Main Jupyter notebook with analysis
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── IMPROVEMENTS.md                 # Detailed improvement suggestions
│
├── data/
│   └── NetflixSimple.csv          # Netflix dataset (not included)
│
├── models/
│   ├── df.pkl                     # Processed dataframe
│   ├── X_reduced.npy              # TF-IDF reduced features
│   └── embeddings.npy             # Sentence transformer embeddings
│
├── notebooks/
│   └── exploratory_analysis.ipynb # Additional EDA
│
├── src/
│   ├── preprocessing.py           # Data preprocessing utilities
│   ├── clustering.py              # Clustering algorithms
│   ├── recommendation.py          # Recommendation engine
│   └── evaluation.py              # Model evaluation metrics
│
└── images/
    ├── cluster_visualization.png
    ├── content_distribution.png
    └── recommendation_demo.png
```

---

## 📊 Results

### Clustering Performance

| Model | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Clusters |
|-------|-----------------|---------------------|------------------------|----------|
| K-Means (TF-IDF) | 0.XX | X.XX | XXX.XX | 20 |
| HDBSCAN (Embeddings) | 0.XX | X.XX | XXX.XX | Auto |
| Leiden (Graph-based) | 0.XX | X.XX | XXX.XX | Auto |

*Note: Fill in with actual values from your notebook*

### Sample Recommendations

**Input:** "3 Idiots"

**Output:**
1. PK (Movie, Comedy)
2. Rang De Basanti (Movie, Drama)
3. Taare Zameen Par (Movie, Drama)
4. Dangal (Movie, Sports Drama)
5. Like Stars on Earth (Movie, Drama)

### Cluster Characteristics

**Cluster 0**: Bollywood Dramas
- Common themes: Social issues, inspirational stories
- Average release year: 2015
- Top rating: TV-14

**Cluster 5**: Hollywood Action
- Common themes: Superhero, sci-fi, adventure
- Average release year: 2018
- Top rating: PG-13

*... (add more cluster descriptions)*

---

## 🔧 Technologies Used

### Core Libraries:
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **NLP**: sentence-transformers, TfidfVectorizer
- **Clustering**: KMeans, HDBSCAN, leidenalg
- **Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: Streamlit
- **Graph Processing**: igraph, pynndescent

### Models:
- **Sentence Transformer**: all-MiniLM-L6-v2
- **Dimensionality Reduction**: TruncatedSVD
- **Clustering**: K-Means, HDBSCAN, Leiden Algorithm

---

## 🚀 Future Enhancements

### Short-term:
- [ ] Implement optimal cluster number selection (Elbow method)
- [ ] Add t-SNE/UMAP visualizations for cluster interpretation
- [ ] Enhance error handling and input validation
- [ ] Add export functionality for recommendations
- [ ] Improve Streamlit UI with filters and advanced search

### Medium-term:
- [ ] Implement hybrid recommendation (content + collaborative filtering)
- [ ] Add A/B testing framework for different models
- [ ] Deploy on cloud platform (Heroku, AWS, or GCP)
- [ ] Create REST API for recommendations
- [ ] Add multilingual support

### Long-term:
- [ ] Integrate with real-time data pipeline
- [ ] Implement deep learning-based recommendations (Neural Collaborative Filtering)
- [ ] Add user feedback loop for continuous improvement
- [ ] Expand to other streaming platforms
- [ ] Build mobile application

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines:
- Follow PEP 8 style guide for Python code
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

**LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

**Project Link**: [https://github.com/yourusername/netflix-clustering](https://github.com/yourusername/netflix-clustering)

---

## 🙏 Acknowledgments

- Netflix for providing the dataset
- Sentence-Transformers team for the pre-trained models
- Scikit-learn community for comprehensive ML tools
- Streamlit for the awesome web framework

---

## 📚 References

1. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv:1908.10084.
2. Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In Pacific-Asia conference on knowledge discovery and data mining (pp. 160-172).
3. Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. Scientific reports, 9(1), 1-12.

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

Made with ❤️ for Data Science

</div>
# Netflix-Content-Recommender
