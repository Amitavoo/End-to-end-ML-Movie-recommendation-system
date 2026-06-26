# 🎬 AI Movie Recommendation System

An end-to-end AI-powered Movie Recommendation System built using **Machine Learning**, **FastAPI**, **TMDB API**, and **Streamlit**. The application combines a content-based recommendation engine with real-time movie metadata to provide intelligent movie suggestions along with posters, genres, release dates, and detailed information.

---

## 📌 Project Overview

This project recommends movies using a **TF-IDF based Content-Based Filtering** approach. A user searches for a movie title, and the system returns:

* 🎥 Movie Details
* 🤖 Similar Movies using Machine Learning
* 🎭 Genre-based Recommendations
* 🖼️ High-quality Posters from TMDB
* 📅 Release Dates
* 📖 Movie Overview

The recommendation engine is served through a **FastAPI backend**, while the user interface is built using **Streamlit**.

---

# 🚀 Features

* Content-Based Movie Recommendation using TF-IDF
* Cosine Similarity based Recommendation Engine
* FastAPI REST API Backend
* TMDB API Integration
* Real-time Movie Search
* Movie Posters and Backdrops
* Genre-based Recommendations
* Interactive Streamlit Frontend
* Modular Project Structure
* Pickle-based Model Serialization

---

# 🛠 Tech Stack

### Backend

* FastAPI
* Pydantic
* HTTPX

### Machine Learning

* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity
* NumPy
* Pandas

### Frontend

* Streamlit

### APIs

* TMDB API

### Others

* Python
* Pickle
* dotenv

---

# 🧠 Machine Learning Pipeline

```
Movie Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Combined Text Features
      │
      ▼
TF-IDF Vectorization
      │
      ▼
TF-IDF Matrix
      │
      ▼
Cosine Similarity
      │
      ▼
Top N Recommended Movies
```

---

# 🏗 System Architecture

```
                  User
                    │
                    ▼
            Streamlit Frontend
                    │
              HTTP Requests
                    │
                    ▼
              FastAPI Backend
                    │
        ┌───────────┴────────────┐
        │                        │
        ▼                        ▼
 TF-IDF Recommendation      TMDB API
        │                        │
        └───────────┬────────────┘
                    ▼
            Combined Response
                    │
                    ▼
             Streamlit UI
```

---

# 📂 Project Structure

```
Movie-Recommendation-System/
│
├── app.py                          # Streamlit Frontend
├── main.py                         # FastAPI Backend
├── end_to_end_movie_recommendation_system.py
│                                   # ML Pipeline
│
├── df.pkl                          # Movie DataFrame
├── indices.pkl                     # Title → Index Mapping
├── tfidf.pkl                       # Trained TF-IDF Vectorizer
├── tfidf_matrix.pkl                # TF-IDF Feature Matrix
│
├── requirements.txt
├── README.md
├── .gitignore
├── .env                            # TMDB API Key (Not committed)
│
└── __pycache__/
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Movie-Recommendation-System.git

cd Movie-Recommendation-System
```

Create Virtual Environment

```bash
python -m venv .venv
```

Activate

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file

```env
TMDB_API_KEY=YOUR_TMDB_API_KEY
```

Get your API key from

https://developer.themoviedb.org/

---

# ▶ Running the Backend

```bash
uvicorn main:app --reload
```

FastAPI Documentation

```
http://127.0.0.1:8000/docs
```

---

# ▶ Running the Frontend

```bash
streamlit run app.py
```

---

# 📡 API Endpoints

| Method | Endpoint         | Description               |
| ------ | ---------------- | ------------------------- |
| GET    | /health          | Health Check              |
| GET    | /home            | Trending / Popular Movies |
| GET    | /tmdb/search     | Search Movies             |
| GET    | /movie/id/{id}   | Movie Details             |
| GET    | /recommend/tfidf | ML Recommendations        |
| GET    | /recommend/genre | Genre Recommendations     |
| GET    | /movie/search    | Combined Search Endpoint  |

---

# 🧮 Recommendation Algorithm

The recommendation engine uses **TF-IDF Vectorization** to convert textual movie features into numerical vectors.

The similarity between movies is computed using **Cosine Similarity**.

```
Movie A
      │
TF-IDF Vector
      │
Cosine Similarity
      │
Movie B
Movie C
Movie D
...
```

The movies with the highest similarity scores are returned as recommendations.

---

# 📈 Future Improvements

* User Authentication
* Collaborative Filtering
* Hybrid Recommendation System
* Deep Learning Based Recommendations
* User Watchlist
* Movie Ratings
* Deployment on Railway / Render
* Docker Support
* CI/CD Pipeline
* PostgreSQL Integration

---

# 👨‍💻 Author

**Amitava Mondal**

B.Tech Computer Science Engineering (AI)

Machine Learning | Deep Learning | Full Stack Development | FastAPI | MERN Stack

---

# 📜 License

This project is intended for educational and portfolio purposes.
