import os
import pickle 
from typing import Optional, List, Dict, Any, Tuple
import numpy as np 
import pandas as pd 
import httpx  
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")

app=FastAPI(
    title="Movie Recommender API",
    version="0.1.0",
 )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# PATH and GLOBAL VARS CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDEX_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "data", "tfidf.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "data", "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf: Any = None
tfidf_matrix: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None

class TMDBMovieCard(BaseModel):
    tmdb_id: int 
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []

class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: str
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float 
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]   

# UTILITY FUNCTIONS
def _norm_title(t: str)-> Optional[str]:
    if not t:
        return None
    return t.lower().strip()

def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"     

async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=500, detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        ) 

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"TMDB error {r.status_code}: {r.text}"
        )

    return r.json()

async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
)-> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for movie in results[:limit]:
        card = TMDBMovieCard(
            tmdb_id=movie.get("id"),
            title=movie.get("title"),
            poster_url=make_img_url(movie.get("poster_path")),
            release_date=movie.get("release_date"),
            genres=movie.get("genre_ids", []),
        )
        out.append(card)
    return out

async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=data.get("id"),
        title=data.get("title") or "",
        overview=data.get("overview") or "",
        release_date=data.get("release_date") or "",
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )

async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:

    """
    Raw TMDB response for keyword search (MULTIPLE results).
    Streamlit will use this for suggestions and grid.
    """
    return await tmdb_get("/search/movie", {"query": query, "language": "en-US", "page": page, "include_adult": "false"})

async def tmdb_search_first(query: str) -> Optional[Dict[str, Any]]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None

def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be;
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """
    try:
        title_to_idx: Dict[str, int] = {}
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        #last resort: if it's a list-like etc.
        raise RuntimeError(
            "indices.pkl must be a dict or pandas Series-like (with .items())"
        )
          
def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TITLE_TO_IDX not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Movie title not found in local index: '{title}'"
    )

from sklearn.metrics.pairwise import cosine_similarity

def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
)  -> List[Tuple[str, float]]:
    """
    Return top_n recommendations based on TFIDF similarity.
    Returns list of (title, score) tuples.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TFIDF model not initialized")

    idx = get_local_idx_by_title(query_title)


    # Get pairwise similarity scores
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Sort descending and get indices
    order = np.argsort(scores)[::-1]

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break

    return out

async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Given a movie title, search TMDB and return a TMDBMovieCard.
    Returns None if not found.
    """
    try:
        m = await tmdb_search_first(title)
        if m is None:
            return None
        card = TMDBMovieCard(
            tmdb_id=m.get("id"),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            genres=m.get("genre_ids", []),
        )
        return card
    except Exception:
        return None

@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf, tfidf_matrix, TITLE_TO_IDX

    # Load pickles
    try:
        with open(DF_PATH, "rb") as f:
            df = pickle.load(f)
        with open(INDEX_PATH, "rb") as f:
            indices_obj = pickle.load(f)
        with open(TFIDF_PATH, "rb") as f:
            tfidf = pickle.load(f)
        with open(TFIDF_MATRIX_PATH, "rb") as f:
            tfidf_matrix = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading pickles: {repr(e)}")

    # Build title to index map
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    #sanity 
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must be a pandas DataFrame with a 'title' column")

#Routes
@app.get("/health")
def health():
    return {"status":"ok"}
#Home feed route for Streamlit (posters)
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular")
    ,limit: int = Query(24, ge=1, le=50)
):
    """
    Home feed for Streamlit (posters).
    category:
        -trending (trending/movie/day)
        -popular, top_rated, upcoming, now_playing (movie/{category})
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language":"en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit) 

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        data = await tmdb_get(f"/movie/{category}", {"language":"en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching home feed: {repr(e)}")

#Search multiple results via keyword
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Search TMDB for movies by keyword.
    Returns multiple results (TMDBMovieCard).
    -dropdown suggestions 
    -grid results
    """ 
    return await tmdb_search_movies(query=query, page=page)
 
#Movie details 
@app.get("/movie/id/{tmdb_id}", response_model= TMDBMovieDetails)
async def movie_details(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)

# Genre recommendations 
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(..., ge=1),
    limit: int = Query(18, ge=1, le=50),
):
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity_desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]

@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50)
) -> List[Dict[str, Any]]:
    recs = tfidf_recommend_titles(title, top_n=limit)
    return [{"title": t, "score": s} for t, s in recs] 

@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_limit: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30)
):
    """
    This endpoint is for when you  have a selected movie and want:
    -movie details 
    - TFIDF recommendations(local) + posters
    - Genre recommendations (TMDB) + posters

    NOTE:
    -It selects the BEST match from TMDB for the given query.
    -If you want Multiple matches, use /tmdb/search
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )
    
    tmdb_id = int(best["id"]) 
    # This is where the rest of your endpoint logic would go.
    # The file seems to be incomplete here.
    # For now, returning a placeholder to avoid a syntax error.
    return {"query": query, "movie_details": {}, "tfidf_recommendations": [], "genre_recommendations": []}