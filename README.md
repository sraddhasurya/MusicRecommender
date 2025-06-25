This project was an attempt to build a music recommendation system that combines Spotify audio features with Genius lyrics embeddings to suggest similar tracks based on both sound and sentiment. Unfortunately, due to Spotify API limitations, this project is no longer actively maintained. Still, it's a solid exploration of combining multiple APIs and machine learning models for intelligent recommendation.

Project Status: Abandoned — Spotify has deprecated or significantly restricted access to its /audio-features endpoint when using client credentials (the most common and scalable method). This endpoint was essential for retrieving the song-level metrics (e.g., tempo, danceability, energy) that this recommender relies on. As a result, the recommendation logic cannot function as originally intended.

Features:
- Spotify integration using Spotipy
- Genius lyrics scraping using LyricsGenius
- Lyrics preprocessing with nltk
- Embedding via sentence-transformers/all-MiniLM-L6-v2
- Cosine similarity between track embeddings

Tech Stack:
- Python 3.10+
- Spotipy
- LyricsGenius
- Transformers (HuggingFace)
- PyTorch
- nltk
- dotenv


Why It Stopped Working: As of mid-2025, Spotify has restricted access to song audio features unless you use OAuth scopes that require user login. This is infeasible for batch or backend recommendation systems. Even when properly authenticated, requests often result in 401/403 errors.

If Spotify updates its API access policies in the future, this project may be revived.

Lessons Learned: API-based ML pipelines are brittle if third-party services change access rules. Merging audio and lyric semantics offers a powerful hybrid approach to music similarity. OAuth vs. client credentials authentication models significantly impact the feasibility of automation.
