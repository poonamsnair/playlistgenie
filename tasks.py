from rq import Queue
from worker import conn
from flask_mail import Message
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from math import ceil
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, GridSearchCV
from redis import Redis

def send_email(to, playlist_url):
    msg = Message("Your Playlist Recommendations", sender="poonnnair@gmail.com", recipients=[to])
    msg.body = f"Here's the link to your recommended playlist: {playlist_url}"
    mail.send(msg)
    
def async_recommendation(user_email, *args, **kwargs):
    result = recommendation_function(*args, **kwargs)
    send_email(user_email, result)
    
def recommendation_function(playlist_id, rec_playlist_id, spotify_token, spotify_username, ratings):
    sp = spotipy.Spotify(auth=spotify_token)
    playlist = sp.playlist(playlist_id)
    tracks = playlist['tracks']['items']

    audio_features = [feature for feature in sp.audio_features(list(ratings.keys())) if feature is not None]

    valid_keys = [feature['id'] for feature in audio_features]
    valid_ratings = {key: ratings[key] for key in valid_keys}

    playlist_df = pd.DataFrame(audio_features)
    playlist_df['ratings'] = list(valid_ratings.values())

    X = playlist_df[["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key",
                     "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]]
    y = playlist_df['ratings']

    if len(X) <= 1:
        return None

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsClassifier()
    max_neighbors = min(30, len(X) - 1)

    param_grid = {
        'n_neighbors': range(1, max_neighbors + 1),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    n_splits = 5

    min_samples_per_class = min(np.bincount(y))
    if min_samples_per_class >= n_splits:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = LeaveOneOut()

    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=cv, n_jobs=-1)

    try:
        grid_search.fit(X_scaled, y)
    except Exception as e:
        return None

    knn = grid_search.best_estimator_

    rec_track_ids = set()
    recommendation_limit = ceil(len(playlist_df) / 2)

    for track_id in playlist_df['id'].tolist():
        try:
            rec_tracks = sp.recommendations(seed_tracks=[track_id], limit=recommendation_limit)['tracks']
            for track in rec_tracks:
                rec_track_ids.add(track['id'])
        except Exception as e:
            return None

    track_chunks = [rec_track_ids[i:i + 100] for i in range(0, len(rec_track_ids), 100)]

    for track_chunk in track_chunks:
        sp.user_playlist_add_tracks(user=spotify_username, playlist_id=rec_playlist_id, tracks=track_chunk)

    return f"https://open.spotify.com/playlist/{rec_playlist_id}"

q = Queue(connection=Redis())