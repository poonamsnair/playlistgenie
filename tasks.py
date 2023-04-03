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
    
    # Use a generator to stream the tracks in batches
    batch_size = 50
    track_batches = iter([playlist['tracks'][i:i+batch_size] for i in range(0, len(playlist['tracks']), batch_size)])
    
    valid_ratings = {}
    audio_features = []
    for batch in track_batches:
        # Only get audio features for tracks that have been rated
        batch_valid_keys = [t['track']['id'] for t in batch if t['track']['id'] in ratings]
        batch_audio_features = [feature for feature in sp.audio_features(batch_valid_keys) if feature is not None]
        batch_valid_ratings = {key: ratings[key] for key in batch_valid_keys}

        for feature in batch_audio_features:
            valid_ratings[feature['id']] = batch_valid_ratings[feature['id']]
            audio_features.append(feature)

    if len(audio_features) <= 1:
        return None

    playlist_df = pd.DataFrame(audio_features)
    playlist_df['ratings'] = list(valid_ratings.values())

    X = playlist_df[["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key",
                     "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]]
    y = playlist_df['ratings']

    scaler = MinMaxScaler()
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
        X_scaled = np.empty_like(X)
        for i in range(0, len(X), 1000):
            X_scaled[i:i+1000] = scaler.fit_transform(X[i:i+1000])
        grid_search.fit(X_scaled, y)
    except Exception as e:
        return None

    knn = grid_search.best_estimator_

    rec_track_ids = set()
    recommendation_limit = ceil(len(playlist_df) / 2)

    track_batches = iter([playlist['tracks'][i:i+batch_size] for i in range(0, len(playlist['tracks']), batch_size)])
    for batch in track_batches:
        for track in batch:
            track_id = track['track']['id']
            try:
                rec_tracks = sp.recommendations(seed_tracks=[track_id], limit=recommendation_limit)['tracks']
                for rec_track in rec_tracks:
                    rec_track_ids.add(rec_track['id'])
            except Exception as e:
                continue

    track_chunks = [list(rec_track_ids)[i:i+100] for i in range(0, len(rec_track_ids), 100)]

    for track_chunk in track_chunks:
        sp.user_playlist_add_tracks(user=spotify_username, playlist_id=rec_playlist_id, tracks=track_chunk)

    return f"https://open.spotify.com/playlist/{rec_playlist_id}"

q = Queue(connection=Redis())