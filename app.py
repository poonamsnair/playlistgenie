import eventlet
from flask import Flask, redirect, request, session, url_for, render_template
from flask_socketio import SocketIO, emit
from flask_bootstrap import Bootstrap
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut, GroupShuffleSplit, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from functools import wraps
from itertools import zip_longest
from flask import flash
from werkzeug.security import generate_password_hash, check_password_hash
import stripe
import random
import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
from math import ceil
import threading
from functools import wraps
import uuid
from flask import jsonify
from flask import request, abort
from collections import OrderedDict


eventlet.monkey_patch()
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')
Bootstrap(app)
socketio = SocketIO(app)



SCOPE = 'user-library-read playlist-modify-public playlist-read-private'
SPOTIPY_REDIRECT_URI = os.environ.get('SPOTIPY_REDIRECT_URI')
SPOTIPY_CLIENT_ID = os.environ.get('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.environ.get('SPOTIPY_CLIENT_SECRET')


@app.errorhandler(Exception)
def handle_unhandled_exception(e):
    logging.exception('Unhandled exception: %s', e)
    error_code = getattr(e, 'code', 500)
    return render_template('error.html', error_code=error_code), error_code

def timeout(seconds=30, error_message='Function call timed out'):
    def decorator(func):
        def _handle_timeout():
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            timer = threading.Timer(seconds, _handle_timeout)
            timer.start()
            try:
                result = func(*args, **kwargs)
            finally:
                timer.cancel()
            return result

        return wraps(func)(wrapper)

    return decorator

def remove_duplicates(tracks):
    unique_tracks = OrderedDict()
    for track in tracks:
        track_id = track['track']['id']
        if track_id not in unique_tracks:
            unique_tracks[track_id] = track
    return list(unique_tracks.values())

def delete_playlist(spotify_token, playlist_id):
    sp = spotipy.Spotify(auth=spotify_token)
    user_id = sp.me()['id']
    sp.user_playlist_unfollow(user=user_id, playlist_id=playlist_id)

def inject_stripe_keys():
    if os.environ.get('APP_ENV', 'test') == 'production':
        publishable_key = os.environ.get('STRIPE_PUBLISHABLE_KEY')
        buy_button_id = os.environ.get('STRIPE_BUY_BUTTON_ID')
    else:
        publishable_key = 'pk_test_51MsPZqIpDJwsO3dTZhBli2IN4yPQgbAsxtj1AWnXWlH8rIvd2rdLEoefmBKXDCLeyPN3O9bDjKirR8VUgHS1zyFk00lbJ3ti7o'
        buy_button_id = 'buy_btn_1MsY4QIpDJwsO3dTMuyXa9KC'

    return {
        'stripe_publishable_key': publishable_key,
        'stripe_buy_button_id': buy_button_id,
    }

@app.context_processor
def inject_vars():
    vars = inject_stripe_keys()
    return vars

def get_spotify_client(access_token):
    client = spotipy.Spotify(auth=access_token)
    return client

def require_spotify_token(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get('spotify_token'):
            auth_manager = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET,
                                        redirect_uri=SPOTIPY_REDIRECT_URI, scope=SCOPE)
            auth_url = auth_manager.get_authorize_url()
            return render_template('index.html', auth_url=auth_url)
        else:
            refresh_token_if_expired() 
        return func(*args, **kwargs)
    return wrapper

def refresh_token_if_expired():
    if session.get('spotify_token_info'):
        auth_manager = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET,
                                    redirect_uri=SPOTIPY_REDIRECT_URI, scope=SCOPE)
        token_info = session.get('spotify_token_info')
        if auth_manager.is_token_expired(token_info):
            token_info = auth_manager.refresh_access_token(token_info['refresh_token'])
            session['spotify_token_info'] = token_info
            session['spotify_token'] = token_info['access_token']


@app.route('/')
def index():
    if session.get('spotify_token'):
        return redirect(url_for('playlists'))

    auth_manager = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET,
                                redirect_uri=SPOTIPY_REDIRECT_URI, scope=SCOPE)
    auth_url = auth_manager.get_authorize_url()
    return render_template('index.html', auth_url=auth_url)


@app.route('/logout')
def logout():
    session.pop('spotify_token', None)
    session.pop('spotify_username', None)
    session.pop('spotify_user_id', None)
    return redirect(url_for('index'))

@app.route('/callback/')
def callback():
    auth_manager = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET,
                                redirect_uri=SPOTIPY_REDIRECT_URI, scope=SCOPE)
    token_info = auth_manager.get_access_token(request.args.get('code'))
    session['spotify_token_info'] = token_info
    session['spotify_token'] = token_info.get('access_token')
    sp = spotipy.Spotify(auth=session['spotify_token'])
    user_data = sp.current_user()
    session['spotify_username'] = user_data['id']
    session['user'] = {
        'playlist_count': 0
    }
    return redirect(url_for('index'))


@app.route('/playlists/')
@require_spotify_token
def playlists():
    sp = spotipy.Spotify(auth=session['spotify_token'])

    if not session.get('spotify_user_id'):
        session['spotify_user_id'] = sp.current_user()['id']

    playlist_id = request.args.get('playlist_id')

    if playlist_id is None:
        playlists = sp.current_user_playlists()
        return render_template('playlist_list.html', playlists=playlists)
    else:
        playlist = sp.playlist(playlist_id)
        tracks = sp.playlist_tracks(playlist_id)['items']
        unique_tracks = remove_duplicates(tracks)
        unique_track_count = len(unique_tracks)
        return render_template('rate_playlists.html', playlist=playlist, tracks=unique_tracks, track_count=unique_track_count)

@app.route('/rate_playlist/<playlist_id>/', methods=['GET', 'POST'])
@require_spotify_token
def rate_playlist(playlist_id):
    if session.get('spotify_token'):
        try:
            sp = spotipy.Spotify(auth=session['spotify_token'])
            playlist = sp.playlist(playlist_id)
            tracks = playlist['tracks']['items']

            # Check if the user has already rated the tracks in this playlist
            if 'ratings' in session and playlist_id in session['ratings']:
                return redirect(url_for('recommendation', playlist_id=playlist_id))

            # Retrieve the track information and generate the Spotify URIs
            for track in tracks:
                track_info = sp.track(track['track']['id'])
                track['spotify_uri'] = track_info['uri']

            return render_template('rate_playlist.html', tracks=tracks, playlist_id=playlist_id)
        except Exception as e:  # Catch the exception as 'e'
            # Include the exception details in the error message
            return render_template('error.html', message=f'Failed to retrieve playlist information. Please try again later. Exception: {str(e)}')
    else:
        return redirect(url_for('index'))



@app.route('/save_ratings/<playlist_id>/', methods=['POST'])
@require_spotify_token
def save_ratings(playlist_id):
    if session.get('spotify_token'):
        sp = spotipy.Spotify(auth=session['spotify_token'])
        playlist = sp.playlist(playlist_id)
        tracks = playlist['tracks']['items']
        ratings = {}
        for track in tracks:
            track_id = track['track']['id']
            rating = request.form.get(f'rating-{track_id}')
            if rating:
                ratings[track_id] = int(rating)
        session['ratings'] = ratings
        session['playlist_id'] = playlist_id
        return redirect(url_for('create_playlist', playlist_id=playlist_id))
    else:
        return redirect(url_for('index'))
    
    
@app.route('/create_playlist/<playlist_id>/', methods=['GET', 'POST'])
@require_spotify_token
def create_playlist(playlist_id):
    if not session.get('spotify_token'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        playlist_name = request.form['playlist_name']
        if not playlist_name:
            return render_template('error.html', message="Playlist name cannot be empty.")
        elif len(playlist_name) > 100:
            return render_template('error.html', message="Playlist name is too long.")

        # Save the playlist name and ID in the session
        session['rec_playlist_name'] = playlist_name
        sp = spotipy.Spotify(auth=session['spotify_token'])
        user_id = sp.me()['id']
        rec_playlist = sp.user_playlist_create(user=user_id, name=playlist_name)
        rec_playlist_id = rec_playlist['id']
        session['rec_playlist_id'] = rec_playlist_id

        return redirect(url_for('recommendation', playlist_id=playlist_id, rec_playlist_id=rec_playlist_id))

    # If the request is a GET and there's a rec_playlist_id in the session, delete the playlist and remove rec_playlist_id from the session
    if request.method == 'GET' and 'rec_playlist_id' in session:
        delete_playlist(session['spotify_token'], session['rec_playlist_id'])
        session.pop('rec_playlist_id', None)

    return render_template('create_playlist.html', playlist_id=playlist_id)


@app.route('/recommendation/<playlist_id>/<rec_playlist_id>/')
@require_spotify_token
def recommendation(playlist_id, rec_playlist_id):
    if session.get('spotify_token'):
        spotify_token = session['spotify_token']
        ratings = session['ratings']
        spotify_username = session['spotify_username']
        request_id = str(uuid.uuid4())
        threading.Thread(target=background_recommendation, args=(playlist_id, rec_playlist_id, request_id, spotify_token, ratings, spotify_username)).start()
        return render_template("recommendation_progress.html", request_id=request_id)
    else:
        return redirect(url_for("index"))
    
def background_recommendation(playlist_id, rec_playlist_id, request_id, spotify_token, ratings, spotify_username):
    def emit_error_and_delete_playlist(request_id, message):
        delete_playlist(spotify_token, rec_playlist_id)
        socketio.emit("recommendation_error", {"request_id": request_id, "message": message}, namespace='/recommendation')
    sp = spotipy.Spotify(auth=spotify_token)
    playlist = sp.playlist(playlist_id)
    tracks = playlist['tracks']['items']
    if not ratings:
        return redirect(url_for('rate_playlist', playlist_id=playlist_id))

    track_ids = list(ratings.keys())

    # Retrieve audio features for only the tracks in the seed playlist that were rated by the user
    audio_features = sp.audio_features(track_ids)

    # Remove NoneType audio features
    audio_features = [feature for feature in audio_features if feature is not None]

    if len(audio_features) < 50:
        emit_error_and_delete_playlist(request_id, "Error: Less than 50 tracks")
    elif len(audio_features) > 100:
        emit_error_and_delete_playlist(request_id, "Error: More than 100 tracks")
    # Convert audio_features to a list of dictionaries
    playlist_data = []
    for feature in audio_features:
        feature_dict = {key: feature[key] for key in feature if key not in ['type', 'uri', 'track_href', 'analysis_url']}
        feature_dict['ratings'] = ratings[feature['id']]
        playlist_data.append(feature_dict)

    feature_keys = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]

    X = [[d[key] for key in feature_keys] for d in playlist_data]
    y = [d['ratings'] for d in playlist_data]

    scaler = MinMaxScaler()

    knn = KNeighborsClassifier()
    max_neighbors = min(30, len(X) - 1)

    param_grid = {
        'n_neighbors': range(1, max_neighbors + 1),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Choose the appropriate cross-validator
    n_splits = 5

    min_samples_per_class = min(np.bincount(y))
    if min_samples_per_class >= n_splits:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = LeaveOneOut()

    grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=cv, n_jobs=2,
                               pre_dispatch='2*n_jobs', scoring='accuracy')

    try:
        X_scaled = scaler.fit_transform(X)
        grid_search.fit(X_scaled, y)
    except Exception as e:
        emit_error_and_delete_playlist(request_id, "Error: Fit transformer error")
    rec_track_ids = set()
    for track_id in [d['id'] for d in playlist_data]:
        try:
            rec_tracks = sp.recommendations(seed_tracks=[track_id], limit=int(len(playlist_data)/2))['tracks']
            for track in rec_tracks:
                rec_track_ids.add(track['id'])
        except Exception as e:
            emit_error_and_delete_playlist(request_id, "Error: Adding tracks to playlist")

    if not rec_track_ids:
        emit_error_and_delete_playlist(request_id, "Error: No tracks found to be added")

    track_chunks = [list(rec_track_ids)[i:i+100] for i in range(0, len(rec_track_ids), 100)]

    for track_chunk in track_chunks:
        try:
            sp.user_playlist_add_tracks(user=spotify_username, playlist_id=rec_playlist_id, tracks=track_chunk)
        except Exception as e:
            logging.error(f"Error adding tracks to playlist: {str(e)}")
            emit_error_and_delete_playlist(request_id, f"Error adding tracks to playlist: {str(e)}")
            return
    socketio.emit("recommendation_done", {"request_id": request_id, "rec_playlist_id": rec_playlist_id}, namespace='/recommendation')
    
@socketio.on("connect", namespace="/recommendation")
def on_connect():
    pass


if __name__ == '__main__':
    env = os.environ.get('APP_ENV', 'test')

    if env == 'production':
        socketio.run(app, port=int(os.environ.get('PORT', 5000)))
    else:
        port = int(os.environ.get('PORT', 8888))
        socketio.run(app, port=port)