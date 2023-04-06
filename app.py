import eventlet
from flask import Flask, redirect, request, session, url_for, render_template, send_from_directory
import secrets
import random
import string
import pandas as pd
import stripe
import random
import logging
import os
import numpy as np
import uuid
import threading
import spotipy
import base64
import hashlib
from flask_socketio import SocketIO, emit
from flask_bootstrap import Bootstrap
from spotipy.oauth2 import SpotifyOAuth
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
from sklearn.model_selection import train_test_split
from math import ceil
from functools import wraps
from flask import jsonify
from flask import request, abort
from collections import OrderedDict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from spotipy.exceptions import SpotifyException
from flask_mobility import Mobility
from flask_caching import Cache
from typing import List
from flask_session import Session
from base64 import urlsafe_b64encode, urlsafe_b64decode
from hashlib import sha256
from os import urandom

eventlet.monkey_patch()
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')
Bootstrap(app)
socketio = SocketIO(app)
Mobility(app)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)


SCOPE = 'user-library-read playlist-modify-public playlist-modify-private playlist-read-private streaming'
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

def refresh_token_if_expired():
    if session.get('spotify_token_info'):
        auth_manager = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET,
                                    redirect_uri=SPOTIPY_REDIRECT_URI, scope=SCOPE)
        token_info = session.get('spotify_token_info')
        if auth_manager.is_token_expired(token_info):
            token_info = auth_manager.refresh_access_token(token_info['refresh_token'])
            session['spotify_token_info'] = token_info
            session['spotify_token'] = token_info['access_token']   

def require_spotify_token(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        auth_manager = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=SCOPE, cache_path='.spotifycache')
        if not auth_manager.validate_token(session.get('spotify_token')):
            auth_url = auth_manager.get_authorize_url()
            return redirect(auth_url)
        else:
            refresh_token_if_expired(auth_manager)
        return func(*args, **kwargs)
    return wrapper

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def get_verifier_and_challenge():
    """Generate a code verifier and its SHA256 challenge."""
    verifier = urlsafe_b64encode(urandom(32)).decode('utf-8')
    challenge = urlsafe_b64encode(sha256(verifier.encode('utf-8')).digest()).decode('utf-8')
    return verifier, challenge
    
def get_spotify_client():
    auth_manager = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=SCOPE)
    return spotipy.Spotify(auth_manager=auth_manager)


@app.route('/logout/')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route("/")
def index():
    session.clear()
    code_verifier = base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8").rstrip("=")
    session["code_verifier"] = code_verifier
    code_challenge = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    code_challenge = base64.urlsafe_b64encode(code_challenge).decode("utf-8").rstrip("=")
    url = f"https://accounts.spotify.com/authorize?response_type=code&client_id={SPOTIPY_CLIENT_ID}&scope={SCOPE}&redirect_uri={SPOTIPY_REDIRECT_URI}&code_challenge_method=S256&code_challenge={code_challenge}&show_dialog=true"
    return redirect(url)


@app.route("/callback")
def callback():
    code = request.args.get("code")
    error = request.args.get("error")
    if error:
        return f"Error: {error}"
    code_verifier = session.pop("code_verifier", None)
    if not code_verifier:
        return "Error: Invalid state"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": SPOTIPY_REDIRECT_URI,
        "client_id": SPOTIPY_CLIENT_ID,
        "code_verifier": code_verifier,
    }
    auth_manager = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=SCOPE)
    token_info = auth_manager.get_access_token(code, code_verifier=code_verifier)
    session["spotify_token_info"] = token_info
    session["spotify_token"] = token_info['access_token']
    return redirect(url_for("playlists"))


def get_playlist_tracks(spotify_client, playlist_id):
    playlist = spotify_client.playlist(playlist_id)
    tracks = playlist['tracks']['items']
    return tracks

def paginate_playlists(playlists: List, limit: int, offset: int):
    start = offset
    end = offset + limit
    return playlists[start:end]

def remove_duplicates(tracks):
    unique_tracks = OrderedDict()
    for track in tracks:
        if track['track'] is not None:
            track_id = track['track']['id']
            if track_id not in unique_tracks:
                unique_tracks[track_id] = track
    return list(unique_tracks.values())


@app.route('/playlists/')
def playlists():
    if not session.get('token_info'):
        return redirect(url_for('index'))
    sp = get_spotify_client()
    limit = 12
    api_limit = 50
    playlist_id = request.args.get('playlist_id')
    offset = int(request.args.get('offset', 0))
    if playlist_id is None:
        raw_playlists = []
        api_offset = 0

        while len(raw_playlists) < offset + limit:
            print(f"Retrieving batch of playlists from offset {api_offset}...")
            playlists_batch = sp.current_user_playlists(limit=api_limit, offset=api_offset)
            if not playlists_batch['items']:
                print("No more playlists to retrieve.")
                break
            raw_playlists.extend(playlists_batch['items'])
            api_offset += api_limit

        unique_track_counts = {}
        playlist_images = {}
        filtered_playlists = []

        for playlist in raw_playlists:
            tracks = get_playlist_tracks(sp, playlist['id'])
            unique_tracks = remove_duplicates(tracks)
            count = len(unique_tracks)

            # Get playlist image
            if playlist['images']:
                playlist_images[playlist['id']] = playlist['images'][0]['url']
            else:
                playlist_images[playlist['id']] = None

            # Add the playlist to filtered_playlists if it has at least one unique track
            if count > 0:
                unique_track_counts[playlist['id']] = count
                filtered_playlists.append(playlist)

        paginated_playlists = paginate_playlists(filtered_playlists, limit, offset)
        previous_offset = max(offset - limit, 0)
        total_playlists = len(filtered_playlists)
        return render_template('playlist_list.html', playlists=paginated_playlists, unique_track_counts=unique_track_counts, playlist_images=playlist_images, offset=offset, previous_offset=previous_offset, total_playlists=total_playlists, limit=limit, request=request)
    else:
        if request.MOBILE:
            return redirect(url_for('mobile_rate_playlist', playlist_id=playlist_id))
        else:
            return redirect(url_for('rate_playlist', playlist_id=playlist_id))




# Add a decorator to handle rate limits
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10),
       retry=retry_if_exception_type(SpotifyException))
def get_playlist_tracks_with_retry(sp, playlist_id):
    return get_playlist_tracks(sp, playlist_id)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10),
       retry=retry_if_exception_type(SpotifyException))
def sp_track_with_retry(sp, track_id):
    return sp.track(track_id)

@app.route('/rate_playlist/<playlist_id>/', methods=['GET', 'POST'])
@require_spotify_token
def rate_playlist(playlist_id):
    if request.MOBILE:
        return redirect(url_for('mobile_rate_playlist', playlist_id=playlist_id))
    
    if session.get('spotify_token'):
        try:
            sp = spotipy.Spotify(auth=session['spotify_token'])

            tracks = get_playlist_tracks_with_retry(sp, playlist_id)
            unique_tracks = remove_duplicates(tracks)

            if 'ratings' in session and playlist_id in session['ratings']:
                return redirect(url_for('recommendation', playlist_id=playlist_id))

            for track in unique_tracks:
                track_info = sp_track_with_retry(sp, track['track']['id'])
                track['spotify_uri'] = track_info['uri']
                
            # Add the access token to the context
            return render_template('rate_playlist.html', tracks=unique_tracks, playlist_id=playlist_id, access_token=session['spotify_token'])
        except Exception as e:
            return render_template('error.html', message=f'Failed to retrieve playlist information. Please try again later. Exception: {str(e)}')
    else:
        return redirect(url_for('index'))


@app.route('/mobile_rate_playlist/<playlist_id>/', methods=['GET', 'POST'])
@require_spotify_token
def mobile_rate_playlist(playlist_id):
    if not request.MOBILE:
        return redirect(url_for('rate_playlist', playlist_id=playlist_id))
    
    if session.get('spotify_token'):
        try:
            sp = spotipy.Spotify(auth=session['spotify_token'])

            tracks = get_playlist_tracks_with_retry(sp, playlist_id)
            unique_tracks = remove_duplicates(tracks)

            if 'ratings' in session and playlist_id in session['ratings']:
                return redirect(url_for('recommendation', playlist_id=playlist_id))

            for track in unique_tracks:
                track_info = sp_track_with_retry(sp, track['track']['id'])
                track['spotify_uri'] = track_info['uri']

            # Add the access token to the context
            return render_template('mobile_rate_playlist.html', tracks=unique_tracks, playlist_id=playlist_id, access_token=session['spotify_token'])
        except Exception as e:
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
            # Set the default rating to 5 if no rating is given
            if rating:
                ratings[track_id] = int(rating)
            else:
                ratings[track_id] = 5
        session['ratings'] = ratings
        session['playlist_id'] = playlist_id
        session['from_save_ratings'] = True
        return redirect(url_for('create_playlist', playlist_id=playlist_id))
    else:
        return redirect(url_for('index'))
    
    
@app.route('/create_playlist/<playlist_id>/', methods=['GET', 'POST'])
@require_spotify_token
def create_playlist(playlist_id):
    if not session.get('spotify_token'):
        return redirect(url_for('index'))   
    if not session.get('from_save_ratings'):
        return redirect(url_for('index'))
    else:
        session.pop('from_save_ratings', None)
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
        socketio.emit("recommendation_error", {"request_id": request_id, "message": message}, namespace='/recommendation')
    sp = spotipy.Spotify(auth=spotify_token)
    playlist = sp.playlist(playlist_id)
    tracks = playlist['tracks']['items']
    if not ratings:
        return redirect(url_for('rate_playlist', playlist_id=playlist_id))
    track_ids = list(ratings.keys())

    # Retrieve audio features for only the tracks in the seed playlist that were rated by the user
    audio_features = sp.audio_features(track_ids)
    socketio.emit("playlist_data_processing", {"request_id": request_id}, namespace='/recommendation')

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
    
    socketio.emit("audio_features_retrieved", {"request_id": request_id}, namespace='/recommendation')
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

    socketio.emit("knn_model_trained", {"request_id": request_id}, namespace='/recommendation')
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
    socketio.emit("recommended_tracks_retrieved", {"request_id": request_id}, namespace='/recommendation')
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