import eventlet
from flask import Flask, redirect, request, session, url_for, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_bootstrap import Bootstrap
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut, GroupShuffleSplit, StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
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
from spotipy import Spotify
from flask import jsonify
from flask import request, abort
from collections import OrderedDict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from spotipy.exceptions import SpotifyException
from flask_mobility import Mobility
from flask_caching import Cache
from typing import List
from spotipy.cache_handler import CacheHandler
from flask_session import Session
import sys
import traceback
import time
import logging
logging.basicConfig(level=logging.DEBUG)

# Import Eventlet and apply monkey patching for better concurrency support
eventlet.monkey_patch()

# Create a new Flask web application using the current module's name as identifier
app = Flask(__name__)

# Set the secret key for the Flask app, used for securely signing session cookies
app.secret_key = os.environ.get('SECRET_KEY')

# Initialize the Bootstrap extension for easy integration with Flask app
Bootstrap(app)

# Initialize Flask-SocketIO for real-time communication between client and server
socketio = SocketIO(app)

# Add mobile support to the Flask app using Flask-Mobility extension
Mobility(app)

app.config['SECRET_KEY'] = os.urandom(64)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
Session(app)

caches_folder = './.spotify_caches/'
if not os.path.exists(caches_folder):
    os.makedirs(caches_folder)

def session_cache_path():
    uuid = session.get('uuid')
    if uuid is None:
        return None
    return caches_folder + uuid

# initalise spotify variables
SCOPE = 'user-library-read playlist-modify-public playlist-modify-private playlist-read-private streaming'
SPOTIPY_REDIRECT_URI = os.environ.get('SPOTIPY_REDIRECT_URI')
SPOTIPY_CLIENT_ID = os.environ.get('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.environ.get('SPOTIPY_CLIENT_SECRET')

# Return Stripe Keys
def inject_stripe_keys():
    # Check if the application environment is set to 'production'
    if os.environ.get('APP_ENV', 'test') == 'production':
        # Get Stripe publishable key from environment variables for production
        publishable_key = os.environ.get('STRIPE_PUBLISHABLE_KEY')
        # Get Stripe buy button ID from environment variables for production
        buy_button_id = os.environ.get('STRIPE_BUY_BUTTON_ID')
    else:
        # Use default test publishable key for non-production environments
        publishable_key = 'pk_test_51MsPZqIpDJwsO3dTZhBli2IN4yPQgbAsxtj1AWnXWlH8rIvd2rdLEoefmBKXDCLeyPN3O9bDjKirR8VUgHS1zyFk00lbJ3ti7o'
        # Use default test buy button ID for non-production environments
        buy_button_id = 'buy_btn_1MsY4QIpDJwsO3dTMuyXa9KC'

    # Return a dictionary containing the selected Stripe keys and IDs
    return {
        'stripe_publishable_key': publishable_key,
        'stripe_buy_button_id': buy_button_id,
    }

# Inject stripe keys into the app
@app.context_processor
def inject_vars():
    vars = inject_stripe_keys()
    return vars

# Custom error handler for your Flask app
@app.errorhandler(Exception)
def handle_unhandled_exception(e):
    username = session.get('username', 'Unknown User')
    # Log the unhandled exception with its details
    logging.exception('Unhandled exception: %s', e)

    # If the exception has an attribute 'code', use it; otherwise, set the default error code to 500 (Internal Server Error)
    error_code = getattr(e, 'code', 500)

    # Get the error message from the exception, if available, or use a default message
    message = getattr(e, 'description', 'An unexpected error occurred. Please try again later.')

    # Render the 'error.html' template with the specified error code, message, and return it along with the error code as HTTP status code
    return render_template('error.html', username=username, error_code=error_code, message=message), error_code

def make_request_with_backoff(func, *args, **kwargs):
    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"Attempt #{retry_count + 1}")
            return func(*args, **kwargs)
        except spotipy.exceptions.SpotifyException as e:
            print(f"SpotifyException caught: {e}")
            if e.http_status == 429:  # Rate limit error
                sleep_time = int(e.headers.get('Retry-After', 0)) + 1
                print(f"Rate limit hit, waiting for {sleep_time} seconds")
                time.sleep(sleep_time)
                retry_count += 1
            else:
                raise e
    raise Exception("Max retries reached")


@app.route('/')
def index():
    if not session.get('uuid'):
        # Step 1. Visitor is unknown, give random ID
        session['uuid'] = str(uuid.uuid4())
    return render_template('index.html')


@app.route('/login')
def login():
    try:
        auth_manager = SpotifyOAuth(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI, scope=SCOPE, cache_path=session_cache_path(), show_dialog=True, requests_timeout=30)
        if request.args.get("code"):
            try:
                # Step 3. Being redirected from Spotify auth page
                auth_manager.get_access_token(request.args.get("code"))
                sp = spotipy.Spotify(auth_manager=auth_manager)
                
                # Use the custom function to make requests
                user_info = make_request_with_backoff(sp.me)
                
                print(f"{user_info['display_name']} ({user_info['id']}) logged in")
                return redirect('/playlists')
            except Exception as e:
                print("Error in Step 3:", e)
                traceback.print_exc(file=sys.stdout)
                return "Error in Step 3: " + str(e), 500

        if not auth_manager.get_cached_token():
            try:
                # Step 2. Display sign in link when no token
                auth_url = auth_manager.get_authorize_url()
                return redirect(auth_url)
            except Exception as e:
                print("Error in Step 2:", e)
                traceback.print_exc(file=sys.stdout)
                return "Error in Step 2: " + str(e), 500

        # Step 4. Signed in, redirect to playlists
        return redirect('/playlists')
    except Exception as e:
        print("Error in login route:", e)
        traceback.print_exc(file=sys.stdout)
        return "Error in login route: " + str(e), 500


@app.route('/logout')
def logout():
    try:
        # Remove the CACHE file (.cache-test) so that a new user can authorize.
        os.remove(session_cache_path())
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    session.clear()
    return redirect('/')

def remove_duplicates(tracks):
    seen_track_ids = set()
    unique_tracks = []

    for track in tracks:
        if track['track'] is None:
            continue

        track_id = track['track']['id']

        if track_id not in seen_track_ids:
            seen_track_ids.add(track_id)
            unique_tracks.append(track)

    return unique_tracks


def delete_playlist(sp, playlist_id):
    user_id = make_request_with_backoff(sp.current_user)["id"]
    sp.user_playlist_unfollow(user=user_id, playlist_id=playlist_id)

def get_playlist_tracks(sp, playlist_id):
    playlist = make_request_with_backoff(sp.playlist, playlist_id)
    return playlist['tracks']['items']


def sp_track(sp, track_id):
    return make_request_with_backoff(sp.track, track_id)

def paginate_playlists(playlists: List, limit: int, offset: int):
    start = offset
    end = offset + limit
    return playlists[start:end]

def check_playlist_before_submit(sp, playlist_id, initial_tracks):
    # Fetch the current playlist
    current_playlist = make_request_with_backoff(sp.playlist, playlist_id)

    # Check if the playlist still exists
    if not current_playlist:
        return {"status": "error", "message": "The playlist no longer exists."}

    # Fetch the current playlist tracks
    current_tracks = current_playlist['tracks']['items']

    # Get the track IDs for initial and current tracks
    initial_track_ids = {track['track']['id'] for track in initial_tracks}
    current_track_ids = {track['track']['id'] for track in current_tracks}

    # Compare the current track IDs with the initially loaded track IDs
    if initial_track_ids != current_track_ids:
        return {
            "status": "error",
            "message": "The playlist has been modified. Please refresh the page to load the updated playlist.",
        }

    return {"status": "success", "message": "Playlist is ready for submission."}

@app.route('/playlists/')
def playlists():
    auth_manager = spotipy.oauth2.SpotifyOAuth(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI,
                                               cache_path=session_cache_path())
    if not auth_manager.get_cached_token():
        return redirect('/')
    sp = spotipy.Spotify(auth_manager=auth_manager)
    limit = 12
    offset = int(request.args.get('offset', 0))
    previous_offset = max(offset - limit, 0)
    api_limit = 50
    api_offset = (offset // api_limit) * api_limit
    user_playlists = make_request_with_backoff(sp.current_user_playlists, limit=api_limit, offset=api_offset)
    user_profile = make_request_with_backoff(sp.me)
    username = user_profile['display_name'] 
    total_playlists = user_playlists['total']
    all_playlists = user_playlists['items']
    playlist_data = []
    unique_track_counts = {}
    for playlist in all_playlists:
        tracks = make_request_with_backoff(get_playlist_tracks, sp, playlist['id'])
        unique_tracks = remove_duplicates(tracks) 
        if len(unique_tracks) >= 1:
            playlist_data.append({
                'id': playlist['id'],
                'name': playlist['name'],  # Add the playlist name
                'images': playlist['images'],  # Add the playlist images
                'unique_track_count': len(unique_tracks),
                'image_url': playlist['images'][0]['url'] if playlist['images'] else None,
                'external_urls': playlist['external_urls']
            })
            unique_track_counts[playlist['id']] = len(unique_tracks)

    total_filtered_playlists = len(playlist_data)  # Update the total playlists count
    paginated_playlists = paginate_playlists(playlist_data, limit, offset)
    playlist_id = request.args.get('playlist_id', None)
    return render_template(
        'playlist_list.html',
        playlists=paginated_playlists,
        playlist_id=playlist_id,
        unique_track_counts=unique_track_counts,
        offset=offset,
        previous_offset=previous_offset,
        total_playlists=total_filtered_playlists,  # Pass the filtered playlists count
        limit=limit,
        request=request,
        username=username
    )
   
@app.route('/rate_playlist/<playlist_id>/', methods=['GET', 'POST'])
def rate_playlist(playlist_id):
    auth_manager = spotipy.oauth2.SpotifyOAuth(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI,
                                               cache_path=session_cache_path())
    if not auth_manager.get_cached_token():
        return redirect('/')
    sp = spotipy.Spotify(auth_manager=auth_manager)
    user_profile = make_request_with_backoff(sp.me)
    username = user_profile['display_name'] 
    playlist = make_request_with_backoff(sp.playlist, playlist_id)
    check_result = check_playlist_before_submit(sp, playlist_id, playlist['tracks']['items'])
    if check_result['status'] == "error":
        message = check_result['message']
        return render_template('error.html', username=username, message=message)
    if request.MOBILE:
        return redirect(url_for('mobile_rate_playlist', username=username, playlist_id=playlist_id))
    try:
        tracks = get_playlist_tracks(sp, playlist_id)
        unique_tracks = remove_duplicates(tracks)
        if 'ratings' in session and playlist_id in session['ratings']:
            return redirect(url_for('recommendation', username=username, playlist_id=playlist_id))
        for track in unique_tracks:
            track_info = sp_track(sp, track['track']['id'])
            track['spotify_uri'] = track_info['uri']
        # Render the template without the access token
        return render_template('rate_playlist.html', tracks=unique_tracks, playlist_id=playlist_id, username=username)
    except Exception as e:
        return render_template('error.html', username=username, message=f'Failed to retrieve playlist information. Please try again later. Exception: {str(e)}')


@app.route('/mobile_rate_playlist/<playlist_id>/', methods=['GET', 'POST'])
def mobile_rate_playlist(playlist_id):
    auth_manager = spotipy.oauth2.SpotifyOAuth(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI,
                                               cache_path=session_cache_path())
    if not auth_manager.get_cached_token():
        return redirect('/')
    sp = spotipy.Spotify(auth_manager=auth_manager)
    user_profile = make_request_with_backoff(sp.me)
    username = user_profile['display_name'] 
    playlist = make_request_with_backoff(sp.playlist, playlist_id)
    check_result = check_playlist_before_submit(sp, playlist_id, playlist['tracks']['items'])
    if check_result['status'] == "error":
        message = check_result['message']
        return render_template('error.html', username=username, message=message)
    if not request.MOBILE:
        return redirect(url_for('rate_playlist', username=username, playlist_id=playlist_id))
    try:
        tracks = get_playlist_tracks(sp, playlist_id)
        unique_tracks = remove_duplicates(tracks)
        if 'ratings' in session and playlist_id in session['ratings']:
            return redirect(url_for('recommendation', username=username, playlist_id=playlist_id))
        for track in unique_tracks:
            track_info = sp_track(sp, track['track']['id'])
            track['spotify_uri'] = track_info['uri']
        # Render the template without the access token
        return render_template('mobile_rate_playlist.html', tracks=unique_tracks, playlist_id=playlist_id, username=username)
    except Exception as e:
        return render_template('error.html', username=username, message=f'Failed to retrieve playlist information. Please try again later. Exception: {str(e)}')



@app.route('/save_ratings/<playlist_id>/', methods=['POST'])
def save_ratings(playlist_id):
    auth_manager = spotipy.oauth2.SpotifyOAuth(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI,
                                               cache_path=session_cache_path())
    if not auth_manager.get_cached_token():
        return redirect('/')
    sp = spotipy.Spotify(auth_manager=auth_manager)
    user_profile = make_request_with_backoff(sp.me)
    username = user_profile['display_name'] 
    playlist = make_request_with_backoff(sp.playlist, playlist_id)
    # Check if the playlist exists and hasn't been modified
    check_result = check_playlist_before_submit(sp, playlist_id, playlist['tracks']['items'])
    if check_result['status'] == "error":
        message = check_result['message']
        return render_template('error.html', username=username, message=message)
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
    return redirect(url_for('create_playlist', playlist_id=playlist_id, username=username))
    
    
@app.route('/create_playlist/<playlist_id>/', methods=['GET', 'POST'])
def create_playlist(playlist_id):
    print("Entering create_playlist route")
    auth_manager = spotipy.oauth2.SpotifyOAuth(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI,
                                               cache_path=session_cache_path())
    if not auth_manager.get_cached_token():
        print("No cached token found, redirecting to /")
        return redirect('/')
    sp = spotipy.Spotify(auth_manager=auth_manager)
    user_profile = make_request_with_backoff(sp.me)
    username = user_profile['display_name'] 
    playlist = make_request_with_backoff(sp.playlist, playlist_id)
    check_result = check_playlist_before_submit(sp, playlist_id, playlist['tracks']['items'])
    if check_result['status'] == "error":
        message = check_result['message']
        return render_template('error.html', username=username, message="Deleted or modified playlist mid process. Please try again.")
    if request.method == 'POST':
        print("POST request detected")
        playlist_name = request.form['playlist_name']
        if not playlist_name:
            return render_template('error.html', username=username, message="Playlist name cannot be empty.")
        elif len(playlist_name) > 100:
            return render_template('error.html', username=username, message="Playlist name is too long.")
        # Save the playlist name and ID in the session
        session['rec_playlist_name'] = playlist_name
        user_id = make_request_with_backoff(sp.me)['id']
        rec_playlist = sp.user_playlist_create(user=user_id, name=playlist_name)
        rec_playlist_id = rec_playlist['id']
        session['rec_playlist_id'] = rec_playlist_id
        print(f"Redirecting to recommendation route with playlist_id: {playlist_id}, rec_playlist_id: {rec_playlist_id}")
        return redirect(url_for('recommendation', playlist_id=playlist_id, username=username, rec_playlist_id=rec_playlist_id))
    print("GET request detected, rendering create_playlist.html")
    return render_template('create_playlist.html', playlist_id=playlist_id, username=username)


@app.route('/recommendation/<playlist_id>/<rec_playlist_id>/')
def recommendation(playlist_id, rec_playlist_id):
    auth_manager = spotipy.oauth2.SpotifyOAuth(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI,
                                               cache_path=session_cache_path())
    if not auth_manager.get_cached_token():
        return redirect('/')
    sp = spotipy.Spotify(auth_manager=auth_manager)
    user_profile = make_request_with_backoff(sp.me)
    username = user_profile['display_name'] 
    user_id = make_request_with_backoff(sp.me)['id']
    session['spotify_username'] = user_id
    session['username'] = username
    ratings = session['ratings']
    request_id = str(uuid.uuid4())
    threading.Thread(target=background_recommendation, args=(playlist_id, rec_playlist_id, request_id, auth_manager, ratings, user_id)).start()
    return render_template("recommendation_progress.html", request_id=request_id, username=username)

def get_user_liked_tracks(sp):
    liked_track_ids = set()
    offset = 0
    limit = 50

    while True:
        results = make_request_with_backoff(lambda: sp.current_user_saved_tracks(limit=limit, offset=offset))
        if not results['items']:
            break

        for item in results['items']:
            liked_track_ids.add(item['track']['id'])
        
        offset += limit

    return liked_track_ids

def background_recommendation(playlist_id, rec_playlist_id, request_id, auth_manager, ratings, spotify_username):
    def emit_error_and_delete_playlist(request_id, message):
        socketio.emit("recommendation_error", {"request_id": request_id, "message": message}, namespace='/recommendation')
    
    sp = spotipy.Spotify(auth_manager=auth_manager)
    liked_track_ids = get_user_liked_tracks(sp)
    playlist = make_request_with_backoff(sp.playlist, playlist_id)
    tracks = playlist['tracks']['items']
    if not ratings:
        return redirect(url_for('rate_playlist', playlist_id=playlist_id))
    track_ids = list(ratings.keys())

    # Retrieve audio features for only the tracks in the seed playlist that were rated by the user
    audio_features = make_request_with_backoff(sp.audio_features, track_ids)
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
    min_neighbors = max(5, len(X) // 10)

    param_grid = {
        'n_neighbors': range(min_neighbors, max_neighbors + 1),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    socketio.emit("knn_model_trained", {"request_id": request_id}, namespace='/recommendation')
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
            rec_tracks = make_request_with_backoff(sp.recommendations, seed_tracks=[track_id], limit=int(len(playlist_data)/2))['tracks']
            for track in rec_tracks:
                # Exclude tracks that are already in the seed playlist or liked by the user
                if track['id'] not in track_ids and track['id'] not in liked_track_ids:
                    rec_track_ids.add(track['id'])
        except Exception as e:
            emit_error_and_delete_playlist(request_id, "Error: Adding tracks to playlist")

    if not rec_track_ids:
        emit_error_and_delete_playlist(request_id, "Error: No tracks found to be added")
    socketio.emit("recommended_tracks_retrieved", {"request_id": request_id}, namespace='/recommendation')
    track_chunks = [list(rec_track_ids)[i:i+100] for i in range(0, len(rec_track_ids), 100)]

    for track_chunk in track_chunks:
        try:
            make_request_with_backoff(sp.user_playlist_add_tracks, user=spotify_username, playlist_id=rec_playlist_id, tracks=track_chunk)
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