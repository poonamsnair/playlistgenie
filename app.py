import eventlet
from flask import Flask, redirect, request, session, url_for, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_bootstrap import Bootstrap
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
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
import requests
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
SCOPE = 'user-library-read playlist-modify-public playlist-modify-private playlist-read-private streaming user-top-read'
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


def make_request_with_backoff(func, *args, max_retries=10, max_requests_per_second=20, timeout=10, **kwargs):
    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"Attempt #{retry_count + 1}")
            return func(*args, timeout=timeout, **kwargs)
        except (spotipy.exceptions.SpotifyException, requests.exceptions.RequestException) as e:
            print(f"Exception caught: {e}")
            if isinstance(e, spotipy.exceptions.SpotifyException) and e.http_status == 429:  # Rate limit error
                sleep_time = (2 ** retry_count) + random.uniform(0, 1)
                print(f"Rate limit hit, waiting for {sleep_time} seconds")
                time.sleep(sleep_time)
            elif isinstance(e, requests.exceptions.RequestException):  # Network or temporary issue
                sleep_time = (2 ** retry_count) + random.uniform(0, 1)
                print(f"Network issue or temporary API issue, waiting for {sleep_time} seconds")
                time.sleep(sleep_time)
            else:
                raise e

            retry_count += 1

        # Sleep to limit the rate of requests
        if retry_count % max_requests_per_second == 0:
            time.sleep(1)

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
            track_info = make_request_with_backoff(sp.track, track['track']['id'])
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
            track_info = make_request_with_backoff(sp.track, track['track']['id'])
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
        rec_playlist = make_request_with_backoff(sp.user_playlist_create, user=user_id, name=playlist_name)
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


def get_user_liked_tracks(sp, limit=50):
    liked_track_ids = []
    offset = 0
    while True:
        try:
            results = make_request_with_backoff(sp.current_user_saved_tracks, limit=limit, offset=offset)
            if not results['items']:
                break

            liked_track_ids.extend([item['track']['id'] for item in results['items']])
            offset += limit
        except Exception as e:
            logging.error(f"Error in get_user_liked_tracks: {str(e)}")
            break

    return liked_track_ids

def get_track_genres(sp, track_id):
    track = make_request_with_backoff(sp.track, track_id)
    artist_id = track['artists'][0]['id']
    artist = make_request_with_backoff(sp.artist, artist_id)
    return artist['genres']

def get_unique_genres(playlist_data):
    unique_genres = set()
    for track in playlist_data:
        unique_genres.update(track['genres'])
    return unique_genres

def one_hot_encode_genres(track_genres, unique_genres):
    return [int(genre in track_genres) for genre in unique_genres]

def get_audio_features(sp, track_ids):
    audio_features = []
    for i in range(0, len(track_ids), 50):
        audio_features.extend(make_request_with_backoff(sp.audio_features, track_ids[i:i+50]))
    return audio_features

def extract_best_model_params(top_rated_tracks):
    best_params = {}
    best_params["target_tempo"] = round(sum(track.get("tempo", 120) for track in top_rated_tracks) / len(top_rated_tracks), 1)
    best_params["target_popularity"] = round(sum(track.get("popularity", 50) for track in top_rated_tracks) // len(top_rated_tracks), 0)
    best_params["target_energy"] = round(sum(track.get("energy", 0.5) for track in top_rated_tracks) / len(top_rated_tracks), 1)

    return best_params

def get_random_tracks(sp, seed_tracks, best_model_params, num_tracks=250, popularity_range=(30, 70), max_retries=10, max_requests_per_second=20, exclude_tracks=None):
    if exclude_tracks is None:
        exclude_tracks = []
        
    if len(seed_tracks) > 5:
        raise ValueError("The number of seed tracks should not exceed 5.")
        
    random_tracks = []

    while len(random_tracks) < num_tracks:
        print(f"Getting recommendations, current random_tracks length: {len(random_tracks)}")
        # Get recommendations based on seed tracks and best model parameters with back-off logic
        recommended_tracks_response = make_request_with_backoff(
            sp.recommendations,
            seed_tracks=seed_tracks,
            limit=100,
            max_retries=max_retries,
            max_requests_per_second=max_requests_per_second,
            **best_model_params
        )

        # Extract the 'tracks' list from the response
        recommended_tracks = recommended_tracks_response.get('tracks', [])

        # Filter tracks by popularity, release year, and exclude tracks
        filtered_tracks = [
            track for track in recommended_tracks
            if popularity_range[0] <= track['popularity'] <= popularity_range[1]
            and (track.get('album') is None or track['album']['release_date'][:4] == '2023')
            and track['id'] not in exclude_tracks
        ]
        random_tracks.extend(filtered_tracks)

    return random_tracks[:num_tracks]


def get_top_recommended_tracks(best_model, scaler, pca, feature_keys, unique_genres, sp, top_rated_tracks, best_model_params, num_tracks=100, exclude_tracks=None, num_recommendations=20, batch_size=50, rating_threshold=8, max_retries=5, max_requests_per_second=20, ):
    found_tracks = 0
    top_recommendations = []

    # Generate a large pool of random tracks released in 2023
    seed_tracks = [track['id'] for track in top_rated_tracks]
    random_tracks = get_random_tracks(sp, seed_tracks, best_model_params)

    # Split random tracks into batches
    random_track_batches = [random_tracks[i:i + batch_size] for i in range(0, len(random_tracks), batch_size)]

    # Retrieve audio features for random tracks
    audio_features = []
    for batch in random_track_batches:
        random_track_ids = [track['id'] for track in batch]
        audio_features_batch = make_request_with_backoff(sp.audio_features, random_track_ids, max_retries=max_retries, max_requests_per_second=max_requests_per_second)
        audio_features += audio_features_batch

        # Get genres for random tracks
        for track, audio_feature in zip(batch, audio_features_batch):
            track['genres'] = get_track_genres(sp, track['id'])

            # Stay within rate limit
            time.sleep(1/max_requests_per_second)

    # Pre-process random tracks for model prediction
    X_random = []
    for track, audio_feature in zip(random_tracks, audio_features):
        if audio_feature is not None:
            feature_dict = {key: audio_feature[key] for key in feature_keys}
            feature_dict['genres'] = track['genres']
            X_random.append([feature_dict[key] for key in feature_keys] + one_hot_encode_genres(feature_dict['genres'], unique_genres))

    # Scale and apply PCA to the features of random tracks
    X_random_scaled = scaler.transform(X_random)
    X_random_pca = pca.transform(X_random_scaled)

    # Make predictions using the trained model
    y_pred = best_model.predict(X_random_pca)

    # Add random tracks with predicted ratings above the threshold to the top recommendations
    for track, rating in zip(random_tracks, y_pred):
        if rating >= rating_threshold:
            top_recommendations.append(track)
            found_tracks += 1

        if found_tracks >= num_recommendations:
            break

    return top_recommendations[:num_recommendations]



def background_recommendation(playlist_id, rec_playlist_id, request_id, auth_manager, ratings, spotify_username):
    def emit_error_and_delete_playlist(request_id, message):
        socketio.emit("recommendation_error", {"request_id": request_id, "message": message}, namespace='/recommendation')
    sp = spotipy.Spotify(auth_manager=auth_manager)
    if not ratings:
        return redirect(url_for('rate_playlist', playlist_id=playlist_id))
    track_ids = list(ratings.keys())
    socketio.emit("playlist_data_analysis", {"request_id": request_id}, namespace='/recommendation')

    # Define feature_keys
    feature_keys = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]

    # Retrieve audio features for only the tracks in the seed playlist that were rated by the user
    audio_features = make_request_with_backoff(sp.audio_features, track_ids)
    socketio.emit("playlist_data_processing", {"request_id": request_id}, namespace='/recommendation')

    # Remove NoneType audio features
    audio_features = [feature for feature in audio_features if feature is not None]
    
    # Handling missing audio features
    for feature in audio_features:
        for key in feature_keys:
            if feature[key] is None:
                feature[key] = 0

    if len(audio_features) < 5:
        emit_error_and_delete_playlist(request_id, "Less than 5 tracks")
    elif len(audio_features) > 100:
        emit_error_and_delete_playlist(request_id, "More than 100 tracks")

  # Convert audio_features to a list of dictionaries
    playlist_data = []
    for feature in audio_features:
        feature_dict = {key: feature[key] for key in feature if key not in ['type', 'uri', 'track_href', 'analysis_url']}
        feature_dict['ratings'] = ratings[feature['id']]
        feature_dict['genres'] = get_track_genres(sp, feature['id'])
        playlist_data.append(feature_dict)

    socketio.emit("audio_features_retrieved", {"request_id": request_id}, namespace='/recommendation')

    unique_genres = get_unique_genres(playlist_data)
    X = [[d[key] for key in feature_keys] + one_hot_encode_genres(d['genres'], unique_genres) for d in playlist_data]
    y = [d['ratings'] for d in playlist_data]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Add PCA for dimensionality reduction
    pca = PCA(n_components=0.95)
    X_scaled_pca = pca.fit_transform(X_scaled)
    socketio.emit("transform_data", {"request_id": request_id}, namespace='/recommendation')

    if len(X_scaled_pca) < 50:
        max_neighbors = min(10, len(X_scaled_pca) - 1)
        min_neighbors = max(3, len(X_scaled_pca) // 10)
    else:
        max_neighbors = min(30, len(X_scaled_pca) - 1)
        min_neighbors = max(5, len(X_scaled_pca) // 10)

    # Model comparison: Decision Trees, Random Forests, and k-Nearest Neighbors (kNN)
    socketio.emit("model_comparison", {"request_id": request_id}, namespace='/recommendation')
    models = [
        {
            'name': 'KNN',
            'estimator': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': range(min_neighbors, max_neighbors + 1),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        },
        {
            'name': 'Decision Tree',
            'estimator': DecisionTreeClassifier(random_state=42),
            'param_grid': {
                'criterion': ['gini', 'entropy'],
                'max_depth': range(1, 11),
                'min_samples_split': range(2, 11),
                'min_samples_leaf': range(1, 11)
            }
        },
        {
            'name': 'Random Forest',
            'estimator': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [10, 20, 50],
                'criterion': ['gini', 'entropy'],
                'max_depth': [4, 5, 6],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [5, 10, 15]
            }
        }
    ]
    
    socketio.emit("best_model", {"request_id": request_id}, namespace='/recommendation')
    best_score = -1
    best_model = None
    best_params = None
    n_splits = 5
    min_samples_per_class = min(np.bincount(y))
    if min_samples_per_class >= n_splits:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = LeaveOneOut()

    print("Starting model optimization")
    socketio.emit("optimization_started", {"request_id": request_id}, namespace='/recommendation')
    for model in models:
        print(f"Optimizing {model['name']}...")
        socketio.emit("optimizing_model", {"request_id": request_id, "model_name": model['name']}, namespace='/recommendation')
        
        grid_search = GridSearchCV(estimator=model['estimator'], param_grid=model['param_grid'], cv=cv, n_jobs=2,
                                pre_dispatch='2*n_jobs', scoring='accuracy')
        grid_search.fit(X_scaled_pca, y)
        
        print(f"Done optimizing {model['name']}, best score: {grid_search.best_score_}")
        socketio.emit("optimization_done", {"request_id": request_id, "model_name": model['name'], "best_score": grid_search.best_score_}, namespace='/recommendation')

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        print(f"Done optimizing {model['name']}, best score: {best_score}")
    print(f"Optimization complete, best model: {best_model}, best params: {best_params}")
    socketio.emit("optimising_model", {"request_id": request_id}, namespace='/recommendation')

    # After selecting the best model
    best_model.fit(X_scaled_pca, y)
    
    # Get top rated tracks from the user-selected playlist
    top_rated_tracks = sorted(playlist_data, key=lambda x: x['ratings'], reverse=True)[:5]

    # Extract target parameters from the top rated tracks
    best_model_params = extract_best_model_params(top_rated_tracks)

    # Get user liked tracks
    user_liked_track_ids = get_user_liked_tracks(sp)

    # Get top recommended tracks using the top rated tracks and best model parameters
    rec_tracks = get_top_recommended_tracks(best_model, scaler, pca, feature_keys, unique_genres, sp, top_rated_tracks, best_model_params, exclude_tracks=user_liked_track_ids)
    
    print("Recommended tracks:", rec_tracks)
    print("Number of recommended tracks:", len(rec_tracks))
    
    # Add the recommended tracks to the playlist
    track_ids = [track['id'] for track in rec_tracks]
    print("Extracted track IDs:", track_ids)
    offset = 0
    while offset < len(track_ids):
        try:
            make_request_with_backoff(sp.user_playlist_add_tracks, user=spotify_username, playlist_id=rec_playlist_id, tracks=track_ids[offset:offset+100])
            print(f"Added tracks {offset}-{offset+99} to the playlist")
            offset += 100
        except Exception as e:
            logging.error(f"Error adding tracks to playlist: {str(e)}")
            emit_error_and_delete_playlist(request_id, f"Adding tracks to playlist: {str(e)}")
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