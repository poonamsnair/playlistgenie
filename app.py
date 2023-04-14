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
                sleep_time = (2 ** retry_count) + random.uniform(0, 1)
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

def get_track_genres(sp, track_id):
    track = make_request_with_backoff(lambda: sp.track(track_id))
    artist_id = track['artists'][0]['id']
    artist = make_request_with_backoff(lambda: sp.artist(artist_id))
    return artist['genres']

def get_unique_genres(playlist_data):
    unique_genres = set()
    for track in playlist_data:
        unique_genres.update(track['genres'])
    return unique_genres

def one_hot_encode_genres(track_genres, unique_genres):
    return [int(genre in track_genres) for genre in unique_genres]

def get_top_tracks_artists_genres_for_playlist(sp, tracks, num_genres=5, num_artists=5):
    genre_count = {}
    artist_count = {}

    for item in tracks:
        track = item['track']
        track_genres = get_track_genres(sp, track['id'])
        for genre in track_genres:
            if genre not in genre_count:
                genre_count[genre] = 1
            else:
                genre_count[genre] += 1

        related_artists = make_request_with_backoff(lambda: sp.artist_related_artists(track['artists'][0]['id']))['artists']
        for artist in related_artists:
            if artist['id'] not in artist_count:
                artist_count[artist['id']] = 1
            else:
                artist_count[artist['id']] += 1

    top_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)[:num_genres]
    top_artists = sorted(artist_count.items(), key=lambda x: x[1], reverse=True)[:num_artists]

    top_genre_names = [genre[0] for genre in top_genres]
    top_artist_ids = [artist[0] for artist in top_artists]

    return top_genre_names, top_artist_ids

def get_audio_features(sp, track_ids):
    audio_features = []
    for i in range(0, len(track_ids), 50):
        audio_features.extend(make_request_with_backoff(sp.audio_features, track_ids[i:i+50]))
    return audio_features

def get_top_recommended_tracks(sp, rec_track_ids, playlist_data, best_model, scaler, pca, unique_genres, feature_keys):
    rec_tracks_data = []
    for track_id in rec_track_ids:
        try:
            track = make_request_with_backoff(sp.track, track_id)
            release_year = int(track['album']['release_date'][:4])

            if release_year < 2022:
                continue
            
            track_audio_features = make_request_with_backoff(sp.audio_features, track_id)
            if track_audio_features is None:
                continue
            track_audio_features = [track_audio_features[0][key] if track_audio_features[0][key] is not None else 0 for key in feature_keys]
            print(f"Track ID: {track_id}, Track: {track}, Audio features: {track_audio_features}")
            track_genres = get_track_genres(sp, track_id)
            track_genres_encoded = [1 if genre in track_genres else 0 for genre in unique_genres]
            track_features = track_audio_features + track_genres_encoded
            print(f"Track genres: {track_genres_encoded}, Final track features: {track_features}")
            scaled_track_features = scaler.transform([track_features])
            pca_track_features = pca.transform(scaled_track_features)
            predicted_rating = best_model.predict(pca_track_features)[0]
            print(f"Predicted rating for track {track_id}: {predicted_rating}")
            rec_tracks_data.append((track_id, predicted_rating))

        except Exception as e:
            print(f"Error while processing track {track_id}: {e}")

    # Sort tracks by predicted rating
    sorted_rec_tracks = sorted(rec_tracks_data, key=lambda x: x[1], reverse=True)
    print("Sorted recommended tracks:", sorted_rec_tracks)
    # Return the top 100 recommended tracks
    top_rec_track_ids = [data[0] for data in sorted_rec_tracks[:100]]
    print("Top 100 recommended track IDs:", top_rec_track_ids)
    rec_tracks = []
    for track_id in top_rec_track_ids:
        try:
            track = make_request_with_backoff(sp.track, track_id)
            rec_tracks.append(track)
        except Exception as e:
            print(f"Error retrieving data for track {track_id}: {str(e)}")
            continue

    return rec_tracks

def get_recently_played(sp):
    recently_played = make_request_with_backoff(sp.current_user_recently_played, limit=50)
    recently_played_ids = [item['track']['id'] for item in recently_played['items']]
    return recently_played_ids

def get_playlist_track_ids(sp, user_id, playlist_name):
    playlists = sp.user_playlists(user_id)
def get_playlist_track_ids(sp, user_id, playlist_name):
    playlists = make_request_with_backoff(sp.user_playlists, user_id)
    target_playlist_id = None
    for playlist in playlists['items']:
        if playlist['name'] == playlist_name:
            target_playlist_id = playlist['id']
            break

    if target_playlist_id is None:
        return []

    results = make_request_with_backoff(sp.playlist_tracks, target_playlist_id)
    track_ids = [item['track']['id'] for item in results['items']]
    return track_ids

def get_seed_playlist_genres(playlist_data):
    seed_genres = set()
    for track in playlist_data:
        seed_genres.update(track['genres'])
    return seed_genres

def get_liked_songs_with_similar_genres(sp, seed_genres):
    liked_track_ids = get_user_liked_tracks(sp)
    liked_songs = []

    for track_id in liked_track_ids:
        track_genres = get_track_genres(sp, track_id)
        if seed_genres.intersection(track_genres):
            liked_songs.append({
                'id': track_id,
                'genres': track_genres
            })

    return liked_songs

def generate_ratings_for_liked_songs(sp, liked_songs, spotify_username):
    ratings = {}
    max_popularity = 100
    max_playlists = 2
    recently_played = get_recently_played(sp)
    on_repeat_ids = get_playlist_track_ids(sp, spotify_username, "On Repeat")
    repeat_rewind_ids = get_playlist_track_ids(sp, spotify_username, "Repeat Rewind")

    for song in liked_songs:
        track_id = song['id']
        track = make_request_with_backoff(sp.track, track_id)
        popularity = track['popularity']

        recency_score = 0
        if track_id in recently_played:
            recency_score += 1

        playlist_score = 0
        if track_id in on_repeat_ids:
            playlist_score += 1
        if track_id in repeat_rewind_ids:
            playlist_score += 1

        # Combine popularity, recency, and playlist scores to compute the final rating
        rating = (popularity / max_popularity) * 0.4 + (recency_score * 0.3) + (playlist_score / max_playlists) * 0.3
        rating = round(rating * 10)  # Convert the rating to a scale of 1-10

        ratings[track_id] = rating

    return ratings


def background_recommendation(playlist_id, rec_playlist_id, request_id, auth_manager, ratings, spotify_username):
    def emit_error_and_delete_playlist(request_id, message):
        socketio.emit("recommendation_error", {"request_id": request_id, "message": message}, namespace='/recommendation')
    
    sp = spotipy.Spotify(auth_manager=auth_manager)
    liked_track_ids = get_user_liked_tracks(sp)
    playlist = make_request_with_backoff(lambda: sp.playlist(playlist_id))
    tracks = playlist['tracks']['items']
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
    
    # Get unique genres in the seed playlist
    seed_genres = get_seed_playlist_genres(playlist_data)

    # Get liked songs with similar genres
    liked_songs = get_liked_songs_with_similar_genres(sp, seed_genres)

    # Generate ratings for liked songs
    liked_songs_ratings = generate_ratings_for_liked_songs(sp, liked_songs, spotify_username)

    # Update the playlist_data and ratings with liked songs and their ratings
    for song in liked_songs:
        track_id = song['id']
        if track_id not in ratings:
            audio_feature = make_request_with_backoff(sp.audio_features, [track_id])[0]
            if audio_feature:
                feature_dict = {key: audio_feature[key] for key in feature_keys}
                feature_dict['ratings'] = liked_songs_ratings[track_id]
                feature_dict['genres'] = song['genres']
                playlist_data.append(feature_dict)

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

    # Get playlist-specific top artists, and genres
    playlist_top_genres, playlist_top_artists = get_top_tracks_artists_genres_for_playlist(sp, tracks)

    # Generate recommendations based on user's top tracks, artists, and genres
    rec_track_ids = set()
    seed_artists = playlist_top_artists[:3]
    seed_genres = playlist_top_genres[:2]
    socketio.emit("top_artists", {"request_id": request_id}, namespace='/recommendation')

    for track_id in [d['id'] for d in playlist_data]:
        try:
            rec_tracks = make_request_with_backoff(sp.recommendations, seed_artists=seed_artists, seed_genres=seed_genres, limit=50, market='from_token')['tracks']
            for track in rec_tracks:
                # Exclude tracks that are already in the seed playlist or liked by the user
                if track['id'] not in track_ids and track['id'] not in liked_track_ids:
                    rec_track_ids.add(track['id'])
        except Exception as e:
            emit_error_and_delete_playlist(request_id, "Adding tracks to playlist")
    
    socketio.emit("top_artists_done", {"request_id": request_id}, namespace='/recommendation')
    # After selecting the best model
    best_model.fit(X_scaled_pca, y)

    # Get top recommended tracks
    rec_tracks = get_top_recommended_tracks(sp, rec_track_ids, playlist_data, best_model, scaler, pca, unique_genres, feature_keys)
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