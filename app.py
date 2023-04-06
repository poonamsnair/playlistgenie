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
import requests
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
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from urllib.parse import urlencode

import eventlet
import os
import spotipy
from flask import Flask, session
from flask_socketio import SocketIO
from flask_mobility import Mobility
from flask_bootstrap import Bootstrap
from flask_session import Session
from spotipy.oauth2 import SpotifyOAuth


eventlet.monkey_patch()
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", default=os.urandom(24))
Bootstrap(app)
socketio = SocketIO(app)
Mobility(app)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Set your Spotify client ID and redirect URI
client_id = os.environ.get("SPOTIFY_CLIENT_ID")
client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
redirect_uri = os.environ.get("SPOTIFY_REDIRECT_URI")
scope = 'user-library-read playlist-modify-public playlist-modify-private playlist-read-private streaming'

# initalise spauth
sp_oauth = SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope)


# index route
@app.route('/')
def index():
    # render index.html
    return render_template('index.html')

# login route
@app.route('/login')
def login():
    # get authorization URL
    auth_url = sp_oauth.get_authorize_url()

    # redirect to authorization URL
    return redirect(auth_url)


# callback route
@app.route('/callback')
def callback():
    # get authorization code from query parameters
    code = request.args.get('code')

    # exchange authorization code for access token
    token_info = sp_oauth.get_access_token(code)

    # store access token in session
    session['spotify_token'] = token_info['access_token']

    # redirect to playlist list page
    return redirect('/playlist_list')

# playlist route (playlist_list.html)
@app.route('/playlist_list')
def playlist_list():
    # check if user is logged in
    if 'spotify_token' not in session:
        return redirect('/')

    # get user's access token
    access_token = session['spotify_token']

    # create Spotify client
    sp = spotipy.Spotify(auth=access_token)

    # get user's playlists
    playlists = sp.current_user_playlists()

    # filter playlists with at least 1 track
    playlists_with_tracks = [playlist for playlist in playlists['items'] if playlist['tracks']['total'] > 0]

    # create list of playlist cards
    playlist_cards = []
    for playlist in playlists_with_tracks:
        # get unique track count
        unique_tracks = set()
        tracks = sp.playlist_tracks(playlist['id'])
        for track in tracks['items']:
            unique_tracks.add(track['track']['id'])
        unique_track_count = len(unique_tracks)

        # disable rate playlist button if unique track count is less than 5 or more than 100
        rate_disabled = unique_track_count < 5 or unique_track_count > 100

        # create playlist card
        playlist_card = {
            'title': playlist['name'],
            'unique_track_count': unique_track_count,
            'image': playlist['images'][0]['url'],
            'rate_disabled': rate_disabled
        }
        playlist_cards.append(playlist_card)

    # paginate playlist cards
    page = request.args.get('page', 1, type=int)
    per_page = 12
    start = (page - 1) * per_page
    end = start + per_page
    paginated_playlist_cards = playlist_cards[start:end]

    # render playlist_list.html with paginated playlist cards
    return render_template('playlist_list.html', playlist_cards=paginated_playlist_cards)


if __name__ == '__main__':
    env = os.environ.get('APP_ENV', 'test')

    if env == 'production':
        socketio.run(app, port=int(os.environ.get('PORT', 5000)))
    else:
        port = int(os.environ.get('PORT', 8888))
        socketio.run(app, port=port)
