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

#initalise spauth
sp_oauth = SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope)


#login route
@app.route('/')
def index():
    # check if user is already logged in
    if 'spotify_token' in session:
        return redirect('/playlist_list')

    # get authorization url
    auth_url = sp_oauth.get_authorize_url()

    # render index.html with authorization url
    return render_template('index.html', auth_url=auth_url)

#callback route
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

#playlist route (playlist_list.html)
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
