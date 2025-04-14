from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from flask_cors import CORS
import random
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load dataset
df = pd.read_csv('songs_dataset_with_filled_thumbnails.csv')

def get_mood_criteria(mood):
    mood_map = {
        'happy': {'valence': (0.6, 1.0), 'energy': (0.6, 1.0)},
        'sad': {'valence': (0.0, 0.4), 'energy': (0.0, 0.5)},
        'energetic': {'valence': (0.5, 1.0), 'energy': (0.7, 1.0)},
        'calm': {'valence': (0.3, 0.6), 'energy': (0.0, 0.5)}
    }
    return mood_map.get(mood, {'valence': (0, 1), 'energy': (0, 1)})

def filter_songs(user_mood, favorite_artists, favorite_genre):
    # Add some small random variation to mood criteria bounds (±0.05)
    mood_criteria = get_mood_criteria(user_mood)
    valence_min, valence_max = mood_criteria['valence']
    energy_min, energy_max = mood_criteria['energy']
    
    # Add slight random variation to the boundaries (±0.05) to get different songs each time
    random_valence_min = max(0, valence_min - random.uniform(0, 0.05))
    random_valence_max = min(1, valence_max + random.uniform(0, 0.05))
    random_energy_min = max(0, energy_min - random.uniform(0, 0.05))
    random_energy_max = min(1, energy_max + random.uniform(0, 0.05))
    
    filtered_df = df[
        (df['valence'].between(random_valence_min, random_valence_max)) &
        (df['energy'].between(random_energy_min, random_energy_max))
    ]

    if favorite_artists:
        # If there are multiple artists, randomly select a subset of them
        if len(favorite_artists) > 1:
            # Always include at least 1 artist, up to all artists
            num_artists = random.randint(1, len(favorite_artists))
            selected_artists = random.sample(favorite_artists, num_artists)
        else:
            selected_artists = favorite_artists
            
        filtered_df = filtered_df[filtered_df['track_artist'].str.contains('|'.join(selected_artists), case=False, na=False)]

    if favorite_genre:
        filtered_df = filtered_df[filtered_df['playlist_genre'].str.contains(favorite_genre, case=False, na=False)]

    return filtered_df

def generate_playlist(user_mood, favorite_artists, favorite_genre, playlist_size=10):
    filtered_songs = filter_songs(user_mood, favorite_artists, favorite_genre)
    
    # If we have enough songs, randomly sample them
    if len(filtered_songs) > playlist_size:
        # Truly random selection - no ranking, just pure randomness
        selected_songs = filtered_songs.sample(playlist_size, random_state=random.randint(1, 10000))
    else:
        selected_songs = filtered_songs
        
        # If we don't have enough songs, try to find more with relaxed criteria
        if len(selected_songs) < playlist_size:
            # Try without genre restriction first
            additional_songs = filter_songs(user_mood, favorite_artists, None)
            additional_songs = additional_songs[~additional_songs['track_name'].isin(selected_songs['track_name'])]
            
            # If still not enough, try with fewer artists
            if len(selected_songs) + len(additional_songs) < playlist_size and favorite_artists and len(favorite_artists) > 1:
                # Try with half the artists
                subset_size = max(1, len(favorite_artists) // 2)
                subset_artists = random.sample(favorite_artists, subset_size)
                more_songs = filter_songs(user_mood, subset_artists, None)
                more_songs = more_songs[~more_songs['track_name'].isin(selected_songs['track_name']) & 
                                     ~more_songs['track_name'].isin(additional_songs['track_name'])]
                additional_songs = pd.concat([additional_songs, more_songs])
            
            # If still not enough, try with just mood
            if len(selected_songs) + len(additional_songs) < playlist_size:
                mood_only_songs = filter_songs(user_mood, [], None)
                mood_only_songs = mood_only_songs[~mood_only_songs['track_name'].isin(selected_songs['track_name']) & 
                                             ~mood_only_songs['track_name'].isin(additional_songs['track_name'])]
                additional_songs = pd.concat([additional_songs, mood_only_songs])
            
            # Randomly select from additional songs to complete the playlist
            needed_songs = playlist_size - len(selected_songs)
            if len(additional_songs) > needed_songs:
                additional_songs = additional_songs.sample(needed_songs, random_state=random.randint(1, 10000))
            
            selected_songs = pd.concat([selected_songs, additional_songs])
    
    # Ensure no duplicates
    selected_songs = selected_songs.drop_duplicates(subset='track_name')
    
    # Limit to required columns
    return selected_songs[['track_id','track_name', 'track_artist', 'playlist_genre', 'track_popularity', 'YouTube URL', 'Thumbnail_URL']]

def generate_multiple_playlists(user_mood, favorite_artists, favorite_genres):
    playlists = {}
    
    # Generate a new timestamp-based seed for each request
    timestamp_seed = int(datetime.now().timestamp() * 1000) % 10000
    random.seed(timestamp_seed)
    
    for genre in favorite_genres:
        # Each genre gets its own random seed derived from the timestamp
        genre_seed = (timestamp_seed + hash(genre)) % 10000
        random.seed(genre_seed)
        np.random.seed(genre_seed)
        
        playlists[genre] = generate_playlist(user_mood, favorite_artists, genre)
    
    return playlists

@app.route('/generate_playlists', methods=['POST'])
def generate_playlists():
    data = request.json
    user_mood = data.get('mood')
    favorite_artists = data.get('artists', [])
    favorite_genres = data.get('genres', [])
    print(user_mood, favorite_artists, favorite_genres)
    # Create a unique seed based on timestamp for this specific request
    random.seed(int(datetime.now().timestamp() * 1000))
    np.random.seed(random.randint(1, 10000))
    
    playlists = generate_multiple_playlists(user_mood, favorite_artists, favorite_genres)

    result = {}
    for genre, playlist in playlists.items():
        playlist_with_urls = []
        playlist_thumbnail = None
        
        # Shuffle the playlist order for extra randomness
        playlist = playlist.sample(frac=1, random_state=random.randint(1, 10000)).reset_index(drop=True)
        
        for _, song in playlist.iterrows():
            youtube_url, thumbnail_url = song['YouTube URL'], song['Thumbnail_URL']
            playlist_with_urls.append({
                'track_id': song['track_id'],
                'track_name': song['track_name'],
                'track_artist': song['track_artist'],
                'playlist_genre': song['playlist_genre'],
                'track_popularity': song['track_popularity'],
                'youtube_url': youtube_url,
                'thumbnail_url': thumbnail_url
            })
            if not playlist_thumbnail:
                playlist_thumbnail = thumbnail_url

        result[genre] = {
            'playlist_thumbnail': playlist_thumbnail,
            'songs': playlist_with_urls
        }
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)