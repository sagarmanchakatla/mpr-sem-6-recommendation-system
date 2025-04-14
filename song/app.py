from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = Flask(__name__)

# Load datasets
history_df = pd.read_csv('history.csv')
liked_df = pd.read_csv('liked.csv')
users_df = pd.read_csv('user.csv')
songs_df = pd.read_csv('songs_dataset_with_filled_thumbnails.csv')

# Mood to features mapping
mood_features = {
    'happy': {'valence': 0.8, 'energy': 0.7, 'danceability': 0.7},
    'sad': {'valence': 0.2, 'energy': 0.3, 'acousticness': 0.7},
    'energetic': {'energy': 0.9, 'danceability': 0.8, 'loudness': 0.7},
    'calm': {'energy': 0.3, 'acousticness': 0.8, 'valence': 0.5},
    'angry': {'energy': 0.8, 'loudness': 0.9, 'valence': 0.3}
}

# Combine liked songs and history for the specific user
def get_user_songs(user_id):
    # Get liked songs
    user_liked_songs = liked_df[liked_df['user_id'] == user_id][['song_title', 'song_artist']].rename(
        columns={'song_title': 'song_name', 'song_artist': 'artist_name'})
    # Get history songs
    user_history_songs = history_df[history_df['user_id'] == user_id][['song_name', 'artist_name']]
    
    # Combine liked and history songs
    user_songs = pd.concat([user_liked_songs, user_history_songs]).drop_duplicates()
    
    return user_songs

# Get user profile data
def get_user_profile(user_id):
    user_data = users_df[users_df['id'] == user_id].iloc[0]
    return {
        'genre': user_data['genre'],
        'fav_artist': user_data['fav_artist'],
        'curr_mood': user_data['curr_mood']
    }

# Adjust features based on user mood
def adjust_for_mood(features, mood):
    if mood.lower() in mood_features:
        mood_params = mood_features[mood.lower()]
        for feature, value in mood_params.items():
            if feature in features.columns:
                features[feature] = features[feature] * 0.7 + value * 0.3  # Blend with mood
    return features

# Function to recommend songs based on popularity
def recommend_popular_songs(top_n=10):
    return songs_df.sort_values(by='track_popularity', ascending=False).head(top_n)

# Function to recommend songs based on user profile, liked songs, and history
def recommend_personalized_songs(user_id, top_n=10):
    # Get the user's profile data
    user_profile = get_user_profile(user_id)
    
    # Get the user's liked and history songs
    user_songs = get_user_songs(user_id)
    
    # Filter songs by user's preferred genre if specified
    if user_profile['genre'] and user_profile['genre'].lower() != 'none':
        genre_filtered_songs = songs_df[
            songs_df['playlist_genre'].str.lower() == user_profile['genre'].lower()
        ]
    else:
        genre_filtered_songs = songs_df
    
    # Filter by favorite artist if specified
    if user_profile['fav_artist'] and user_profile['fav_artist'].lower() != 'none':
        artist_filtered_songs = genre_filtered_songs[
            genre_filtered_songs['track_artist'].str.lower().str.contains(
                user_profile['fav_artist'].lower()
            )
        ]
    else:
        artist_filtered_songs = genre_filtered_songs
    
    # If no user songs, return songs filtered by genre/artist/mood
    if user_songs.empty:
        feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                          'speechiness', 'acousticness', 'instrumentalness', 
                          'liveness', 'valence', 'tempo']
        
        # Adjust for mood
        mood_adjusted_songs = artist_filtered_songs.copy()
        mood_adjusted_songs[feature_columns] = adjust_for_mood(
            mood_adjusted_songs[feature_columns], 
            user_profile['curr_mood']
        )
        
        # Return popular songs from filtered set
        return mood_adjusted_songs.sort_values(
            by=['track_popularity'], 
            ascending=False
        ).head(top_n)
    
    # Filter the songs dataset to include only the user's liked and history songs
    user_song_mask = (
        songs_df['track_name'].isin(user_songs['song_name']) | 
        songs_df['track_artist'].isin(user_songs['artist_name'])
    )
    user_song_details = songs_df[user_song_mask]
    
    if user_song_details.empty:
        print("No matching songs found in the songs dataset.")
        return recommend_popular_songs(top_n)  # Fallback to popular songs
    
    # Extract features for content-based filtering
    feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                      'speechiness', 'acousticness', 'instrumentalness', 
                      'liveness', 'valence', 'tempo']
    user_song_features = user_song_details[feature_columns]
    
    # Normalize the features
    scaler = MinMaxScaler()
    try:
        user_song_features_normalized = scaler.fit_transform(user_song_features)
    except ValueError as e:
        print(f"Error during normalization: {e}")
        return recommend_popular_songs(top_n)  # Fallback to popular songs
    
    # Compute the average feature vector for the user's songs
    user_profile_features = user_song_features_normalized.mean(axis=0).reshape(1, -1)
    
    # Filter songs by genre and artist
    candidate_songs = artist_filtered_songs.copy()
    
    # Adjust for mood
    candidate_songs[feature_columns] = adjust_for_mood(
        candidate_songs[feature_columns], 
        user_profile['curr_mood']
    )
    
    # Compute similarity between the user profile and candidate songs
    candidate_features = candidate_songs[feature_columns]
    candidate_features_normalized = scaler.transform(candidate_features)
    similarity_scores = cosine_similarity(
        user_profile_features, 
        candidate_features_normalized
    ).flatten()
    
    # Add similarity scores to the songs dataset
    candidate_songs['similarity'] = similarity_scores
    
    # Exclude songs already liked or listened to by the user
    recommended_songs = candidate_songs[
        ~candidate_songs['track_name'].isin(user_songs['song_name'])
    ]
    
    # Sort by similarity and popularity, then get the top N recommendations
    recommended_songs = recommended_songs.sort_values(
        by=['similarity', 'track_popularity'], 
        ascending=[False, False]
    ).head(top_n)
    
    return recommended_songs

# Flask API Endpoints
@app.route('/api/recommend/popular', methods=['GET'])
def get_popular_recommendations():
    try:
        top_n = int(request.args.get('top_n', 10))
        recommendations = recommend_popular_songs(top_n)
        return jsonify({
            'success': True,
            'recommendations': recommendations[['track_name', 'track_artist', 'track_popularity', 'YouTube URL']].to_dict('records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recommend/personalized', methods=['GET'])
def get_personalized_recommendations():
    try:
        user_id = request.args.get('user_id', "91cacd43-a2f1-49e8-8fe3-6ac2d66e61e1")
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'user_id parameter is required'
            }), 400
            
        top_n = int(request.args.get('top_n', 10))
        recommendations = recommend_personalized_songs(user_id, top_n)
        
        if recommendations.empty:
            return jsonify({
                'success': False,
                'error': 'No recommendations found for this user'
            }), 404
            
        return jsonify({
            'success': True,
            'recommendations': recommendations[['track_name', 'track_artist', 'track_popularity', 'YouTube URL']].to_dict('records'),
            'user_profile': get_user_profile(user_id)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)