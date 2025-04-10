import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load datasets
history_df = pd.read_csv('history.csv')
liked_df = pd.read_csv('liked.csv')
songs_df = pd.read_csv('songs_dataset_with_filled_thumbnails.csv')

# Combine liked songs and history for the specific user
def get_user_songs(user_id):
    # Get liked songs
    user_liked_songs = liked_df[liked_df['user_id'] == user_id][['song_title', 'song_artist']].rename(
        columns={'song_title': 'song_name', 'song_artist': 'artist_name'})
    # Get history songs
    user_history_songs = history_df[history_df['user_id'] == user_id][['song_name', 'artist_name']]
    
    # Combine liked and history songs
    user_songs = pd.concat([user_liked_songs, user_history_songs]).drop_duplicates()
    
    # print(user_songs)
    return user_songs

# Function to recommend songs based on user's liked and history songs
def recommend_songs(user_id, top_n=10):
    # Get the user's liked and history songs
    user_songs = get_user_songs(user_id)
    
    if user_songs.empty:
        print("No liked or history songs found for this user.")
        return pd.DataFrame()  # Return empty DataFrame if no songs found
    
    # Filter the songs dataset to include only the user's liked and history songs
    user_song_details = songs_df[songs_df['track_name'].isin(user_songs['song_name'])] | songs_df['track_artist'].isin(user_songs['artist_name']) | songs_df['playlist_genre'].isin(user_songs['song_name']) 
    print(user_song_details)
    if user_song_details.empty:
        print("No matching songs found in the songs dataset.")
        return pd.DataFrame()  # Return empty DataFrame if no matching songs found
    
    # Extract features for content-based filtering
    feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    user_song_features = user_song_details[feature_columns]
    
    if user_song_features.empty:
        print("No features found for the user's songs.")
        return pd.DataFrame()  # Return empty DataFrame if no features found
    
    # Normalize the features
    scaler = MinMaxScaler()
    try:
        user_song_features_normalized = scaler.fit_transform(user_song_features)
    except ValueError as e:
        print(f"Error during normalization: {e}")
        return pd.DataFrame()  # Return empty DataFrame if normalization fails
    
    # Compute the average feature vector for the user's songs
    user_profile = user_song_features_normalized.mean(axis=0).reshape(1, -1)
    
    # Compute similarity between the user profile and all songs in the dataset
    all_song_features = songs_df[feature_columns]
    all_song_features_normalized = scaler.transform(all_song_features)
    similarity_scores = cosine_similarity(user_profile, all_song_features_normalized).flatten()
    
    # Add similarity scores to the songs dataset
    songs_df['similarity'] = similarity_scores
    
    # Exclude songs already liked or listened to by the user
    recommended_songs = songs_df[~songs_df['track_name'].isin(user_songs['song_name'])]
    
    # Sort by similarity and get the top N recommendations
    recommended_songs = recommended_songs.sort_values(by='similarity', ascending=False).head(top_n)
    
    return recommended_songs

# Example usage
user_id = '91cacd43-a2f1-49e8-8fe3-6ac2d66e61e1'  # Replace with actual user ID
recommendations = recommend_songs(user_id)
if not recommendations.empty:
    print(recommendations[['track_name', 'track_artist', 'track_popularity', 'YouTube URL']])
else:
    print("No recommendations found for this user.")