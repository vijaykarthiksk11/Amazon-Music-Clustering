# Amazon-Music-Clustering

A comprehensive Jupyter Notebook pipeline for clustering songs using unsupervised learning techniques like **KMeans** and **PCA**. This project analyzes musical features such as danceability, energy, and tempo to group similar songs together into clusters (e.g., party tracks, chill acoustic, instrumental, etc.).

# 🧪 Dataset Assumptions
Your dataset (e.g., song_dataset.csv) should contain numerical features commonly found in Spotify-style audio data:

Example feature columns:

danceability

energy

loudness

speechiness

acousticness

instrumentalness

liveness

valence

tempo

duration_ms

Optional text columns (e.g. track_name, artist_name) will be dropped during preprocessing.

# 🧠 Clustering Process
1. Load & Clean Data

Drop text columns like track_name, artist_name, track_id

Check for and optionally remove missing values or duplicates

2. Feature Engineering

Select only numerical features relevant to audio characteristics

3. Visualize Distributions

KDE plots for each feature to understand spread

4. Standardize Features

Normalize all values using StandardScaler

5. PCA (Optional)

Reduce dimensions to 2D for visualization

6. Choose Optimal k

Elbow method (Inertia)

Silhouette score method

7. Apply KMeans

Cluster the songs and assign labels

8. Evaluate Clustering

Inertia

Silhouette score

Davies-Bouldin Index

9. Visualize Clusters

Heatmap of average feature values per cluster

PCA scatterplot colored by cluster

Boxplots per feature/cluster

10. Export Results

Save final dataset to songs_with_clusters.csv

# 🙌 Acknowledgments

Built using:

Pandas

Scikit-learn

Seaborn

Matplotlib
