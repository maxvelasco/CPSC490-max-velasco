import hdf5_getters
import numpy as np
import os
from glob import glob
import pickle
import pandas as pd


from collections import Counter

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.animation import FuncAnimation

from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap

from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap


# NOTE: This uses a subset of the Million Song Dataset, but it could not be pushed to github due to size constraints.
# Download the subset from here: http://millionsongdataset.com/pages/getting-dataset/#subset
# and store it within the project directory as 'MillionSongSubset'

all_sample_songs_data = []
all_sample_songs_targets = []
RELATIVE_DATASET_ROOT_PATH = './MillionSongSubset'

genre_color_map = {
        'blues': 'blue', 
        'electronic': 'black', 
        'reggae': 'green', 
        'pop': 'magenta', 
        'rock': 'orange', 
        'r&b': 'purple', 
        'country': 'lime', 
        'folk': 'cyan', 
        'hip hop': 'red', 
        'disco': 'gold', 
        'jazz': 'maroon', 
        'classical': 'pink', 
        'punk': 'darkslategrey', 
        'metal': 'olive',
        'Uncategorized': 'white'
    }


def extract_features(file_path):
    
    h5 = hdf5_getters.open_h5_file_read(file_path)

    # Scalar features
    scalar_features = np.array([
        hdf5_getters.get_tempo(h5, 0),
        hdf5_getters.get_key(h5, 0),
        hdf5_getters.get_mode(h5, 0),
        hdf5_getters.get_time_signature(h5, 0),
        hdf5_getters.get_danceability(h5, 0),
        hdf5_getters.get_energy(h5, 0),
        hdf5_getters.get_loudness(h5, 0),
        hdf5_getters.get_duration(h5, 0)
    ])

    # Temporal features -- variable lengths
    bars_start = np.array(hdf5_getters.get_bars_start(h5, 0))
    num_bars = len(bars_start)
    if num_bars > 1:
        intervals = np.diff(bars_start)
        bars_mean_interval = np.mean(intervals)
        bars_std_interval = np.std(intervals)
    else:
        bars_mean_interval = 0.0
        return None


    # --- Sections Start ---
    sections_start = np.array(hdf5_getters.get_sections_start(h5, 0))
    num_sections = len(sections_start)
    if num_sections > 1:
        sections_intervals = np.diff(sections_start)
        sections_mean_interval = np.mean(sections_intervals)
        sections_std_interval = np.std(sections_intervals)
    else:
        sections_mean_interval = 0.0
        sections_std_interval = 0.0
        return None


    # --- Segments Start ---
    segments_start = np.array(hdf5_getters.get_segments_start(h5, 0))
    num_segments = len(segments_start)
    if num_segments > 1:
        segments_intervals = np.diff(segments_start)
        segments_mean_interval = np.mean(segments_intervals)
        segments_std_interval = np.std(segments_intervals)
    else:
        segments_mean_interval = 0.0
        segments_std_interval = 0.0
        return None

    # --- Beats Start ---
    beats_start = np.array(hdf5_getters.get_beats_start(h5, 0))
    num_beats = len(beats_start)
    if num_beats > 1:
        beats_intervals = np.diff(beats_start)
        beats_mean_interval = np.mean(beats_intervals)
        beats_std_interval = np.std(beats_intervals)
    else:
        beats_mean_interval = 0.0
        beats_std_interval = 0.0
        return None

    # --- Tatums Start ---
    tatums_start = np.array(hdf5_getters.get_tatums_start(h5, 0))
    num_tatums = len(tatums_start)
    if num_tatums > 1:
        tatums_intervals = np.diff(tatums_start)
        tatums_mean_interval = np.mean(tatums_intervals)
        tatums_std_interval = np.std(tatums_intervals)
    else:
        tatums_mean_interval = 0.0
        tatums_std_interval = 0.0
        return None

    # --- Segments Loudness Start ---
    segments_loudness_start = np.array(hdf5_getters.get_segments_loudness_start(h5, 0))
    if segments_loudness_start.size > 0:
        loudness_start_mean = np.mean(segments_loudness_start)
        loudness_start_std = np.std(segments_loudness_start)
    else:
        loudness_start_mean = 0.0
        loudness_start_std = 0.0
        return None


    # Timbre
    segments_timbre = np.array(hdf5_getters.get_segments_timbre(h5, 0))
    if segments_timbre.size > 0:
        timbre_mean = np.mean(segments_timbre, axis=0)
        timbre_std = np.std(segments_timbre, axis=0)
    else:
        timbre_mean = np.zeros(12)
        timbre_std = np.zeros(12)
        return None

    # Pitches
    segments_pitches = np.array(hdf5_getters.get_segments_pitches(h5, 0))
    if segments_pitches.size > 0:
        pitches_mean = np.mean(segments_pitches, axis=0)
        pitches_std = np.std(segments_pitches, axis=0)
    else:
        pitches_mean = np.zeros(12)
        pitches_std = np.zeros(12)
        print("Segments pitches is empty.")
        return None


    h5.close()


    song_vector = np.concatenate([
        scalar_features,
        [num_bars, bars_mean_interval, bars_std_interval],
        [num_sections, sections_mean_interval, sections_std_interval],
        [num_segments, segments_mean_interval, segments_std_interval],
        [num_beats, beats_mean_interval, beats_std_interval],
        [num_tatums, tatums_mean_interval, tatums_std_interval],
        timbre_mean,
        timbre_std,
        pitches_mean,
        pitches_std,
        [loudness_start_mean, loudness_start_std]
    ])

    return song_vector
    


def get_all_h5_files():
    h5_files = []
    for root, dirs, files in os.walk(RELATIVE_DATASET_ROOT_PATH):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))

    print(f"Found {len(h5_files)} h5 files")
    return h5_files



def get_magd_genre_dict():
    # Using MSD Allmusic Genre Dataset (Top MAGD) from: https://www.ifs.tuwien.ac.at/mir/msd/download.html#groundtruth
        # Description: https://www.ifs.tuwien.ac.at/mir/msd/TopMAGD.html
        # Data: https://www.ifs.tuwien.ac.at/mir/msd/partitions/msd-topMAGD-genreAssignment.cls
    magd_file_path = '/Users/maxvelasco/Desktop/spotify/MSongsDB/PythonSrc/magd.txt'

    genre_dict = {}
    with open(magd_file_path, 'r') as file:
        for line in file:
            song_id, genre = line.strip().split('\t')
            genre_dict[song_id] = genre
    return genre_dict


def get_top_terms(file_path, n_terms=5):
    '''Get top n_terms from MSD to describe the song's artist'''
    h5 = hdf5_getters.open_h5_file_read(file_path)
    artist_terms = np.array(hdf5_getters.get_artist_terms(h5, 0))
    top_terms = artist_terms[:n_terms] if artist_terms.size > 0 else None
    h5.close()
    return top_terms


def get_id_and_genre(file_path):
    h5 = hdf5_getters.open_h5_file_read(file_path)

    track_id = hdf5_getters.get_track_id(h5, 0).decode('utf-8') # get track id + decode it from binary format

    try:
        genre = magd_genre_dict[track_id]
    except KeyError: 
        genre = 'Undefined'

    h5.close()
    return track_id, genre


def create_genre_mapping(all_terms):
    '''Begin constructing a mapping of terms to primary genres to create the genre hierarchy'''

    genre_hierarchy = {
        'rock': [b'rock', b'classic rock'],
        'hip hop': [b'hip hop', b'rap', b'gangster rap', b'underground rap', b'alternative rap'],
        'jazz': [b'jazz'],
        'r&b': [b'r&b', b'soul', b'funk', b'doo-wop', b'motown'],
        'blues': [b'blues', b'blues-rock'],
        'country': [b'country', b'country rock'],
        'electronic': [b'electronic', b'progressive house', b'house', b'techno', b'electronica', b'dubstep', b'drum and bass', b'uk garage', b'eurodance'],
        'classical': [b'classical'],
        'pop': [b'pop', b'dance pop', b'electropop', b'pop rock'],
        'punk': [b'punk', b'pop punk', b'punk rock'],
        'reggae': [b'reggae', b'ska', b'dancehall', b'dub'],
        'metal': [b'metal', b'heavy metal'],
        'disco': [b'disco', b'italian disco'],
        'folk': [b'folk', b'folk rock', b'folk-pop', b'pop folk'],
    }

    genre_map = {}
    for main_genre, related_terms in genre_hierarchy.items():
        # iterate over all related subgenre terms
        for term in related_terms:
            genre_map[term] = main_genre


    for term in all_terms:
        if term not in genre_map:
            group_term = check_grouping_terms(term)
            if group_term:
                genre_map[term] = group_term

    return genre_map, list(set(genre_map.values()))


def check_grouping_terms(term):
    '''
    Uses a dictionary of genre grouping terms to check whether the term contains a grouping term, which are terms that should be used to group a subgenre into a primary genre 
    (e.g. 'rap' is a grouping term for the primary genre of 'hip hop', so 'gangster rap' is auto-grouped as 'hip hop')
    '''

    genre_grouping_terms = {
        'rock': ['rock'],
        'hip hop': ['rap', 'hip hop'],
        'jazz': ['jazz'],
        'r&b': ['r&b', 'soul', 'funk'],
        'blues': ['blues', 'blues-rock'],
        'country': ['country'],
        'electronic': ['house', 'techno'],
        'reggae': ['reggae', 'ska', 'dancehall'],
        'punk': ['punk'],
        'metal': ['metal'],
        'disco': ['disco'],
        'folk': ['folk'],
        'classical': ['classical']
    }

    for main_genre, related_terms in genre_grouping_terms.items():
        for t in related_terms:
            if t in term.decode('utf-8'):
                return main_genre
    return None

def assign_primary_genre(terms, genre_map):
    '''Assign primary genre based on the terms used to describe the song's artist and the mapping of terms to primary genres'''
    genre_scores = {}
    
    for i, term in enumerate(terms):
        term = term.lower()

        matched_term = None
        for genre_term in genre_map:
            if genre_term in term or term in genre_term:
                matched_term = genre_term
                break
                
        if matched_term and matched_term in genre_map:
            main_genre = genre_map[matched_term]

            # decreasing weight for later terms since earlier terms carry more weight
            score = 1.0 - (i * 0.15) 
            if main_genre in genre_scores:
                genre_scores[main_genre] += score
            else:
                genre_scores[main_genre] = score
                
    if not genre_scores:
        return None
        
    return max(genre_scores.items(), key=lambda x: x[1])[0]



def load_and_process_data(max_files=None, force_reload=False):
    cache_dir="./cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"msd_processed_{max_files}.pkl")
    
    # return cached data if it exists and if force_reload=False
    if os.path.exists(cache_file) and not force_reload:
        print(f"Loading cached processed data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    h5_files = get_all_h5_files()
    print(f"Found {len(h5_files)} h5 files")
    
    all_terms = []
    for i, h5_file in enumerate(h5_files):
        if i % 1000 == 0:
            print(f"Processing file {i}/{len(h5_files)}")
        try:
            terms = get_top_terms(h5_file)
            all_terms.extend(terms)
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            
    term_counts = {}
    for term in all_terms:
        if term:
            term = term.lower()
            term_counts[term] = term_counts.get(term, 0) + 1


    genre_map, unique_genres = create_genre_mapping(term_counts)
    print(f"Mapped to {len(unique_genres)} primary genres: {unique_genres}")

    all_release_years = []
    magd_genre_array = []
    track_id_array = []

    for i, h5_file in enumerate(h5_files):
        if i % 1000 == 0:
            print(f"Processing file {i}/{len(h5_files)}")
        try:
            primary_genre = assign_primary_genre(get_top_terms(h5_file), genre_map)
            song_vector = extract_features(h5_file)
            year = get_release_year(h5_file)
            track_id, genre = get_id_and_genre(h5_file)

            if song_vector is not None:
                all_sample_songs_data.append(song_vector)
                all_sample_songs_targets.append(primary_genre if primary_genre else 'Uncategorized')
                all_release_years.append(year)
                magd_genre_array.append(genre)
                track_id_array.append(track_id)

        except Exception as e:
            print(f"Error processing {h5_file}: {e}")

    print(f"Processed {len(all_sample_songs_data)} songs and got {len(all_sample_songs_targets)} targets, and {len(all_release_years)} release years")

    
    cache_data = {
        'h5_files': h5_files,
        'term_counts': term_counts,
        'genre_map': genre_map,
        'unique_genres': unique_genres,
        'song_vectors': all_sample_songs_data,
        'song_targets': all_sample_songs_targets,
        'song_years': all_release_years,
        'magd_genres': magd_genre_array,
        'track_ids': track_id_array
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
        
    return cache_data


def get_release_year(file_path):
    h5 = hdf5_getters.open_h5_file_read(file_path)
    year = hdf5_getters.get_year(h5, 0)
    h5.close()

    return year




def make_interactive_plot(X_embedded, y_genres):
    '''Creates an interactive plot that allows genres to be toggled on/off -- used in presentations throughout the semester'''
    tsne_df = pd.DataFrame({
        'x': X_embedded[:, 0],
        'y': X_embedded[:, 1],
        'genre': y_genres
    })

    fig = px.scatter(
        tsne_df, x='x', y='y', color='genre',
        color_discrete_map=genre_color_map,
        opacity=0.7, 
        title='Interactive t-SNE Visualization',
        width=1000, height=800,
        hover_data=['genre'],
    )

    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
    fig.update_layout(
        legend_title_text='Music Genre',
        xaxis=dict(title=None, showticklabels=False, showgrid=False),
        yaxis=dict(title=None, showticklabels=False, showgrid=False),
        plot_bgcolor='#f8f9fa'
    )

    fig.write_html('interactive_genres.html')
    fig.show()


def final_tsne(X, targets):
    '''Perform dimensionality reduction using t-SNE'''

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=40,           
        early_exaggeration=2,   
        learning_rate='auto',       
        n_iter=5000,             
        init='pca',
        random_state=73
    )

    X_embedded = tsne.fit_transform(features_scaled)


    # Save and plot results
    np.save("tsne_results.npy", X_embedded)    
    plot_tsne(X_embedded, targets)


def plot_tsne(X_embedded, y_genres):
    plt.figure(figsize=(12, 10), facecolor='white') 
    for genre, color in genre_color_map.items():
        genre_mask = np.array([g == genre for g in y_genres])
        plt.scatter(
            X_embedded[genre_mask, 0], 
            X_embedded[genre_mask, 1], 
            c=color, alpha=0.6, s=15,   
            label=genre
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(False)
    plt.title(f'Dimensionality Reduction via t-SNE', color='black', fontsize=16)
    plt.tight_layout()
    plt.savefig("tsne_plot.png", format='png', dpi=300)
    plt.show()

    
    return X_embedded




def make_timelapse(X_embedded, y_genres):

    years = add_spotify_years()
    unique_years = np.sort(np.unique(years))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('white')
    
    # update the frame
    def update(frame):
        current_year = unique_years[frame]
        
        ax.clear()
        ax.set_facecolor('white')
        ax.set_title(f"t-SNE Visualization — Year: {int(current_year)}", fontsize=16, color='black')
        ax.grid(False)
        
        x_min, x_max = np.min(X_embedded[:, 0]), np.max(X_embedded[:, 0])
        y_min, y_max = np.min(X_embedded[:, 1]), np.max(X_embedded[:, 1])
        ax.set_xlim(x_min - 5, x_max + 5)
        ax.set_ylim(y_min - 5, y_max + 5)
        
        # only plot points with 0 > year <= current_year for each genre
        for genre, color in genre_color_map.items():
            mask = (np.array(y_genres) == genre) & (np.array(years) <= current_year) & (np.array(years) > 0)
            ax.scatter(
                X_embedded[mask, 0], 
                X_embedded[mask, 1], 
                c=color, alpha=0.6, s=15, 
                label=genre
            )

        fig.subplots_adjust(right=0.8, wspace=0.1)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        return ax,

    
    ani = FuncAnimation(fig, update, frames=len(unique_years), repeat=False)
    ani.save("tsne_time_lapse.mp4", fps=1, dpi=150)


def make_timelapse_snapshots(X_embedded, y_genres):

    years = add_spotify_years()
    checkpoint_years = [1950, 1960, 1970, 1985, 1995, 2005, 2011]

    years_arr  = np.array(years)
    genres_arr = np.array(y_genres)

    fig, axes = plt.subplots(
        nrows=4, ncols=2,
        figsize=(12, 10),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    x_min, x_max = X_embedded[:,0].min(), X_embedded[:,0].max()
    y_min, y_max = X_embedded[:,1].min(), X_embedded[:,1].max()

    for ax, ck in zip(axes, checkpoint_years):
        ax.set_title(str(ck), fontsize=10)
        ax.set_xlim(x_min - 5, x_max + 5)
        ax.set_ylim(y_min - 5, y_max + 5)
        ax.set_facecolor('white')
        ax.grid(False)
        
        mask_year = (years_arr > 0) & (years_arr <= ck)
        
        for genre, color in genre_color_map.items():
            sel = mask_year & (genres_arr == genre)
            if ax is axes[0]:
                # Only label on the first subplot
                ax.scatter(
                    X_embedded[sel,0],
                    X_embedded[sel,1],
                    c=color, s=5, alpha=0.6,
                    label=genre
                )
            else:
                ax.scatter(
                    X_embedded[sel,0],
                    X_embedded[sel,1],
                    c=color, s=5, alpha=0.6
                )

    axes[-1].axis('off')

    # adjust the plot because it was cutting off
    fig.subplots_adjust(
        left=0.05,   
        right=0.75,
        top=0.88,   
        bottom=0.05,
        hspace=0.3,
        wspace=0.2
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles, labels,
        title='Genre',
        loc='center left',
        bbox_to_anchor=(0.78, 0.5),
        fontsize=8
    )

    fig.suptitle(
        "t‑SNE Timelapse Snapshots by Release Year",
        fontsize=16,
        y=0.95
    )

    plt.savefig("timelapse_snapshots.png", format='png', dpi=300)
    plt.show()



def gmm_clustering(X_embedded):
    gmm = GaussianMixture(n_components=5, random_state=42)
    gmm.fit(X_embedded)
    cluster_labels_gmm = gmm.predict(X_embedded)


    method = "GMM"
    labels = cluster_labels_gmm
    title = f"t-SNE Clusters using {method}"

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=labels,              
        cmap="tab10",
        s=15,
        alpha=0.8
    )
    plt.colorbar(scatter, label='Cluster Label')
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(False)
    plt.show()


    plot_cluster_genre_distribution(X_embedded, labels, song_targets, method)


    df = pd.DataFrame({'genre': song_targets, 'cluster': labels})
    genre_cluster_percentages = pd.crosstab(df['genre'], df['cluster'], normalize='index') * 100

    print(genre_cluster_percentages)



def plot_cluster_genre_distribution(X_embedded, cluster_labels, targets, algo_name):
    unique_clusters = sorted([c for c in np.unique(cluster_labels) if c != -1])        
        
    genres = sorted(list(set(targets)))
    cluster_composition = pd.DataFrame(0, 
                                        index=[f"Cluster {c}" for c in unique_clusters],
                                        columns=genres)
    
    # calculate cluster centroids
    centroids = []
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        centroids.append(np.mean(X_embedded[cluster_mask], axis=0))
    
    # calculate genre percentages
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        cluster_genres = [targets[i] for i in range(len(targets)) if cluster_mask[i]]
        genre_counts = Counter(cluster_genres)
        
        for genre, count in genre_counts.items():
            cluster_composition.loc[f"Cluster {cluster_id}", genre] = count / cluster_size * 100
            
        print(f"\nCluster {cluster_id} ({cluster_size} items)")
        print("-" * 40)
        for genre, pct in sorted([(g, cluster_composition.loc[f"Cluster {cluster_id}", g]) 
                                    for g in genres if cluster_composition.loc[f"Cluster {cluster_id}", g] > 0],
                                key=lambda x: x[1], reverse=True):
            # Only show genres with at least 1% representation
            if pct >= 1.0:  
                print(f"{genre:<15}: {pct:.1f}%")
    
    
    # # Filter genres with some representation
    active_genres = [g for g in genres if cluster_composition[g].max() >= 5.0]
    filtered_composition = cluster_composition[active_genres]
    
    # transpose to make genres be the rows and the clusters be the columns
    transposed_data = filtered_composition.values.T


    fig, ax = plt.subplots(figsize=(18, 10), facecolor='white')
    im = ax.imshow(transposed_data, aspect='auto', cmap='viridis', origin='lower')

    ax.set_xticks(range(len(unique_clusters)))
    ax.set_xticklabels([f"Cluster {c}" for c in unique_clusters], rotation=45, ha='right')

    ax.set_yticks(range(len(active_genres)))
    ax.set_yticklabels(active_genres, rotation=45, ha='right')

    ax.set_title(f'{algo_name} Cluster Composition by Genre', color='black', fontsize=16)


    # push the whole heatmap over to the left and use more of the right side:
    fig.subplots_adjust(right=0.98, top=0.95)


    for i, g in enumerate(active_genres):        
        for j, c in enumerate(unique_clusters):  
            pct = filtered_composition.iloc[j, i]
            if pct > 0:
                plt.text(j, i, f"{pct:.1f}%", ha='center', va='center', 
                        color='white' if pct < 20 else 'black')


    plt.savefig(f'{algo_name}_cluster_composition.png', dpi=300, bbox_inches='tight')
    plt.show()

    
    plt.figure(figsize=(14, 12), facecolor='white')    
    cluster_cmap = plt.cm.get_cmap('tab20', max(20, len(unique_clusters)))
    
    if -1 in np.unique(cluster_labels):
        noise_mask = cluster_labels == -1
        plt.scatter(
            X_embedded[noise_mask, 0],
            X_embedded[noise_mask, 1],
            color='gray', alpha=0.2, s=10,
            label='Noise'
        )
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        plt.scatter(
            X_embedded[cluster_mask, 0],
            X_embedded[cluster_mask, 1],
            color=cluster_cmap(i % 20), alpha=0.7, s=15,
            label=f'Cluster {cluster_id}'
        )
        
        # find main genre for this cluster
        main_genre = filtered_composition.loc[f"Cluster {cluster_id}"].idxmax()
        main_pct = filtered_composition.loc[f"Cluster {cluster_id}"].max()
        
        # add centroid with genre label
        centroid = centroids[i]
        plt.scatter(centroid[0], centroid[1], s=200, color='black', alpha=0.8)
        plt.annotate(
            f"C{cluster_id}: {main_genre} ({main_pct:.0f}%)",
            xy=(centroid[0], centroid[1]),
            xytext=(0, 0),
            textcoords="offset points",
            ha='center', va='center',
            color='black', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc=cluster_cmap(i % 20), alpha=0.8)
        )
    
    plt.title(f'{algo_name} Clusters with Dominant Genres', color='black', fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"gmm_clustering_5.png", dpi=600)
    plt.show()




def add_spotify_years():
    ''' Many songs have no year labeled, so add the year using Spotify API
        Code for this is in get_spotify_data.py
        
        Started with ~50% years labeled --> ~85% years labeled 
    '''
    
    spotify_df = pd.read_csv("spotify_track_info.csv")
    spotify_df["year"] = spotify_df["year"].fillna(0)
    spotify_df.set_index("msd_id", inplace=True)
    spotify_years = spotify_df["year"].to_dict()

    release_years_full = []

    for i, year in enumerate(song_release_years):
        if year == 0:
            msd_id = track_ids[i]
            spotify_year = spotify_years[msd_id]
            release_years_full.append(spotify_year)

        else:
            release_years_full.append(year)
    
    return release_years_full


def plot_genre_pie(genre_counts):
    '''Creates pie chart showing the distribution of genres in the dataset; created and used for the Poster Presentation'''
    labels = list(genre_counts.keys())
    sizes = [genre_counts[label] for label in labels]
    colors = [genre_color_map.get(label, 'black') for label in labels]

    def customize_autopct(values):
        def autopct(pct):
            total = sum(values)
            count = int(round(pct * total / 100.0))

            if pct < 3.0:
                return ''
            else:
                return f'{count}'
        return autopct


    fig, ax = plt.subplots(figsize=(12, 10))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
            rotatelabels=True,
            autopct=customize_autopct(sizes))

    for txt in texts:
        if txt.get_text() == 'rock':
            txt.set_rotation(0)
            txt.set_ha('center')
            txt.set_va('center')

    for i, t in enumerate(autotexts):
        # txt = t.get_text().strip('%')
        # try:
        #     pct = float(txt)
        # except ValueError:
        #     pct = 0
        
        if labels[i] in ('electronic', 'jazz'):
            t.set_color('white')

    plt.axis('equal')
    plt.title('Genre Breakdown of Million Song Dataset', y=1.05)
    plt.savefig('genre_breakdown_pie_chart.png', dpi=300, bbox_inches='tight')
    plt.show()




if __name__ == "__main__":
    all_files = get_all_h5_files()
    magd_genre_dict = get_magd_genre_dict()

    cache_data = load_and_process_data(force_reload=False)

    h5_files, term_counts, genre_map, primary_genres, song_vectors, song_targets, song_release_years, magd_genres, track_ids = cache_data.values()
    print(f"Found {len(song_vectors)} song vectors and {len(song_targets)} targets")
    

    # NOTE Call to perform dimensionality reduction using t-SNE
    final_tsne(song_vectors, song_targets)
    

    # NOTE Call to load and plot the t-SNE results    
    X_embedded = np.load("tsne_results.npy", allow_pickle=True)
    # plot_tsne(X_embedded, song_targets)

    # NOTE Call to create t-SNE plot that allows you to toggle genres on/off
    # make_interactive_plot(X_embedded, song_targets)

    # NOTE Call these to create the t-SNE timelapse video and snapshots, respectively
    # make_timelapse(X_embedded, song_targets)
    # make_timelapse_snapshots(X_embedded, song_targets)


    # NOTE Call to perform GMM clustering on the t-SNE results
    gmm_clustering(X_embedded)


    # NOTE Call to create pie chart of genre distribution
    # plot_genre_pie(genre_counts=Counter(song_targets))


