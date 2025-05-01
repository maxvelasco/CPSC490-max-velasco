import requests
import json

import hdf5_getters
import os
import pandas as pd

DATASET_ROOT_PATH = '/Users/maxvelasco/Desktop/spotify/HDF5/MillionSongSubset'


def get_spotify_track_info(artist, track, access_token):
    '''Use the song's title and artist (from the MSD) to search for the track on Spotify'''

    base_url = "https://api.spotify.com/v1/search"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    params = {
        "q": f'track:"{track}" artist:"{artist}"',
        "type": "track"
    }

    response = requests.get(base_url, headers=headers, params=params)
    
    # Handle errors if necessary
    if response.status_code != 200:
        print(f"Request failed ({response.status_code}): {response.text}")
        return []

    try:
        data = response.json()
    except json.JSONDecodeError:
        # This means the response wasn't valid JSON (e.g. empty or HTML error)
        print("JSON decode error. Response text was:", response.text)
        return []
    
    # If parsing succeeds, proceed
    track_items = data.get("tracks", {}).get("items", [])
    if not track_items:
        return []

    # Collect release_date and id for each track result
    results = []
    for item in track_items:
        release_date = item["album"]["release_date"]
        release_year = release_date.split("-")[0]  # Only take the year -- formatted as YYYY-MM-DD
        track_id = item["id"]
        results.append((release_year, track_id))

    return results

def get_all_h5_files():
    h5_files = []
    for root, dirs, files in os.walk(DATASET_ROOT_PATH):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))

    print(f"Found {len(h5_files)} h5 files")
    return h5_files

def process_all_tracks(all_files):
    i = 0
    all_results = []
    
    # msd_to_spotify_id = {}
    for file_path in all_files:
        if i % 100 == 0:
            print(f"Processing file {i}/{len(all_files)}")

        h5 = hdf5_getters.open_h5_file_read(file_path)
        msd_id = hdf5_getters.get_track_id(h5, 0).decode('utf-8')
        song_title = hdf5_getters.get_title(h5, 0).decode('utf-8')
        song_artist = hdf5_getters.get_artist_name(h5, 0).decode('utf-8')
        h5.close()

        # pass decoded song title and artist to the lookup function
        result_list = get_spotify_track_info(song_artist, song_title, access_token)

        year = result_list[0][0] if result_list != [] else None
        track_id = result_list[0][1] if result_list != [] else None # NOTE: this is the Spotify ID (different than the EchoNest ID used with the MSD)
        
        entry = {"msd_id": msd_id, "year": year, "spotify_id": track_id}
        all_results.append(entry)


        i += 1

    df = pd.DataFrame(all_results)
    df.to_csv("spotify_track_info.csv", index=False)

    
    
if __name__ == "__main__":

    # NOTE: this access token expires after 1 hour. See: https://developer.spotify.com/documentation/web-api/tutorials/getting-started to get your own
    access_token = "BQA90cIDe8Hrwdqs2N3N2qNIS-cybALllQYDkd6jJy0kWgqgUH8PQ5kGqngOZvEqi_csP5QW88IJxdhRJxGx35MACUnMCrduVGenE8K8cB181NmFkOv7z_PWPTIegF3DNC_hevAbA80"

    
    process_all_tracks(get_all_h5_files())

    df = pd.read_csv("spotify_track_info.csv")

    df["year"] = df["year"].fillna(0)
    df.set_index("msd_id", inplace=True)
    id_to_year = df["year"].to_dict()    
