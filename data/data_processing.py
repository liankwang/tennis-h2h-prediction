import pandas as pd
import glob
import os
import numpy as np

from .compute_player_features import compute_player_features

def create_train_test_sets(matches, player_df, cutoff, comb='diff'):
    """
    Create training and testing sets based on the cutoff year.
    """
    print("Creating train and test sets with cutoff year:", cutoff)

    # Filter player df for training and testing
    player_df_train = player_df[player_df['year'] < cutoff]
    matches_train = matches[matches['year'] < cutoff]
    matches_test = matches[matches['year'] == cutoff]
    
    # Compute player features using train matches
    player_features = compute_player_features(player_df_train)
    print("Done computing player features for train. Shape:", player_features.shape)
    
    # Create train and test set
    train_set = create_set(player_features, matches_train, comb)
    test_set = create_set(player_features, matches_test, comb)
    
    return train_set, test_set

def read_atp_matches(data_dir=None):
    """
    Read all ATP match files from 1968 to 2024 and combine them into a single DataFrame.
    Returns:
        pd.DataFrame: Combined DataFrame containing all ATP matches
    """
    if data_dir is not None:
        print("Reading ATP matches from directory:", data_dir)
    else:
        raise ValueError("data_dir cannot be None. Please provide a valid directory path.")

    # Create a pattern to match all ATP match files from 1968 to 2024
    pattern = os.path.join(data_dir, 'atp_matches_[0-9][0-9][0-9][0-9].csv')
    
    # Get all matching files
    match_files = glob.glob(pattern)
    
    if not match_files:
        print("No files found! Checking if directory exists...")
        if os.path.exists(data_dir):
            print(f"Directory exists. Listing all files in {data_dir}:")
            print(os.listdir(data_dir))
        else:
            print(f"Directory does not exist: {data_dir}")
        return None
    
    # Read and combine all files
    dfs = []
    for file in sorted(match_files):
        try:
            df = pd.read_csv(file)            
            # Add a year column based on the filename
            year = int(os.path.basename(file).split('_')[2].split('.')[0])
            df['year'] = year
            dfs.append(df)
            print(f"Successfully processed {file}")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
            
    
    # Combine all DataFrames
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal number of matches: {len(combined_df)}")
        print(f"Date range: {combined_df['year'].min()} to {combined_df['year'].max()}")
        print(f"Columns in the dataset: {combined_df.columns.tolist()}")
    else:
        print("No files were successfully read")
        return None

    # Expand to one row per player per match
    player_df = expand_matches(combined_df)
    return combined_df, player_df

def create_set(player_features, matches, comb):
    if comb == 'concat':
        print("Creating concatenated set...")
        return create_concat_set(player_features, matches)
    elif comb == 'diff':
        print("Creating difference set...")
        return create_difference_set(player_features, matches)

def create_concat_set(player_features, matches):
    # Get winner features with prefix 'w_'
    feats_w = player_features.add_prefix('w_')
    matches_w = matches[['winner_id', 'loser_id']]
    matches_w = matches_w.merge(feats_w, left_on='winner_id', right_on='w_player_id', how='left')
    matches_w.dropna(subset=['w_player_id'], inplace=True)
    matches_w.drop(columns=['w_player_id', 'w_player_name'], inplace=True)

    # Get loser features with prefix 'l_'
    feats_l = player_features.add_prefix('l_')
    matches_full = matches_w.merge(feats_l, left_on='loser_id', right_on='l_player_id', how='left')
    matches_full.dropna(subset=['l_player_id'], inplace=True)
    matches_full.drop(columns=['l_player_id', 'l_player_name'], inplace=True)

    # Prepare features: winner then loser, label=1
    feats_wl = pd.concat(
        [matches_full.filter(regex='^w_').reset_index(drop=True),
         matches_full.filter(regex='^l_').reset_index(drop=True)],
        axis=1
    )
    feats_wl['label'] = 1

    # Prepare features: loser then winner, label=0
    feats_lw = pd.concat(
        [matches_full.filter(regex='^l_').reset_index(drop=True),
         matches_full.filter(regex='^w_').reset_index(drop=True)],
        axis=1
    )
    feats_lw['label'] = 0

    # Combine both
    combined_set = pd.concat([feats_wl, feats_lw], ignore_index=True)

    # Rename columns to get rid of now-incorrect prefixes
    combined_set = combined_set.rename(
    columns=lambda x: x.replace('w_', '1_', 1) if x.startswith('w_') 
                      else x.replace('l_', '2_', 1) if x.startswith('l_') 
                      else x
    )

    return combined_set

def create_difference_set(player_features, matches):
    # Merge winner features into matches
    matches_w = matches[['winner_id', 'loser_id']].merge(
        player_features, left_on='winner_id', right_on='player_id', how='inner', suffixes=('', '_w')
    )

    # Merge loser features into matches
    matches_full = matches_w.merge(
        player_features, left_on='loser_id', right_on='player_id', how='inner', suffixes=('', '_l')
    )

    # Drop non-numeric features
    matches_full.drop(columns=['player_id', 'winner_id', 'loser_id', 'player_name', 'player_id_l', 'player_name_l',],
                    inplace=True)

    # Drop all non numeric columns TEMPORARY; mostly win rates
    non_numeric_cols = matches_full.select_dtypes(include=['object']).columns
    matches_full.drop(columns=non_numeric_cols, inplace=True)

    # Get feature columns for winner and loser
    feats_w = [col for col in matches_full.columns if not col.endswith('_l')]
    feats_l = [col for col in matches_full.columns if col.endswith('_l')]
    assert len(feats_w) == len(feats_l), "Feature columns do not match in length"

    # Compute difference vectors for each match
    diff = matches_full[feats_w].copy()
    for col1, col2 in zip(feats_w, feats_l):
        diff[col1] = matches_full[col1] - matches_full[col2]

    # Add labels
    diff['label'] = 1

    # Augment dataset with negative labels
    flipped = -diff.drop(columns=['label'], inplace=False).copy()
    flipped['label'] = 0

    # Combine positive and negative samples
    combined_set = pd.concat([diff, flipped], ignore_index=True)

    return combined_set
    
def expand_matches(matches):
    print("Expanding matches to one row per player...")
    match_rows = []

    for i, row in matches.iterrows():
        for role in ['winner', 'loser']:
            # Compute return stats
            opp_prefix = 'l' if role == 'winner' else 'w'
            opp_1st_in = row[f'{opp_prefix}_1stIn']
            opp_1st_won = row[f'{opp_prefix}_1stWon']
            opp_2nd_won = row[f'{opp_prefix}_2ndWon']
            opp_svpt = row[f'{opp_prefix}_svpt']

            return_1st_won = opp_1st_in - opp_1st_won if pd.notna(opp_1st_in) and pd.notna(opp_1st_won) else np.nan
            return_1st_total = opp_1st_in
            return_2nd_won = opp_svpt - opp_1st_in - opp_2nd_won if pd.notna(opp_svpt) and pd.notna(opp_1st_in) and pd.notna(opp_2nd_won) else np.nan
            return_2nd_total = opp_svpt - opp_1st_in if pd.notna(opp_svpt) and pd.notna(opp_1st_in) else np.nan
            return_total_won = return_1st_won + return_2nd_won

            player_stats = {
                'year': row['year'],
                'tourney_id': row['tourney_id'],
                'tourney_name': row['tourney_name'],
                'tourney_date': row['tourney_date'],
                'tourney_level': row['tourney_level'],
                'player_id': row[f'{role}_id'],
                'player_name': row[f'{role}_name'],
                'player_seed': row[f'{role}_seed'],
                'player_rank': row[f'{role}_rank'],
                'player_points': row[f'{role}_rank_points'],
                'player_hand': row[f'{role}_hand'],
                'player_ht': row[f'{role}_ht'],
                'player_ioc': row[f'{role}_ioc'],
                'player_age': row[f'{role}_age'],
                'surface': row['surface'],
                'is_winner': 1 if role == 'winner' else 0,
                'score': row['score'],
                'round': row['round'],
                'minutes': row['minutes'],
                'best_of': row['best_of'],
                'year': row['year'],
                'aces': row[f'{role[0]}_ace'],
                'double_faults': row[f'{role[0]}_df'],
                '1st_in': row[f'{role[0]}_1stIn'],
                '1st_won': row[f'{role[0]}_1stWon'],
                '2nd_won': row[f'{role[0]}_2ndWon'],
                'bp_saved': row[f'{role[0]}_bpSaved'],
                'bp_faced': row[f'{role[0]}_bpFaced'],
                'svpt': row[f'{role[0]}_svpt'],

                # Return stats
                'return_1st_won': return_1st_won,
                'return_1st_total': return_1st_total,
                'return_2nd_won': return_2nd_won,
                'return_2nd_total': return_2nd_total,
                'return_total_won': return_total_won,
                'return_total_pts': opp_svpt
            }
            match_rows.append(player_stats)


    return pd.DataFrame(match_rows)

if __name__ == "__main__":
    # Example usage
    data_dir = 'data/tennis_atp-master'
    matches, player_df = read_atp_matches(data_dir)

    train_set, test_set = create_train_test_sets(matches, player_df, 2020)
    print("Train set shape:", train_set.shape)
    print("Test set shape:", test_set.shape)
