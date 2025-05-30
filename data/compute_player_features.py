import pandas as pd
import numpy as np
import numpy as np


def compute_player_features(player_df):
    """
    Compute player features from ATP matches.
    
    Args:
        matches (pd.DataFrame): DataFrame containing ATP matches.
        
    Returns:
        pd.DataFrame: DataFrame with player features.
    """
    print("Now computing player features...")

    # Compute serve and return stats
    player_features = compute_stats(player_df)

    # Deal with missing values
    player_features = process_missing_values(player_features)
    print("Done computing player features. Shape:", player_features.shape)
    print("Features are:", player_features.columns.tolist())

    return player_features
    
def compute_stats(player_df):
    print("Computing serve and return stats...")

    # Calculate serve and return stats
    #print("Num of rows with NA for svpt:", len(player_df[player_df['svpt'].isna()]))
    player_df = player_df[player_df['svpt'].notna()].copy()

    player_df['ace_rate'] = player_df['aces'] / player_df['svpt']
    player_df['df_rate'] = player_df['double_faults'] / player_df['svpt'] 
    player_df['1st_serve_in_pct'] = player_df['1st_in'] / player_df['svpt']
    player_df['1st_serve_win_pct'] = player_df['1st_won'] / player_df['1st_in']
    player_df['2nd_serve_win_pct'] = player_df['2nd_won'] / (player_df['svpt'] - player_df['1st_in'] - player_df['double_faults'])
    player_df['2nd_serve_in_pct'] = (player_df['svpt'] - player_df['1st_in'] - player_df['double_faults']) / (player_df['svpt'] - player_df['1st_in'])
    player_df.loc[player_df['bp_saved'] < 0, 'bp_saved'] = 0
    player_df['bp_saved_pct'] = player_df['bp_saved'] / player_df['bp_faced']

    # Print rows with NA values for 1st_serve_in_pct
    #print("Num of rows with NA for 1st_serve_in_pct:", len(player_df[player_df['1st_serve_in_pct'].isna()]))

    # Calculate return stats
    player_df['return_1st_win_pct'] = player_df['return_1st_won'] / player_df['return_1st_total']
    player_df['return_2nd_win_pct'] = player_df['return_2nd_won'] / player_df['return_2nd_total']
    player_df['return_total_win_pct'] = player_df['return_total_won'] / player_df['return_total_pts']


    # Step 1: Group by player_id and compute means
    player_means = player_df.groupby('player_id').agg({
        'player_name': 'first',
        '1st_serve_win_pct': lambda x: x.dropna().mean(),
        '1st_serve_in_pct': lambda x: x.dropna().mean(),
        '2nd_serve_win_pct': lambda x: x.dropna().mean(),
        '2nd_serve_in_pct': lambda x: x.dropna().mean(),
        'ace_rate': lambda x: x.dropna().mean(),
        'df_rate': lambda x: x.dropna().mean(),
        'bp_saved_pct': lambda x: x.dropna().mean(),
        'return_1st_win_pct': lambda x: x.dropna().mean(),
        'return_2nd_win_pct': lambda x: x.dropna().mean(),
        'return_total_win_pct': lambda x: x.dropna().mean()
    }).reset_index()

    # Rename columns to include '_mean'
    player_means = player_means.rename(columns={
        '1st_serve_win_pct': '1st_serve_win_pct_mean',
        '1st_serve_in_pct': '1st_serve_in_pct_mean',
        '2nd_serve_win_pct': '2nd_serve_win_pct_mean',
        '2nd_serve_in_pct': '2nd_serve_in_pct_mean',
        'ace_rate': 'ace_rate',
        'df_rate': 'df_rate',
        'bp_saved_pct': 'bp_saved_pct',
        'return_1st_win_pct': 'return_1st_win_pct',
        'return_2nd_win_pct': 'return_2nd_win_pct',
        'return_total_win_pct': 'return_total_win_pct'
    })

    # Step 2: Group by player_id and compute stds (only for relevant stats)
    player_stds = player_df.groupby('player_id').agg({
        '1st_serve_win_pct': lambda x: x.dropna().std(),
        '2nd_serve_win_pct': lambda x: x.dropna().std()
    }).reset_index()

    # Rename columns to include '_std'
    player_stds = player_stds.rename(columns={
        '1st_serve_win_pct': '1st_serve_win_pct_std',
        '2nd_serve_win_pct': '2nd_serve_win_pct_std'  
    })

    # Step 3: Merge mean and std results
    player_features = pd.merge(player_means, player_stds, on='player_id')

    # Compute match stats
    print("Computing match stats...")

    # Calculate win rate
    win_rate = player_df.groupby('player_id')['is_winner'].mean().reset_index()
    win_rate.columns = ['player_id', 'win_rate']

    # Calculate average match length (in minutes)
    avg_match_length = player_df.groupby('player_id')['minutes'].mean().reset_index()
    avg_match_length.columns = ['player_id', 'avg_match_length'] 

    # Calculate average number of sets played per match
    player_df['sets_played'] = player_df['score'].apply(get_sets_played)
    sets_per_match = player_df.groupby('player_id')['sets_played'].mean().reset_index()
    sets_per_match.columns = ['player_id', 'sets_per_match']

    # Calculate win rate by round
    round_wins = pd.pivot_table(
        player_df,
        values='is_winner',
        index='player_id',
        columns='round',
        aggfunc='mean'
    ).reset_index()
    round_wins.columns = ['player_id'] + [f'win_rate_{col}' for col in round_wins.columns[1:]]

    # Calculate win rate by surface
    surface_wins = pd.pivot_table(
        player_df,
        values='is_winner', 
        index='player_id',
        columns='surface',
        aggfunc='mean'
    ).reset_index()
    surface_wins.columns = ['player_id'] + [f'win_rate_{col}' for col in surface_wins.columns[1:]]

   # Merge all new features with player_features
    player_features = player_features.merge(win_rate, on='player_id', how='left')
    player_features = player_features.merge(avg_match_length, on='player_id', how='left')
    player_features = player_features.merge(sets_per_match, on='player_id', how='left')
    player_features = player_features.merge(round_wins, on='player_id', how='left')
    player_features = player_features.merge(surface_wins, on='player_id', how='left')

    #### Calculate career level stats
    # Calculate total matches played
    match_counts = player_df.groupby('player_id').size().reset_index()
    match_counts.columns = ['player_id', 'total_matches']

    # Calculate years active
    years_active = player_df.groupby('player_id')['year'].agg(['min', 'max']).reset_index()
    years_active['num_years_active'] = years_active['max'] - years_active['min'] + 1
    years_active = years_active.rename(columns={'min': 'first_year', 'max': 'last_year'})

    # Merge into player_features
    player_features = player_features.merge(match_counts, on='player_id', how='left')
    player_features = player_features.merge(years_active[['player_id', 'num_years_active']], on='player_id', how='left')

    # Merge first and last year played
    # player_features = player_features.merge(years_active[['player_id', 'first_year', 'last_year']], on='player_id', how='left')

    return player_features


def compute_serve_return_stats(player_df):
    print("Computing serve and return stats...")

    # Calculate serve and return stats
    print("Num of rows with NA for svpt:", len(player_df[player_df['svpt'].isna()]))
    player_df = player_df[player_df['svpt'].notna()].copy()

    player_df['ace_rate'] = player_df['aces'] / player_df['svpt']
    player_df['df_rate'] = player_df['double_faults'] / player_df['svpt'] 
    player_df['1st_serve_in_pct'] = player_df['1st_in'] / player_df['svpt']
    player_df['1st_serve_win_pct'] = player_df['1st_won'] / player_df['1st_in']
    player_df['2nd_serve_win_pct'] = player_df['2nd_won'] / (player_df['svpt'] - player_df['1st_in'] - player_df['double_faults'])
    player_df['2nd_serve_in_pct'] = (player_df['svpt'] - player_df['1st_in'] - player_df['double_faults']) / (player_df['svpt'] - player_df['1st_in'])
    player_df.loc[player_df['bp_saved'] < 0, 'bp_saved'] = 0
    player_df['bp_saved_pct'] = player_df['bp_saved'] / player_df['bp_faced']

    # Print rows with NA values for 1st_serve_in_pct
    print("Num of rows with NA for 1st_serve_in_pct:", len(player_df[player_df['1st_serve_in_pct'].isna()]))

    # Calculate return stats
    player_df['return_1st_win_pct'] = player_df['return_1st_won'] / player_df['return_1st_total']
    player_df['return_2nd_win_pct'] = player_df['return_2nd_won'] / player_df['return_2nd_total']
    player_df['return_total_win_pct'] = player_df['return_total_won'] / player_df['return_total_pts']


    # Step 1: Group by player_id and compute means
    player_means = player_df.groupby('player_id').agg({
        'player_name': 'first',
        '1st_serve_win_pct': lambda x: x.dropna().mean(),
        '1st_serve_in_pct': lambda x: x.dropna().mean(),
        '2nd_serve_win_pct': lambda x: x.dropna().mean(),
        '2nd_serve_in_pct': lambda x: x.dropna().mean(),
        'ace_rate': lambda x: x.dropna().mean(),
        'df_rate': lambda x: x.dropna().mean(),
        'bp_saved_pct': lambda x: x.dropna().mean(),
        'return_1st_win_pct': lambda x: x.dropna().mean(),
        'return_2nd_win_pct': lambda x: x.dropna().mean(),
        'return_total_win_pct': lambda x: x.dropna().mean()
    }).reset_index()

    # Rename columns to include '_mean'
    player_means = player_means.rename(columns={
        '1st_serve_win_pct': '1st_serve_win_pct_mean',
        '1st_serve_in_pct': '1st_serve_in_pct_mean',
        '2nd_serve_win_pct': '2nd_serve_win_pct_mean',
        '2nd_serve_in_pct': '2nd_serve_in_pct_mean',
        'ace_rate': 'ace_rate',
        'df_rate': 'df_rate',
        'bp_saved_pct': 'bp_saved_pct',
        'return_1st_win_pct': 'return_1st_win_pct',
        'return_2nd_win_pct': 'return_2nd_win_pct',
        'return_total_win_pct': 'return_total_win_pct'
    })

    # Step 2: Group by player_id and compute stds (only for relevant stats)
    player_stds = player_df.groupby('player_id').agg({
        '1st_serve_win_pct': lambda x: x.dropna().std(),
        '2nd_serve_win_pct': lambda x: x.dropna().std()
    }).reset_index()

    # Rename columns to include '_std'
    player_stds = player_stds.rename(columns={
        '1st_serve_win_pct': '1st_serve_win_pct_std',
        '2nd_serve_win_pct': '2nd_serve_win_pct_std'  
    })

    # Step 3: Merge mean and std results
    player_features = pd.merge(player_means, player_stds, on='player_id')

    return player_features

def compute_match_stats(player_df):
    print("Computing match stats...")
    # Calculate win rate
    win_rate = player_df.groupby('player_id')['is_winner'].mean().reset_index()
    win_rate.columns = ['player_id', 'win_rate']

    # Calculate average match length (in minutes)
    avg_match_length = player_df.groupby('player_id')['minutes'].mean().reset_index()
    avg_match_length.columns = ['player_id', 'avg_match_length'] 

    # Calculate average number of sets played per match
    player_df['sets_played'] = player_df['score'].apply(get_sets_played)
    sets_per_match = player_df.groupby('player_id')['sets_played'].mean().reset_index()
    sets_per_match.columns = ['player_id', 'sets_per_match']

    # Calculate win rate by round
    round_wins = pd.pivot_table(
        player_df,
        values='is_winner',
        index='player_id',
        columns='round',
        aggfunc='mean'
    ).reset_index()
    round_wins.columns = ['player_id'] + [f'win_rate_{col}' for col in round_wins.columns[1:]]

    # Calculate win rate by surface
    surface_wins = pd.pivot_table(
        player_df,
        values='is_winner', 
        index='player_id',
        columns='surface',
        aggfunc='mean'
    ).reset_index()
    surface_wins.columns = ['player_id'] + [f'win_rate_{col}' for col in surface_wins.columns[1:]]

    # Calculate total matches played
    match_counts = player_df.groupby('player_id').size().reset_index()
    match_counts.columns = ['player_id', 'total_matches']

    # Calculate years active
    years_active = player_df.groupby('player_id')['year'].agg(['min', 'max']).reset_index()
    years_active['num_years_active'] = years_active['max'] - years_active['min'] + 1
    years_active = years_active.rename(columns={'min': 'first_year', 'max': 'last_year'})

    ### Merge all stats into a player features DataFrame
    player_features = player_df.drop_duplicates(subset='player_id', keep='first')[['player_id', 'player_name']].reset_index(drop=True)
    player_features = player_features.merge(win_rate, on='player_id', how='left')
    player_features = player_features.merge(avg_match_length, on='player_id', how='left')
    player_features = player_features.merge(sets_per_match, on='player_id', how='left')
    player_features = player_features.merge(round_wins, on='player_id', how='left')
    player_features = player_features.merge(surface_wins, on='player_id', how='left')
    player_features = player_features.merge(match_counts, on='player_id', how='left')
    player_features = player_features.merge(years_active[['player_id', 'num_years_active']], on='player_id', how='left')
    player_features = player_features.merge(years_active[['player_id', 'first_year', 'last_year']], on='player_id', how='left')

    return player_features

def get_sets_played(score):

    """ Helper function to count the average number of sets played ."""

    if pd.isna(score) or score in ['W/O', 'DEF', 'RET']:
        return None
    sets = score.split(' ')
    # Remove any additional info in parentheses
    sets = [s.split('(')[0] for s in sets]
    # Only count completed sets
    completed_sets = [s for s in sets if '-' in s]
    return len(completed_sets)

def process_missing_values(player_features):
    print("Processing missing values...")
    print("Number of players before processing:", len(player_features))

    # Only keep players with at least 10 matches
    player_features = player_features[player_features['total_matches'] >= 10]

    # Remove players with missing 1st serve win percentage (these are mostly players with only 1 match and they didn't serve)
    player_features[player_features['1st_serve_win_pct_mean'].isna()]
    print("Number of players with missing 1st serve win percentage:", len(player_features[player_features['1st_serve_win_pct_mean'].isna()]))

    # Remove some columns
    player_features = player_features.drop(columns=['win_rate_BR', 'win_rate_ER', 'win_rate_RR', 'win_rate_R128', 'win_rate_Carpet'],
                                        errors='ignore')
    # Impute missing values for win rate with 'NA'
    # Get all win_rate columns
    win_rate_cols = [col for col in player_features.columns if col.startswith('win_rate_')]

    # Fill NA values in win_rate columns with 'NA'
    player_features[win_rate_cols] = player_features[win_rate_cols].fillna(0)

    return player_features
