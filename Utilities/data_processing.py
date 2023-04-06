import numpy as np 
import pandas as pd
 

def mask_first(x):
    result = np.ones_like(x)
    result[0] = 0
    return result
   
def train_test_split(df):
    df_test = df.copy(deep=True)
    df_train = df.copy(deep=True)

    # Group by user_id and select only the first item for each user (our holdout).
    df_test = df_test.groupby(['user_id']).first()
    df_test['user_id'] = df_test.index
    df_test = df_test[['user_id', 'artist_id', 'plays']]

    # Remove the same items as we for our test set in our training set.
    mask = df.groupby(['user_id'])['user_id'].transform(mask_first).astype(bool)
    df_train = df.loc[mask]

    return df_train, df_test

def load_dataset(working_dir):

    df = pd.read_csv(working_dir + '/SPOTIFY_DATA/spotify_dataset.csv', on_bad_lines='skip') 
    df.columns = ['user_id', 'artist', 'track', 'playlist'] 
    df = df[['user_id', 'artist', 'track']]
    df = df.dropna().drop_duplicates() # to get rid of potential (artist, song) appearing in multiple playlists of user 

    # create interaction column, by counting number of unique songs a user likes of the artist
    interactions = df.groupby(['user_id', 'artist'])['track'].count().reset_index().rename(columns={'track':'plays'})
    df = df.merge(interactions, on = ['user_id', 'artist'], how = 'inner')[['user_id', 'artist', 'plays']]
    
    artists_per_user = df.groupby(by=["user_id"])['artist'].unique().reset_index().rename(columns={'artist':'unique_artists'})
    artists_per_user['consumed_item_count'] = [len(x) for x in artists_per_user['unique_artists'] ]
    artists_per_user = artists_per_user.query("consumed_item_count > 5")
    valid_users = artists_per_user['user_id'].unique()
    df = df.query("user_id in @valid_users")
    
    
     # Create a numeric user_id and artist_id column
    df['user_id'] = df['user_id'].astype("category").cat.codes
    df['artist_id'] = df['artist'].astype("category").cat.codes
    df = df.drop_duplicates().reset_index()

    # Create a lookup frame to get artist names back in readable form later.
    item_lookup = df[['artist_id', 'artist']].drop_duplicates()
    item_lookup['artist_id'] = item_lookup.artist_id.astype(int)

    # Grab the columns we need in the order we need them.
    df = df[['user_id', 'artist_id', 'plays']]

    # Create training and test sets.
    df_train, df_test = train_test_split(df)

    # Create lists of all unique users and artists
    users = list(np.sort(df.user_id.unique()))
    items = list(np.sort(df.artist_id.unique()))

    # Get all user ids and item ids.
    uids = np.array(df_train.user_id.astype(int).tolist())
    iids = np.array(df_train.artist_id.astype(int).tolist())

    return uids, iids, df_train, df_test, users, items, item_lookup, df

def get_train_instances(uids, iids, num_neg, items):
    """  Returns:
          user_input : A list containing the records' user_id
          item_input : A list containing the records' artist_id
          labels     : A list of labels. 0 indicating that the user has not consumed the item (negative) 
                                       1 indicating that the user has  consumed the item (positive) 
    """
    user_input, item_input, labels = [],[],[]
    zipped = set(zip(uids, iids))
    
    for (u, i) in zip(uids,iids):
        # Add  positive interaction
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
    
        # Sample a number of random negative interactions
        for t in range(num_neg):
            j = np.random.randint(len(items))
            while (u, j) in zipped:
                j = np.random.randint(len(items))
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    
    return user_input, item_input, labels


def get_holdout_item(user_lookup_id, df_test, item_lookup):
    holdout_item = df_test.query("user_id == @user_lookup_id")
    return holdout_item.merge(item_lookup, on = 'artist_id')


def user_has_consumed(df, user_lookup_id, item_lookup):
    user_data = df.query(" user_id == @user_lookup_id ")
    user_data = user_data.merge(item_lookup, on = ['artist_id'], how = 'left')
    return user_data