import numpy as np 
import pandas as pd
 

# def get_negatives(uids, iids, items, df_test):
#     negativeList = []
#     test_u = df_test['user_id'].values.tolist()
#     test_i = df_test['artist_id'].values.tolist()

#     test_ratings = list(zip(test_u, test_i))
#     zipped = set(zip(uids, iids))

#     for (u, i) in test_ratings:
#         negatives = []
#         negatives.append((u, i))
#         for t in range(100):
#             j = np.random.randint(len(items)) # Get random item id.
#             while (u, j) in zipped: # Check if there is an interaction
#                 j = np.random.randint(len(items)) # If yes, generate a new item id
#             negatives.append(j) # Once a negative interaction is found we add it.
#         negativeList.append(negatives)

#     return pd.DataFrame(negativeList)

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

