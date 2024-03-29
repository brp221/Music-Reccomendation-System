import numpy as np 
import pandas as pd 
from tqdm import tqdm

def user_has_consumed(df, user_lookup_id, item_lookup):
    user_data = df.query(" user_id == @user_lookup_id ")
    user_data = user_data.merge(item_lookup, on = ['artist_id'], how = 'left')
    return user_data

def get_metrics(recommendations, df_train, df_test, user_lookup_id, item_lookup, k=10):
    # retrieve holdout item
    hold_out = df_test.query("user_id == @user_lookup_id")["artist_id"].values[0]

    # get all items the user consumed AND that the model was trained on 
    train_consumed = user_has_consumed(df_train, user_lookup_id, item_lookup)

    # filter out the recommended items which the user already consumed 
    recommendations = recommendations.query("item_id not in @train_consumed").sort_values(by='probability', ascending=False).reset_index()
    
    # get index of holdout item (recommendation position)
    holdouts_index = recommendations.index[recommendations['item_id']==hold_out]

    return holdouts_index[0], recommendations.loc[holdouts_index[0]]['probability']

def model_predict(model, user_lookup_id, items, item_lookup, items_pred, get_item_name = False):
    users_pred = np.full(len(items), user_lookup_id, dtype='int32')
    # items_pred = np.array(items, dtype='int32')
    predictions = pd.DataFrame(model.predict([users_pred,items_pred],batch_size=3000, verbose=0)).reset_index()
    predictions.columns = ['item_id', 'probability']
    #.sort_values(by='0', ascending=False).reset_index()
    if(get_item_name):
        predictions = predictions.merge(item_lookup, right_on='artist_id', left_on='item_id', how = 'left')
    return predictions



def evaluate_hit_rates(model, df_train, df_test, items, item_lookup, items_pred, enrich_df, enrich = True, records = 160000):
    count = 0
    holdout_index_arr, holdout_probability= [],[]
    for user_lookup_id in tqdm(df_train.user_id.unique()):
        # 1st create recommendations 
        recommendations = model_predict(model, user_lookup_id, items, item_lookup, items_pred, False) 
        # 2nd get_metrics
        holdout_index, holdout_prob = get_metrics(recommendations, df_train, df_test, user_lookup_id, item_lookup)
        # 3rd record hit_rate + index of occurance 
        holdout_index_arr.append(holdout_index)
        holdout_probability.append(holdout_prob)
        count += 1
        if(count > records):
            break
    sys_performance = pd.DataFrame.from_dict({"user_id" : df_train.user_id.unique()[:(records + 1)], 
                                   "reccomended_rank" : holdout_index_arr, 
                                   "reccomendation_prob" : holdout_probability})
    if(enrich):
        sys_performance = sys_performance.merge(enrich_df, how = 'left', on = 'user_id')
        sys_performance['correctly_labeled'] =  [0 if x <=0.5 else 1 for x in sys_performance.reccomendation_prob]
        
    return sys_performance


