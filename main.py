import numpy as np 
import pandas as pd 
import os 
import seaborn as sns 
import matplotlib.pyplot as plt 

os.chdir("/Users/bratislavpetkovic/Desktop/WGU/CAPSTONE/Song_Recommendation_Engine/")
from Utilities.data_processing import load_dataset, get_train_instances
from Utilities.model_utilities import create_NCF, model_predict
from Utilities.model_evaluation import user_has_consumed, evaluate_hit_rates, get_metrics



#__________________________________________DATA__________________________________________
uids, iids, df_train, df_test, users, items, item_lookup, df = load_dataset(os.getcwd())
user_input, item_input, labels = get_train_instances( uids, iids, 4, items)
X = [np.array(user_input), np.array(item_input)]
y = np.array(labels)
# prepped_data = pd.DataFrame.from_dict({"user": user_input, "item": item_input, "label":labels})

user_input_8, item_input_8, labels_8 = get_train_instances( uids, iids, 8, items)
X_8 = [np.array(user_input_8), np.array(item_input_8)]
y_8 = np.array(labels_8)

user_input_2, item_input_2, labels_2 = get_train_instances( uids, iids, 2, items)
X_2 = [np.array(user_input_2), np.array(item_input_2)]
y_2 = np.array(labels_2)


#__________________________________________NeuMF MODEL_________________________________________

# NeuMF model initiation 
latent_features = 8
epochs = 12
batch_size = 10000
learning_rate = 0.05

model = create_NCF(users, items, latent_features=8, learning_rate = 0.05, dense_layers=[64, 32, 16, 8], reg_layers=[0, 0, 0, 0], reg_mf=0)
hist = model.fit( X, y , batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)

model_heavy = create_NCF(users, items, latent_features=12, learning_rate = 0.05, dense_layers=[128, 64, 32, 16], reg_layers=[0, 0, 0, 0], reg_mf=0)
hist_heavy = model_heavy.fit( X, y , batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
rec_sys_heavy_performance = evaluate_hit_rates(model_heavy, df_train, df_test, items, item_lookup, items_pred)
rec_sys_heavy_performance = rec_sys_heavy_performance.merge(enrich_df, how = 'left', on = 'user_id')
rec_sys_heavy_performance['correctly_labeled'] =  [0 if x <=0.5 else 1 for x in rec_sys_heavy_performance.reccomendation_prob]
print("Average Probability: ", np.mean(rec_sys_heavy_performance.reccomendation_prob))
print("Accuracy Classification: ", np.mean(rec_sys_heavy_performance.correctly_labeled))


model_heavy_2 = create_NCF(users, items, latent_features=16, learning_rate = 0.05, dense_layers=[128, 64, 32, 16], reg_layers=[0, 0, 0, 0], reg_mf=0)
hist_heavy_2 = model_heavy_2.fit( X, y , batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)

model_heavy_3 = create_NCF(users, items, latent_features=24, learning_rate = 0.05, dense_layers=[256, 128, 64, 32], reg_layers=[0, 0, 0, 0], reg_mf=0)
hist_heavy_3 = model_heavy_3.fit( X_8, y_8 , batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
rec_sys_heavy_3_performance = evaluate_hit_rates(model_heavy_3, df_train, df_test, items, item_lookup, items_pred)
rec_sys_heavy_3_performance = rec_sys_heavy_3_performance.merge(enrich_df, how = 'left', on = 'user_id')
rec_sys_heavy_3_performance['correctly_labeled'] =  [0 if x <=0.5 else 1 for x in rec_sys_heavy_3_performance.reccomendation_prob]
print("Average Probability: ", np.mean(rec_sys_heavy_3_performance.reccomendation_prob))
print("Accuracy Classification: ", np.mean(rec_sys_heavy_3_performance.correctly_labeled))

model_heavy_3 = create_NCF(users, items, latent_features=24, learning_rate = 0.05, dense_layers=[256, 128, 64, 32], reg_layers=[0, 0, 0, 0], reg_mf=0)
hist_heavy_3 = model_heavy_3.fit( X_2, y_2 , batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
rec_sys_heavy_3_performance = evaluate_hit_rates(model_heavy_3, df_train, df_test, items, item_lookup, items_pred)
rec_sys_heavy_3_performance = rec_sys_heavy_3_performance.merge(enrich_df, how = 'left', on = 'user_id')
rec_sys_heavy_3_performance['correctly_labeled'] =  [0 if x <=0.5 else 1 for x in rec_sys_heavy_3_performance.reccomendation_prob]
print("Average Probability: ", np.mean(rec_sys_heavy_3_performance.reccomendation_prob))
print("Accuracy Classification: ", np.mean(rec_sys_heavy_3_performance.correctly_labeled))


model_heaviest = create_NCF(users, items, latent_features=12, learning_rate = 0.05, dense_layers=[256, 128, 64, 32, 16], reg_layers=[0, 0, 0, 0, 0], reg_mf=0)
hist_heaviest = model_heaviest.fit( X, y , batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)

#__________________________________________MODEL EVALUATION_________________________________________

items_pred = np.array(items, dtype='int32')

rec_sys_performance = evaluate_hit_rates(model, df_train, df_test, items, item_lookup, items_pred)
enrich_df = df.groupby(by=['user_id'])['artist_id'].count().reset_index().rename(columns={'artist_id':'#_items_consumed'})
rec_sys_performance = rec_sys_performance.merge(enrich_df, how = 'left', on = 'user_id')
rec_sys_performance['correctly_labeled'] =  [0 if x <=0.5 else 1 for x in rec_sys_performance.reccomendation_prob]
print("Average Probability: ", np.mean(rec_sys_performance.reccomendation_prob))
print("Accuracy Classification: ", np.mean(rec_sys_performance.correctly_labeled))

rec_sys_heavy_performance = evaluate_hit_rates(model_heavy, df_train, df_test, items, item_lookup, items_pred)
rec_sys_heavy_performance = rec_sys_heavy_performance.merge(enrich_df, how = 'left', on = 'user_id')
rec_sys_heavy_performance['correctly_labeled'] =  [0 if x <=0.5 else 1 for x in rec_sys_heavy_performance.reccomendation_prob]
print("Average Probability: ", np.mean(rec_sys_heavy_performance.reccomendation_prob))
print("Accuracy Classification: ", np.mean(rec_sys_heavy_performance.correctly_labeled))

rec_sys_heavy_2_performance = evaluate_hit_rates(model_heavy_2, df_train, df_test, items, item_lookup, items_pred)
rec_sys_heavy_2_performance = rec_sys_heavy_2_performance.merge(enrich_df, how = 'left', on = 'user_id')
rec_sys_heavy_2_performance['correctly_labeled'] =  [0 if x <=0.5 else 1 for x in rec_sys_heavy_2_performance.reccomendation_prob]
print("Average Probability: ", np.mean(rec_sys_heavy_2_performance.reccomendation_prob))
print("Accuracy Classification: ", np.mean(rec_sys_heavy_2_performance.correctly_labeled))

rec_sys_heavy_3_performance = evaluate_hit_rates(model_heavy_3, df_train, df_test, items, item_lookup, items_pred)
rec_sys_heavy_3_performance = rec_sys_heavy_3_performance.merge(enrich_df, how = 'left', on = 'user_id')
rec_sys_heavy_3_performance['correctly_labeled'] =  [0 if x <=0.5 else 1 for x in rec_sys_heavy_3_performance.reccomendation_prob]
print("Average Probability: ", np.mean(rec_sys_heavy_3_performance.reccomendation_prob))
print("Accuracy Classification: ", np.mean(rec_sys_heavy_3_performance.correctly_labeled))

rec_sys_heaviest_performance = evaluate_hit_rates(model_heaviest, df_train, df_test, items, item_lookup, items_pred)
rec_sys_heaviest_performance = rec_sys_heaviest_performance.merge(enrich_df, how = 'left', on = 'user_id')
rec_sys_heaviest_performance['correctly_labeled'] =  [0 if x <=0.5 else 1 for x in rec_sys_heaviest_performance.reccomendation_prob]
print("Average Probability: ", np.mean(rec_sys_heaviest_performance.reccomendation_prob))
print("Accuracy Classification: ", np.mean(rec_sys_heaviest_performance.correctly_labeled))







# sns.histplot(rec_sys_performance, x='reccomendation_prob')s

# sns.regplot(rec_sys_performance, x='reccomendation_prob', y = '#_items_consumed', fit_reg=True,
#             scatter_kws = {"color": "black", "alpha": 0.5},
#             line_kws = {"color": "red"})
# plt.ylim(0, 5000)

# user_lookup_id = 9752

# already_consumed = user_has_consumed(df, user_lookup_id, item_lookup)

# reccomendation_69 = model_predict(model, user_lookup_id, items, item_lookup, items_pred, get_item_name = False)
# holdout_index, holdout_prob = get_metrics(reccomendation_69, df_train, df_test, user_lookup_id, item_lookup)
