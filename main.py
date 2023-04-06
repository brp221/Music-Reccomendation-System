import numpy as np 
import pandas as pd 
import os 
import seaborn as sns 
import matplotlib.pyplot as plt 

os.chdir("/Users/bratislavpetkovic/Desktop/WGU/CAPSTONE/Song_Recommendation_Engine/")
from Utilities.data_processing import load_dataset, get_train_instances, get_holdout_item, user_has_consumed
from Utilities.model_utilities import create_NCF, model_predict
from Utilities.model_evaluation import evaluate_hit_rates, get_metrics




#__________________________________________DATA__________________________________________
uids, iids, df_train, df_test, users, items, item_lookup, df = load_dataset(os.getcwd())
user_input, item_input, labels = get_train_instances( uids, iids, 4, items)
X = [np.array(user_input), np.array(item_input)]
y = np.array(labels)

X_test = [np.array(df_test.user_id), np.array(df_test.artist_id)]
y_test = np.ones(len(df_test.user_id))

df_train.to_csv("df_train.csv", )
df_test.to_csv("df_test.csv")

# prepped_data = pd.DataFrame.from_dict({"user": user_input, "item": item_input, "label":labels})


items_pred = np.array(items, dtype='int32')
enrich_df = df.groupby(by=['user_id'])['artist_id'].count().reset_index().rename(columns={'artist_id':'#_items_consumed'})

#__________________________________________NeuMF MODEL_________________________________________

# NeuMF model initiation 
epochs = 12
batch_size = 10000
learning_rate = 0.01
bs_2_lr = batch_size / learning_rate


model_8 = create_NCF(users, items, latent_features=64, learning_rate = learning_rate, dense_layers=[512, 256, 128, 64], reg_layers=[0.15, 0.1, 0.05, 0.0, 0.0], reg_mf=0)
hist_8 = model_8.fit( X, y , batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, validation_data = [X_test, y_test])
print(model_8.summary())
sys_metrics_8 = evaluate_hit_rates(model_8, df_train, df_test, items, item_lookup, items_pred, enrich_df)
print("Average Probability: ", np.mean(sys_metrics_8.reccomendation_prob))
print("Accuracy Classification: ", np.mean(sys_metrics_8.correctly_labeled))

#__________________________________________MODEL EVALUATION_________________________________________



user_lookup_id = 6
holdout_item = get_holdout_item(user_lookup_id, df_test, item_lookup).artist_id[0]
print(holdout_item)
consumed_df = user_has_consumed(df, user_lookup_id, item_lookup)
print(consumed_df)

predictions = model_predict(model_8, user_lookup_id, items, item_lookup, items_pred, get_item_name = True )
prediction_holdout = predictions.query("artist_id == @holdout_item")
print(prediction_holdout)

sys_metrics_8 = evaluate_hit_rates(model_8, df_train, df_test, items, item_lookup, items_pred, enrich_df)
print("Average Probability: ", np.mean(sys_metrics_8.reccomendation_prob))
print("Accuracy Classification: ", np.mean(sys_metrics_8.correctly_labeled))


#__________________________________________EXPLORATORY DATA ANALYSIS_________________________________________

sys_metrics_small = sys_metrics_8[sys_metrics_8["#_items_consumed"] < 500]
sys_metrics_small['bins'] = pd.cut(sys_metrics_8["#_items_consumed"], [0,25,50,75,100, 150, 200, 250, 300, 350, 500], include_lowest=True,right = False)
consumption_vs_accuracy = sys_metrics_small.groupby(["bins"])["correctly_labeled"].mean().reset_index()

sns.barplot(consumption_vs_accuracy, x = 'bins', y = 'correctly_labeled')
plt.xlabel('#of items consumed')
plt.ylabel('recall accuracy')
plt.xticks(fontsize=10, rotation=25)
plt.title('Consumption vs Recommendation Accuracy', fontsize=10)


sns.histplot(sys_metrics_8['#_items_consumed'])
plt.xlim(0, 5000)


print("TESTING  Loss: {0}  Accuracy: {1}".format(*model_8.evaluate(X_test, y_test, verbose=0)))

training_loss = hist_8.history['loss']
test_loss = hist_8.history['val_loss']

training_acc = hist_8.history['accuracy']
test_acc = hist_8.history['val_accuracy']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('LOSS vs EPOCHS')
plt.show();

# Visualize accuracy history
plt.plot(epoch_count, training_acc, 'r--')
plt.plot(epoch_count, test_acc, 'b-')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('ACCURACY vs EPOCHS')
plt.show();


