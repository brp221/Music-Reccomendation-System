from tensorflow.keras.layers import Dense, Embedding, Flatten, Input, Concatenate, Multiply
# from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import pandas as pd 
import numpy as np 
from keras.utils.vis_utils import plot_model


def create_NCF(users, items, latent_features, learning_rate, dense_layers=[64, 32, 16, 8], reg_layers=[0, 0, 0, 0], reg_mf=0):
    
    #input layer 
    user_input = Input(shape = (1,), name='user_input' )
    item_input = Input(shape = (1,), name='item_input' )

    #embeddings 
    MF_user = Embedding(input_dim = len(users), output_dim = latent_features, embeddings_regularizer= l2(reg_layers[0])) #, embeddings_initializer = emb_initializer , ), 
    MF_item = Embedding(input_dim = len(items), output_dim = latent_features, embeddings_regularizer= l2(reg_layers[0]))
    
    MLP_user = Embedding(input_dim = len(users), output_dim = int(dense_layers[0]/2), embeddings_regularizer= l2(reg_layers[0]))
    MLP_item = Embedding(input_dim = len(items), output_dim = int(dense_layers[0]/2), embeddings_regularizer= l2(reg_layers[0])) 
    
    # flatten and mulitply/concatenate
    MF_user_latent = Flatten()(MF_user(user_input))
    MF_item_latent = Flatten()(MF_item(item_input))
    MF_latent = Multiply() ([ MF_user_latent, MF_item_latent ])
    
    MLP_user_latent = Flatten()(MLP_user(user_input))
    MLP_item_latent = Flatten()(MLP_item(item_input))
    MLP_latent = Concatenate() ([ MLP_user_latent, MLP_item_latent ])

    mlp_vector = MLP_latent
    for i in range(1,len(dense_layers)):
        layer = Dense(dense_layers[i],activity_regularizer=l2(reg_layers[i]),activation='relu',name='layer%d' % i)
        mlp_vector = layer(mlp_vector)
    
    
    predict_layer = Concatenate()([MF_latent, mlp_vector])
    result = Dense(1, activation='sigmoid', name='result') # kernel_initializer='lecun_uniform',
    
    model = Model(inputs=[user_input,item_input], outputs=result(predict_layer))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def model_predict(model, user_lookup_id, items, item_lookup, items_pred, get_item_name = False):
    users_pred = np.full(len(items), user_lookup_id, dtype='int32')
    # items_pred = np.array(items, dtype='int32')
    predictions = pd.DataFrame(model.predict([users_pred,items_pred],batch_size=3000, verbose=0)).reset_index()
    predictions.columns = ['item_id', 'probability']
    #.sort_values(by='0', ascending=False).reset_index()
    if(get_item_name):
        predictions = predictions.merge(item_lookup, right_on='artist_id', left_on='item_id', how = 'left')
    return predictions

def visualize_model(model):
    model.summary()
    plot_model(model)