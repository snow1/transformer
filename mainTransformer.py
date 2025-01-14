#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import keras.layers
#print(pd.__version__) #1.3.5
# Read Training Data
train_data = pd.read_excel('data/14-Subjects-Dataset/Training_data.xlsx', header=None)
train_data = np.array(train_data).astype('float32')

# Read Training Labels
train_labels = pd.read_excel('data/14-Subjects-Dataset/Training_labels.xlsx', header=None)
train_labels = np.array(train_labels).astype('float32')
train_labels = np.squeeze(train_labels)

# Read Testing Data
test_data = pd.read_excel('data/14-Subjects-Dataset/Test_data.xlsx', header=None)
test_data = np.array(test_data).astype('float32')

# Read Testing Labels
test_labels = pd.read_excel('data/14-Subjects-Dataset/Test_labels.xlsx', header=None)
test_labels = np.array(test_labels).astype('float32')
test_labels = np.squeeze(test_labels)


class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.5):
        super(TransformerBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim), ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output)
        return out


class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = tf.reshape(x, [-1, maxlen, embed_dim])
        out = x + positions
        return out

maxlen = 3      # 1 Only consider 3 input time points
embed_dim = 640  # Features of each time point
num_heads = 3 #8   # Number of attention heads
ff_dim = 64     # Hidden layer size in feed forward network inside transformer

def main():

    # Input Time-series
    inputs = keras.layers.Input(shape=(maxlen*embed_dim,)) #3*97= 291
    embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim)
    x = embedding_layer(inputs)

    # Encoder Architecture
    transformer_block_1 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
    transformer_block_2 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
    x = transformer_block_1(x)
    x = transformer_block_2(x)

    # Output
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    #print(model.summary())
    #print model architecture
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="binary_crossentropy",
                metrics=['accuracy','loss','precision','recall'])
                #metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall()])

    history = model.fit(
        train_data, train_labels, batch_size=128, epochs=100, validation_data=(test_data, test_labels)
    )

    #print the model accuaracy
    print("Model Accuracy: " + str(model.evaluate(test_data, test_labels)[1])
            +"\nModel Loss: " + str(model.evaluate(test_data, test_labels)[0])
            + "\nModel Precision: " + str(model.evaluate(test_data, test_labels)[2])
            + "\nModel Recall: " + str(model.evaluate(test_data, test_labels)[3]))
    
if __name__ == "__main__":
    main()