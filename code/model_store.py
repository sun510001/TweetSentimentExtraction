# -*- coding:utf-8 -*-

from transformers import *
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adadelta, Nadam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy

from __init__ import *


def loss_fn(y_true, y_pred):
    # adjust the targets for sequence bucketing
    ll = tf.shape(y_pred)[1]
    y_true = y_true[:, :ll]
    loss = categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=LABEL_SMOOTHING)
    loss = tf.reduce_mean(loss)
    return loss


def single_model_2(input_x):
    output_x = tf.keras.layers.Dropout(0.1)(input_x[0])
    output_x = tf.keras.layers.Conv1D(256, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU(alpha=.001)(output_x)
    output_x = tf.keras.layers.BatchNormalization()(output_x)
    output_x = tf.keras.layers.Conv1D(96, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU(alpha=.001)(output_x)
    output_x = tf.keras.layers.BatchNormalization()(output_x)
    output_x = tf.keras.layers.Dense(1)(output_x)
    output_x = tf.keras.layers.Flatten()(output_x)
    output_x = tf.keras.layers.Activation('softmax')(output_x)
    return output_x


def single_model_1(input_x):
    output_x = tf.keras.layers.Dropout(0.1)(input_x[0])
    output_x = tf.keras.layers.Conv1D(128, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    # output_x = tf.keras.layers.BatchNormalization()(output_x)
    output_x = tf.keras.layers.Conv1D(64, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    # output_x = tf.keras.layers.BatchNormalization()(output_x)
    output_x = tf.keras.layers.Dense(1)(output_x)
    output_x = tf.keras.layers.Flatten()(output_x)
    output_x = tf.keras.layers.Activation('softmax')(output_x)
    return output_x


def single_model_3(input_x):
    output_x = tf.keras.layers.Dropout(0.1)(input_x[0])
    output_x = tf.keras.layers.Conv1D(128, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    #     output_x = tf.keras.layers.BatchNormalization()(output_x)
    # output_x = tf.keras.layers.Conv1D(64, 2, padding='same')(output_x)
    # output_x = tf.keras.layers.LeakyReLU()(output_x)
    # output_x = tf.keras.layers.BatchNormalization()(output_x)
    output_x = tf.keras.layers.Dense(128)(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    output_x = tf.keras.layers.Dense(1)(output_x)
    output_x = tf.keras.layers.Flatten()(output_x)
    output_x = tf.keras.layers.Activation('softmax')(output_x)
    return output_x


def single_model_4(input_x):  # best
    output_x = tf.keras.layers.Dropout(0.1)(input_x[0])
    output_x = tf.keras.layers.Conv1D(768, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    # output_x = tf.keras.layers.BatchNormalization()(output_x)
    output_x = tf.keras.layers.Conv1D(96, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    # output_x = tf.keras.layers.BatchNormalization()(output_x)
    output_x = tf.keras.layers.Dense(96)(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    output_x = tf.keras.layers.Dense(1)(output_x)
    output_x = tf.keras.layers.Flatten()(output_x)
    output_x = tf.keras.layers.Activation('softmax')(output_x)
    return output_x


def single_model_5(input_x):
    output_x = tf.keras.layers.Dropout(0.1)(input_x[0])
    output_x = tf.keras.layers.Conv1D(768, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    # output_x = tf.keras.layers.BatchNormalization()(output_x)
    output_x = tf.keras.layers.Conv1D(256, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    output_x = tf.keras.layers.Conv1D(96, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    # output_x = tf.keras.layers.BatchNormalization()(output_x)
    # output_x = tf.keras.layers.Dense(64)(output_x)
    # output_x = tf.keras.layers.LeakyReLU()(output_x)

    # output_x = tf.keras.layers.Flatten()(output_x)
    output_x = tf.keras.layers.Dropout(0.2)(output_x)
    output_x = tf.keras.layers.Dense(1, activation='sigmoid')(output_x)
    # output_x = tf.keras.layers.Activation('softmax')(output_x)
    return output_x


def single_model_best(input_x):
    # 0.707
    output_x = tf.keras.layers.Dropout(0.1)(input_x[0])
    output_x = tf.keras.layers.Conv1D(768, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    output_x = tf.keras.layers.Conv1D(64, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    # output_x = tf.keras.layers.Dense(64)(output_x)
    # output_x = tf.keras.layers.LeakyReLU()(output_x)
    output_x = tf.keras.layers.Dense(1)(output_x)
    output_x = tf.keras.layers.Flatten()(output_x)
    output_x = tf.keras.layers.Activation('softmax')(output_x)
    return output_x


def single_model(input_x):
    output_x = tf.keras.layers.Dropout(0.1)(input_x[0])
    output_x = tf.keras.layers.Conv1D(768, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    # output_x = tf.keras.layers.Conv1D(96, 2, padding='same')(output_x)
    # output_x = tf.keras.layers.LeakyReLU()(output_x)

    output_x = tf.keras.layers.Dense(1)(output_x)
    output_x = tf.keras.layers.Flatten()(output_x)
    output_x = tf.keras.layers.Activation('softmax')(output_x)
    return output_x


def single_model_large(input_x):
    output_x = tf.keras.layers.Dropout(0.1)(input_x[0])
    # output_x = tf.keras.layers.Conv1D(128, 2, padding='same')(output_x)
    # output_x = tf.keras.layers.LeakyReLU()(output_x)
    output_x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(output_x)
    # output_x = tf.keras.activations.tanh(output_x)
    # output_x = tf.keras.layers.Flatten()(output_x)
    # output_x = tf.keras.layers.GRU(768, dropout=0.2, return_sequences=True)(output_x)
    # output_x = tf.keras.layers.Conv1D(96, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    # output_x = tf.keras.layers.Dense(128)(output_x)
    # output_x = tf.keras.layers.LeakyReLU()(output_x)
    output_x = tf.keras.layers.Dense(1)(output_x)
    output_x = tf.keras.layers.Flatten()(output_x)
    output_x = tf.keras.layers.Activation('softmax')(output_x)
    return output_x


def build_model(MAX_LEN, PAD_ID, PATH):
    ids = layers.Input((MAX_LEN,), dtype=tf.int32)
    att = layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = layers.Input((MAX_LEN,), dtype=tf.int32)
    padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)

    lens = MAX_LEN - tf.reduce_sum(padding, -1)
    max_len = tf.reduce_max(lens)
    ids_ = ids[:, :max_len]
    att_ = att[:, :max_len]
    tok_ = tok[:, :max_len]

    config = RobertaConfig.from_pretrained(PATH + 'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH + 'pretrained-roberta-base.h5', config=config)
    # config = RobertaConfig.from_pretrained(PATH + "config.json")
    # bert_model = TFRobertaModel.from_pretrained(PATH + "tf_model.h5", config=config)

    x = bert_model(ids_, attention_mask=att_, token_type_ids=tok_)

    # x1 = single_model(x)
    # x2 = single_model(x)
    x1 = single_model_large(x)
    x2 = single_model_large(x)

    model = Model(inputs=[ids, att, tok], outputs=[x1, x2])
    # model.summary()
    # optimizer = SGD(lr=3e-5, decay=1e-6, momentum=0.5, nesterov=True)
    # optimizer = Nadam(learning_rate=3e-5)
    optimizer = Adam(learning_rate=3e-5)
    # optimizer = RMSprop()
    model.compile(loss=loss_fn, optimizer=optimizer)
    # model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # this is required as `model.predict` needs a fixed size!
    x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)
    x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    padded_model = Model(inputs=[ids, att, tok], outputs=[x1_padded, x2_padded])
    return model, padded_model
