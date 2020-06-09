# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers
import math
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adadelta

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=config)


def train_data():
    ct = train.shape[0]
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')
    start_tokens = np.zeros((ct, MAX_LEN), dtype='int32')
    end_tokens = np.zeros((ct, MAX_LEN), dtype='int32')

    for k in range(train.shape[0]):

        # FIND OVERLAP
        text1 = " " + " ".join(train.loc[k, 'text'].split())
        text2 = " ".join(train.loc[k, 'selected_text'].split())
        idx = text1.find(text2)
        chars = np.zeros((len(text1)))
        chars[idx:idx + len(text2)] = 1
        if text1[idx - 1] == ' ':
            chars[idx - 1] = 1
        enc = tokenizer.encode(text1)

        # ID_OFFSETS
        offsets = []
        idx = 0
        for t in enc.ids:
            w = tokenizer.decode([t])
            offsets.append((idx, idx + len(w)))
            idx += len(w)

        # START END TOKENS
        toks = []
        for i, (a, b) in enumerate(offsets):
            sm = np.sum(chars[a:b])
            if sm > 0:
                toks.append(i)

        s_tok = sentiment_id[train.loc[k, 'sentiment']]
        input_ids[k, :len(enc.ids) + 3] = [0, s_tok] + enc.ids + [2]
        attention_mask[k, :len(enc.ids) + 3] = 1
        if len(toks) > 0:
            start_tokens[k, toks[0] + 2] = 1
            end_tokens[k, toks[-1] + 2] = 1
    return input_ids, attention_mask, token_type_ids, start_tokens, end_tokens


def test_data():
    test = pd.read_csv('../data/test.csv').fillna('')

    ct = test.shape[0]
    input_ids_t = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask_t = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids_t = np.zeros((ct, MAX_LEN), dtype='int32')

    for k in range(test.shape[0]):
        # INPUT_IDS
        text1 = " " + " ".join(test.loc[k, 'text'].split())
        enc = tokenizer.encode(text1)
        s_tok = sentiment_id[test.loc[k, 'sentiment']]
        input_ids_t[k, :len(enc.ids) + 3] = [0, s_tok] + enc.ids + [2]
        attention_mask_t[k, :len(enc.ids) + 3] = 1
    return input_ids_t, attention_mask_t, token_type_ids_t, test


'''Build roBERTa Model'''


# def save_weights(model, dst_fn):
#     weights = model.get_weights()
#     with open(dst_fn, 'wb') as f:
#         pickle.dump(weights, f)


# def load_weights(model, weight_fn):
#     with open(weight_fn, 'rb') as f:
#         weights = pickle.load(f)
#     model.set_weights(weights)
#     return model


def loss_fn(y_true, y_pred):
    # adjust the targets for sequence bucketing
    ll = tf.shape(y_pred)[1]
    y_true = y_true[:, :ll]
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred,
                                                    from_logits=False, label_smoothing=LABEL_SMOOTHING)
    loss = tf.reduce_mean(loss)
    return loss


def single_model_2(input_x):
    output_x = tf.keras.layers.Dropout(0.2)(input_x[0])
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


def single_model(input_x):
    output_x = tf.keras.layers.Dropout(0.1)(input_x[0])
    output_x = tf.keras.layers.Conv1D(128, 2, padding='same')(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    # output_x = tf.keras.layers.BatchNormalization()(output_x)
    output_x = tf.keras.layers.Conv1D(64, 2, padding='same')(output_x)
    # output_x = tf.keras.layers.LeakyReLU()(output_x)
    # output_x = tf.keras.layers.BatchNormalization()(output_x)
    output_x = tf.keras.layers.Dense(64)(output_x)
    output_x = tf.keras.layers.LeakyReLU()(output_x)
    output_x = tf.keras.layers.Dense(1)(output_x)
    output_x = tf.keras.layers.Flatten()(output_x)
    output_x = tf.keras.layers.Activation('softmax')(output_x)
    return output_x


def build_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)

    lens = MAX_LEN - tf.reduce_sum(padding, -1)
    max_len = tf.reduce_max(lens)
    ids_ = ids[:, :max_len]
    att_ = att[:, :max_len]
    tok_ = tok[:, :max_len]

    config = RobertaConfig.from_pretrained(PATH + 'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH + 'pretrained-roberta-base.h5', config=config)
    x = bert_model(ids_, attention_mask=att_, token_type_ids=tok_)

    x1 = single_model(x)
    x2 = single_model(x)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1, x2])
    # model.summary()
    # optimizer = SGD(lr=3e-5, decay=1e-6, momentum=0.5, nesterov=True)
    optimizer = Adam(learning_rate=3e-5)
    # optimizer = RMSprop(lr=3e-5, rho=0.9, epsilon=1e-06)
    model.compile(loss=loss_fn, optimizer=optimizer)

    # this is required as `model.predict` needs a fixed size!
    x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)
    x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded, x2_padded])
    return model, padded_model


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if (len(a) == 0) & (len(b) == 0):
        return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def train_model(input_ids, input_ids_t, attention_mask, attention_mask_t, token_type_ids, token_type_ids_t,
                start_tokens, end_tokens):
    '''Train roBERTa Model'''
    jac = []
    oof_start = np.zeros((input_ids.shape[0], MAX_LEN))
    oof_end = np.zeros((input_ids.shape[0], MAX_LEN))
    preds_start = np.zeros((input_ids_t.shape[0], MAX_LEN))
    preds_end = np.zeros((input_ids_t.shape[0], MAX_LEN))
    loss_val_count = 100000
    skf = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=SEED)
    for fold, (idxT, idxV) in enumerate(skf.split(input_ids, train.sentiment.values)):
        K.clear_session()
        model, padded_model = build_model()

        # checkpoint
        sv = tf.keras.callbacks.ModelCheckpoint(
            model_path % (VER, fold), monitor='val_loss', verbose=DISPLAY, save_best_only=True,
            save_weights_only=True, mode='auto', save_freq='epoch')

        inpT = [input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]]
        targetT = [start_tokens[idxT,], end_tokens[idxT,]]
        inpV = [input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]]
        targetV = [start_tokens[idxV,], end_tokens[idxV,]]
        # sort the validation data
        shuffleV = np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == PAD_ID).sum(), reverse=True))
        inpV = [arr[shuffleV] for arr in inpV]
        targetV = [arr[shuffleV] for arr in targetV]
        weight_fn = '../data/model_out/%s-roberta-%i.tf' % (VER, fold)

        print('#' * 30)
        print('### FOLD %i' % (fold))
        # print('#' * 25)

        for epoch in range(1, EPOCHS + 1):
            # sort and shuffle: We add random numbers to not have the same order in each epoch
            shuffleT = np.int32(
                sorted(range(len(inpT[0])), key=lambda k: (inpT[0][k] == PAD_ID).sum() + np.random.randint(-3, 3),
                       reverse=True))
            # shuffle in batches, otherwise short batches will always come in the beginning of each epoch
            num_batches = math.ceil(len(shuffleT) / BATCH_SIZE)
            batch_inds = np.random.permutation(num_batches)
            shuffleT_ = []
            for batch_ind in batch_inds:
                shuffleT_.append(shuffleT[batch_ind * BATCH_SIZE: (batch_ind + 1) * BATCH_SIZE])
            shuffleT = np.concatenate(shuffleT_)
            # reorder the input data
            inpT = [arr[shuffleT] for arr in inpT]
            targetT = [arr[shuffleT] for arr in targetT]
            history = model.fit(inpT, targetT, epochs=epoch, initial_epoch=epoch - 1, batch_size=BATCH_SIZE, verbose=1,
                                callbacks=[sv], validation_data=(inpV, targetV),
                                shuffle=False)  # don't shuffle in `fit`
            history = history.history['val_loss']
            if history[0] < loss_val_count:
                # print('Predicting OOF...')
                oof_start[idxV,], oof_end[idxV,] = padded_model.predict(
                    [input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]], verbose=DISPLAY)
                # print('Predicting Test...')
                preds = padded_model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=DISPLAY)
                preds_start += preds[0] / skf.n_splits
                preds_end += preds[1] / skf.n_splits

                # DISPLAY FOLD JACCARD
                all = []
                for k in idxV:
                    a = np.argmax(oof_start[k,])
                    b = np.argmax(oof_end[k,])
                    if a > b:
                        st = train.loc[k, 'text']  # IMPROVE CV/LB with better choice here
                    else:
                        text1 = " " + " ".join(train.loc[k, 'text'].split())
                        enc = tokenizer.encode(text1)
                        st = tokenizer.decode(enc.ids[a - 2:b - 1])
                    all.append(jaccard(st, train.loc[k, 'selected_text']))
                jac.append(np.mean(all))
                temp = np.mean(all)

                print("+" * 30)
                print("Fold:", fold, "Epoch:", epoch, "best_score:", temp, "val_loss:", history[0])
                loss_val_count = history[0]

        # print('Saving model...')
        # save_weights(model, weight_fn)

        # print('Loading model...')
        # model.load_weights(model_path % (VER, fold))
        # load_weights(model, weight_fn)

        # print('Predicting OOF...')
        # oof_start[idxV,], oof_end[idxV,] = padded_model.predict(
        #     [input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]], verbose=DISPLAY)
        #
        # print('Predicting Test...')
        # preds = padded_model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=DISPLAY)
        # preds_start += preds[0] / skf.n_splits
        # preds_end += preds[1] / skf.n_splits
        #
        # # DISPLAY FOLD JACCARD
        # all = []
        # for k in idxV:
        #     a = np.argmax(oof_start[k,])
        #     b = np.argmax(oof_end[k,])
        #     if a > b:
        #         st = train.loc[k, 'text']  # IMPROVE CV/LB with better choice here
        #     else:
        #         text1 = " " + " ".join(train.loc[k, 'text'].split())
        #         enc = tokenizer.encode(text1)
        #         st = tokenizer.decode(enc.ids[a - 2:b - 1])
        #     all.append(jaccard(st, train.loc[k, 'selected_text']))
        # jac.append(np.mean(all))
        # temp = np.mean(all)
        #
        # print('>>>> FOLD %i Jaccard =' % (fold), temp)
        # if temp > best_result:
        #     all2 = []
        #     for k in range(input_ids_t.shape[0]):
        #         a = np.argmax(preds_start[k,])
        #         b = np.argmax(preds_end[k,])
        #         if a > b:
        #             st = test.loc[k, 'text']
        #         else:
        #             text1 = " " + " ".join(test.loc[k, 'text'].split())
        #             enc = tokenizer.encode(text1)
        #             st = tokenizer.decode(enc.ids[a - 2:b - 1])
        #         all2.append(st)
        #     test['selected_text'] = all2
        #     test[['textID', 'selected_text']].to_csv('../data/submission.' + VER + '.' + str(fold) + '.csv',
        #                                              index=False)
        #     pd.set_option('max_colwidth', 60)
        #     best_result = temp
    return preds_start, preds_end


def load_model_predict(input_ids_t, attention_mask_t, token_type_ids_t):
    preds_start = np.zeros((input_ids_t.shape[0], MAX_LEN))
    preds_end = np.zeros((input_ids_t.shape[0], MAX_LEN))
    n_splits = FOLD
    for i in range(5):
        print('#' * 40)
        print('### MODEL %i' % (i + 1))
        print('#' * 40)

        K.clear_session()
        model, padded_model = build_model()

        weight_fn = '../data/model_out/{0}-roberta-{1}.tf'.format(VER, i)
        # weight_fn = '../data/v4_model/'
        print(weight_fn)
        print('Loading model...')
        model.load_weights(weight_fn)

        print('Predicting Test...')
        preds = padded_model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=DISPLAY)
        preds_start += preds[0] / n_splits
        preds_end += preds[1] / n_splits
        print()

    all = []
    for k in range(input_ids_t.shape[0]):
        a = np.argmax(preds_start[k,])
        b = np.argmax(preds_end[k,])
        if a > b:
            st = test.loc[k, 'text']
        else:
            text1 = " " + " ".join(test.loc[k, 'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a - 2:b - 1])
        all.append(st)

    test['selected_text'] = all
    # testing for better score
    test['selected_text'] = test['selected_text'].apply(
        lambda x: x.replace('!!!!', '!') if len(x.split()) == 1 else x)
    test['selected_text'] = test['selected_text'].apply(
        lambda x: x.replace('..', '.') if len(x.split()) == 1 else x)
    test['selected_text'] = test['selected_text'].apply(
        lambda x: x.replace('...', '.') if len(x.split()) == 1 else x)

    test[['textID', 'selected_text']].to_csv('submission.csv', index=False)
    print('Done.')
    # pd.set_option('max_colwidth', 60)
    # test.sample(25)



if __name__ == '__main__':
    '''Load Libraries, Data, Tokenizer'''

    MAX_LEN = 96
    PATH = '../input/'
    tokenizer = tokenizers.ByteLevelBPETokenizer(
        vocab_file=PATH + 'vocab-roberta-base.json',
        merges_file=PATH + 'merges-roberta-base.txt',
        lowercase=True,
        add_prefix_space=True
    )
    FOLD = 5
    EPOCHS = 3  # originally 3
    BATCH_SIZE = 32  # originally 32
    PAD_ID = 1
    SEED = 1616
    LABEL_SMOOTHING = 0.1
    VER = 'v7'
    # fold_num = '1'
    model_path = '../data/model_out/%s-roberta-%i.tf'
    DISPLAY = 0  # USE display=1 FOR INTERACTIVE
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
    train = pd.read_csv('../data/train.csv').fillna('')

    input_ids, attention_mask, token_type_ids, start_tokens, end_tokens = train_data()
    input_ids_t, attention_mask_t, token_type_ids_t, test = test_data()
    train_model(input_ids, input_ids_t, attention_mask, attention_mask_t, token_type_ids,
                token_type_ids_t, start_tokens, end_tokens)
    # load_model_predict(input_ids_t, attention_mask_t, token_type_ids_t)
