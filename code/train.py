# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import math
import pickle
import os
import re

from __init__ import *
from model_store import build_model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=config)


def save_weights(model, dst_fn):
    weights = model.get_weights()
    with open(dst_fn, 'wb') as f:
        pickle.dump(weights, f)
    return 0


def load_weights(model, weight_fn):
    with open(weight_fn, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    return model


def train_data():
    train = pd.read_csv(TRAIN_PATH).fillna('nan')

    # check_train = train[['sentiment', 'text', 'selected_text']].loc[train['sentiment'] == 'neutral']
    # check_train = check_train[['text', 'selected_text']].loc[check_train['text'].str.contains('http')]
    # check_train_2 = check_train.loc[~check_train['selected_text'].str.contains('http')]

    ct = train.shape[0]
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')
    start_tokens = np.zeros((ct, MAX_LEN), dtype='int32')
    end_tokens = np.zeros((ct, MAX_LEN), dtype='int32')

    for k in range(train.shape[0]):
        # FIND OVERLAP
        text1 = ' ' + ' '.join(train.loc[k, 'text'].split())
        text2 = ' '.join(train.loc[k, 'selected_text'].split())
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

    return input_ids, attention_mask, token_type_ids, start_tokens, end_tokens, train


def test_data():
    test = pd.read_csv(TEST_PATH).fillna('nan')

    ct = test.shape[0]
    input_ids_t = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask_t = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids_t = np.zeros((ct, MAX_LEN), dtype='int32')

    for k in range(test.shape[0]):
        # INPUT_IDS
        text1 = ' ' + ' '.join(test.loc[k, 'text'].split())
        enc = tokenizer.encode(text1)
        s_tok = sentiment_id[test.loc[k, 'sentiment']]
        input_ids_t[k, :len(enc.ids) + 3] = [0, s_tok] + enc.ids + [2]
        attention_mask_t[k, :len(enc.ids) + 3] = 1
    return input_ids_t, attention_mask_t, token_type_ids_t, test


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if (len(a) == 0) & (len(b) == 0):
        return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def train_model(input_ids, attention_mask, token_type_ids, start_tokens, end_tokens):
    loss_val_count = []

    skf = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=SEED)
    for fold, (idxT, idxV) in enumerate(skf.split(input_ids, train.sentiment.values)):
        K.clear_session()
        model, padded_model = build_model(MAX_LEN, PAD_ID, PATH)

        inpT = [input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]]
        targetT = [start_tokens[idxT,], end_tokens[idxT,]]
        inpV = [input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]]
        targetV = [start_tokens[idxV,], end_tokens[idxV,]]
        # sort the validation data
        shuffleV = np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == PAD_ID).sum(), reverse=True))
        inpV = [arr[shuffleV] for arr in inpV]
        targetV = [arr[shuffleV] for arr in targetV]

        best_val_loss = 10000
        print('\nFOLD', '{0} '.format(fold) * 30, '\n')

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
                                callbacks=[], validation_data=(inpV, targetV),
                                shuffle=False)  # don't shuffle in `fit`
            history = history.history['val_loss']

            if best_val_loss >= history[0]:
                best_val_loss = history[0]
                print('Best result in this fold, saving the weight of model...\n')
        save_weights(model, weight_fn.format(VER, fold))
        loss_val_count.append(best_val_loss)
    return loss_val_count


def text_proc(st):
    pattern = re.compile('^i ', re.IGNORECASE)
    st = re.sub(pattern, '', st)
    return st


def load_model_predict(input_ids_t, attention_mask_t, token_type_ids_t):
    preds_start = np.zeros((input_ids_t.shape[0], MAX_LEN))
    preds_end = np.zeros((input_ids_t.shape[0], MAX_LEN))
    # oof_start = np.zeros((input_ids.shape[0], MAX_LEN))
    # oof_end = np.zeros((input_ids.shape[0], MAX_LEN))

    jac = []

    # 6.13 record val train data
    all_record = []
    text_record = []
    sel_text_val = []
    senti_record = []
    all_2 = []

    skf = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=SEED)
    for fold, (idxT, idxV) in enumerate(skf.split(input_ids, train.sentiment.values)):
        print('FOLD{0} '.format(fold) * 10)

        K.clear_session()
        model, padded_model = build_model(MAX_LEN, PAD_ID, PATH)

        print('Loading model...')
        model = load_weights(padded_model, weight_fn.format(VER, fold))

        print('Predicting Test...')
        preds = model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=DISPLAY)
        preds_start += preds[0] / FOLD
        preds_end += preds[1] / FOLD

        # print('Predicting OOF...')
        # oof_start[idxV,], oof_end[idxV,] = padded_model.predict(
        #     [input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]], verbose=DISPLAY)

        """-----------"""

    #     # DISPLAY FOLD JACCARD
    #     all = []
    #
    #     for k in idxV:
    #         a = np.argmax(oof_start[k,])
    #         b = np.argmax(oof_end[k,])
    #         if a > b:
    #             st = train.loc[k, 'text']  # IMPROVE CV/LB with better choice here
    #         else:
    #             text1 = " " + " ".join(train.loc[k, 'text'].split())
    #             enc = tokenizer.encode(text1)
    #             st = tokenizer.decode(enc.ids[a - 2:b - 1])
    #             st = text_proc(st)
    #
    #             # TODO .................
    #
    #         all_record.append(st)
    #         text_record.append(train.loc[k, 'text'])
    #         senti_record.append(train.loc[k, 'sentiment'])
    #         sel_text_val.append(train.loc[k, 'selected_text'])
    #         all_2.append(jaccard(st, train.loc[k, 'selected_text']))
    #
    #         all.append(jaccard(st, train.loc[k, 'selected_text']))
    #     jac.append(np.mean(all))
    #
    # df_record = pd.DataFrame(data={'sentiment': senti_record, 'text': text_record, 'sel_text_true': sel_text_val,
    #                                'selected_text': all_record, 'score': all_2},
    #                          columns=['sentiment', 'text', 'sel_text_true', 'selected_text', 'score'])
    # df_record.to_csv('recode_val_train.csv')

    print('\n')
    print('END ' * 12)

    print('Version:', VER)
    global loss_val_global
    if len(loss_val_global) == 0:
        loss_val_global = ['Null'] * FOLD
    for j in range(len(jac)):
        print('Fold', j, '| jac_acc:', jac[j], 'loss_val:', loss_val_global[j])
    print('       ', 'jac_mean:', np.mean(jac), '\n')

    all_test = []
    for k in range(input_ids_t.shape[0]):
        a = np.argmax(preds_start[k,])
        b = np.argmax(preds_end[k,])
        if a > b:
            st = test.loc[k, 'text']
        else:
            text1 = ' ' + ' '.join(test.loc[k, 'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a - 2:b - 1])
            st = text_proc(st)
            # TODO .................

        all_test.append(st)

    test['selected_text'] = all_test

    test['selected_text'] = test['selected_text'].apply(
        lambda x: x.replace('!!!!', '!') if len(x.split()) == 1 else x)
    test['selected_text'] = test['selected_text'].apply(
        lambda x: x.replace('..', '.') if len(x.split()) == 1 else x)
    test['selected_text'] = test['selected_text'].apply(
        lambda x: x.replace('...', '.') if len(x.split()) == 1 else x)

    test[['textID', 'selected_text']].to_csv('submission.csv', index=False)
    pd.set_option('display.max_colwidth', -1)
    print(test[['text', 'selected_text']].sample(25))
    print()


def score_string(df):
    # df[['sel_text_true', 'selected_text']]
    list_true = df['sel_text_true'].tolist()
    list_gen = df['selected_text'].tolist()
    all = []
    for i in range(len(list_true)):
        all.append(jaccard(list_gen[i], list_true[i]))
    score = np.mean(all)
    print(score)


if __name__ == '__main__':
    SEED = 16028
    VER = 'v28'

    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    loss_val_global = []

    input_ids, attention_mask, token_type_ids, start_tokens, end_tokens, train = train_data()
    input_ids_t, attention_mask_t, token_type_ids_t, test = test_data()

    if IS_KAGGLE == 0:
        loss_val_global = train_model(input_ids, attention_mask, token_type_ids, start_tokens, end_tokens)
        pass
    load_model_predict(input_ids_t, attention_mask_t, token_type_ids_t)

    """-------------------"""
    #
    # # train = pd.read_csv(TRAIN_PATH).fillna('')
    # A_load = pd.read_csv('recode_val_train.csv').dropna()
    # # A_load_proc = A_load.loc[(A_load['sentiment'] != 'neutral') & (A_load['score'] != 1)]
    # # A_load_proc = A_load.loc[(A_load['sentiment'] != 'neutral')]
    # A_load_proc = A_load
    #
    #
    # # a = A_load_proc.loc[(A_load_proc['text'].str.contains('I '))]
    #
    # def del_ele(input):
    #     pattern = re.compile('^i ', re.IGNORECASE)
    #     result = re.sub(pattern, '', input[1])
    #     result = re.sub(r'^ ', '', result)
    #     return result
    #
    #
    # A_load_proc['selected_text'] = A_load_proc[['text', 'selected_text', 'sel_text_true']].apply(del_ele, axis=1)
    # A_load_proc['selected_text'] = A_load_proc['selected_text'].apply(
    #     lambda x: x.replace('!!!!', '!') if len(x.split()) == 1 else x)
    # A_load_proc['selected_text'] = A_load_proc['selected_text'].apply(
    #     lambda x: x.replace('..', '.') if len(x.split()) == 1 else x)
    # A_load_proc['selected_text'] = A_load_proc['selected_text'].apply(
    #     lambda x: x.replace('...', '.') if len(x.split()) == 1 else x)
    # # A_load_proc['sel_text_true'] = A_load_proc['sel_text_true'].
    # score_string(A_load)
    # print()
