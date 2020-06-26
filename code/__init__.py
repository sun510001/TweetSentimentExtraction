# -*- coding:utf-8 -*-

import tokenizers


IS_KAGGLE = 0
M_PATH = 'modelv40'  # kaggle model folder name
MAX_LEN = 96
FOLD = 5
EPOCHS = 5  # originally 3
BATCH_SIZE = 32  # originally 32
PAD_ID = 1
LABEL_SMOOTHING = 0.1
DISPLAY = 1  # USE display=1 FOR INTERACTIVE


sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
if IS_KAGGLE == 0:
    TEST_PATH = '../data/test.csv'
    TRAIN_PATH = '../data/train.csv'
    MODEL_FOLDER = '../data/model_out/'  # v2635-best/
    # PATH = '../input/tf-roberta-large/'
    PATH = '../input/tf-roberta-base/'
else:
    TEST_PATH = '../input/tweet-sentiment-extraction/test.csv'
    TRAIN_PATH = '../input/tweet-sentiment-extraction/train.csv'
    MODEL_FOLDER = '../input/{0}/'.format(M_PATH)
    PATH = '../input/tf-roberta/'

weight_fn = MODEL_FOLDER + '{0}-roberta-{1}.h5'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=PATH + 'vocab-roberta-base.json',
    merges_file=PATH + 'merges-roberta-base.txt',
    # vocab_file=PATH + 'vocab.json',
    # merges_file=PATH + 'merges.txt',
    lowercase=True,
    add_prefix_space=True,
)

# a = tokenizer.encode('ï').ids
# print()