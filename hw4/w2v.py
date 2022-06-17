import os
from utils import load_training_data
from utils import load_testing_data
from gensim.models import Word2Vec


def train_word2vec(x):
    # 訓練 word to vector 的 word embedding
    model = Word2Vec(x, vector_size=250, window=5, min_count=5, workers=8, epochs=10, sg=1)
    return model


if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = load_training_data('training_label.txt')
    train_x_no_label = load_training_data('training_nolabel.txt')

    print("loading testing data ...")
    test_x = load_testing_data('testing_data.txt')

    model = train_word2vec(train_x + train_x_no_label + test_x)
    # model = train_word2vec(train_x + test_x)

    print("saving model ...")
    model.save(os.path.join('w2v_new.model'))
