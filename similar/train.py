import model
from dataset import Dataset
import argparse
import tensorflow as tf


def main():
    args = parse_args()

    word_dict = tf.contrib.lookup.index_table_from_file(
        args.vocab_fname,
        num_oov_buckets=2,
        key_column_index=0,
        delimiter='\t'
    )

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())

        train_data = Dataset(sess, args.train_fname, word_dict, repeat_count=5, shuffle=True)
        dev_data = Dataset(sess, args.dev_fname, word_dict)
        test_data = Dataset(sess, args.test_fname, word_dict)

        #print(collections.Counter(itertools.chain(*map(lambda d: np.argmax(d[2], axis=1), dev_data))))
        #print(collections.Counter(itertools.chain(*map(lambda d: np.argmax(d[2], axis=1), test_data))))
        #print(collections.Counter(itertools.chain(*map(lambda d: np.argmax(d[2], axis=1), train_data))))
        model_config = {
            "vocab_size": sess.run(word_dict.size()),
            "embedding_dim": 100,
            "lstm_dim": 100,
            "label_num": 2,
            "learning_rate": 3e-4,
        }

        m = model.SiameseLSTM(sess, model_config).initialize()
        m.train(train_data, dev_data)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_fname", default="./data/train_test/vocab.txt",
                                         help="Vocabulary file name")

    parser.add_argument("--train_fname", default="./data/train_test/train.csv",
                                         help="Training data file name")

    parser.add_argument("--dev_fname", default="./data/train_test/dev.csv",
                                       help="Development data file name")

    parser.add_argument("--test_fname", default="./data/train_test/test.csv",
                                        help="Testing data file name")
    return parser.parse_args()


if __name__ == "__main__":
    exit(main())

