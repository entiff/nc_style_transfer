import sentencepiece as spm


def make_vocab_file():
    spm.SentencePieceTrainer.train(input='./colloquial_literary.txt', model_prefix='spm',
                                   vocab_size=50000)


if __name__ == '__main__':
    make_vocab_file()
