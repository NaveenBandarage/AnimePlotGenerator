class config:
    #storing the parameters in a way that makes it easy to access later can just use the config. notation
    #https://trungtran.io/2019/02/08/text-generation-with-pytorch/ - referencing this code
    embedding_dim = 100
    epochs = 15
    hidden_dim = 512
    tokenizer = nltk.word_tokenize
    batch_size = 32
    sequence_len = 30
    model_path = 'lm_lrdecay_drop.bin'

#creating a dataset from which we can use


def create_dataset(synopsis, batch_size, seq_len):
    np.random.seed(0)

    synopsis = synopsis.apply(lambda x: str(x).lower()).values
    synopsis_text = ' '.join(synopsis)

    tokens = config.tokenizer(synopsis_text)
    global num_batches
    num_batches = int(len(tokens)/(seq_len*batch_size))
    tokens = tokens[:num_batches*batch_size*seq_len]

    words = sorted(set(tokens))
    #w2i = word two index and i2w = index to word.
    w2i = {w: i for i, w in enumerate(words)}
    i2w = {i: w for i, w in enumerate(words)}

    tokens = [w2i[tok] for tok in tokens]
    target = np.zeros_like((tokens))
    target[:-1] = tokens[1:]
    target[-1] = tokens[0]

    input_tok = np.reshape(tokens, (batch_size, -1))
    target_tok = np.reshape(target, (batch_size, -1))

    vocab_size = len(i2w)
    return input_tok, target_tok, vocab_size, w2i, i2w


def create_batches(input_tok, target_tok, batch_size, seq_len):

    num_batches = np.prod(input_tok.shape)//(batch_size*seq_len)

    for i in range(0, num_batches*seq_len, seq_len):
        yield input_tok[:, i:i+seq_len], target_tok[:, i:i+seq_len]
