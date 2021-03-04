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
