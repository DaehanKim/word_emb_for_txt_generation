import numpy as np 
from tqdm import tqdm
import datetime

def get_number_of_words(path):
    num_line = 0
    with open(path,'rt',encoding='utf8') as f:
        for line in f:
            num_line += 1
    return num_line

# config
FILE_NAME = 'word_list_03020941.txt'
NUM_WORD = get_number_of_words(FILE_NAME)
DIM = 128

def build_random_embedding():
    emb_lst = []
    for i in tqdm(range(NUM_WORD), total=NUM_WORD, desc='build random embedding'):
        emb_lst.append(' '.join(['{:.5f}'.format(item) for item in np.random.randn(128)]))

    emb_txt = '\n'.join(emb_lst) + '\n'
    now = datetime.datetime.now()
    with open('word_emb_{:02d}{:02d}{:02d}{:02d}.txt'.format(now.month,now.day,now.hour,now.minute),'wt',encoding='utf8') as f:
        f.write(emb_txt)
    print('result saved!')

def build_using_kor2vec(model_path):
    from kor2vec import Kor2Vec
    import torch.nn as nn

    # load kor2vec model
    lm = Kor2Vec.load(model_path)
    emb_lst = []
    with open(FILE_NAME,'rt',encoding='utf8') as f:
        for line in tqdm(f,total=NUM_WORD, desc = 'build using kor2vec'):
            emb_lst.append(lm.embedding(line.strip()).detach().squeeze(1).numpy())
    
    # save results
    now = datetime.datetime.now()
    np.savetxt('word_emb_{:02d}{:02d}{:02d}{:02d}.txt'.format(now.month,now.day,now.hour,now.minute),
        np.array(emb_lst).squeeze(1), 
        fmt="%.5f")


if __name__ == '__main__':
    model_path = 'kor2vec03010231.checkpoint.ep0'
    build_using_kor2vec(model_path)