from kor2vec import Kor2Vec
import os
import datetime

#config

SAVE_INTERVAL = 10
EPOCH = 100

def load_model():
    # load latest model
    cp_list = [item for item in os.listdir('.') if item.endswith('.ep0')]
    if len(cp_list) ==0 : 
        print('no saved model, Initializing...')
        return Kor2Vec(embed_size=128)
    cp_list.sort()
    cp_path = cp_list[-1]
    print('Loading model {}...'.format(cp_path))
    return Kor2Vec.load(cp_path)


def train_on(model, corpus_path, save_path, do_save_results):
    model.train(corpus_path = corpus_path, model_path = save_path, batch_size = 128, epochs=1)
    # model.save()

def remove_checkpoints():
    # let last 5 checkpoint remain
    cp_list = [item for item in os.listdir('.') if item.endswith('.ep0')]
    cp_list.sort()
    if len(cp_list)>=5:
        to_remove = cp_list[-5:]
    else: 
        to_remove = []
    for rem_ in to_remove:
        os.remove(rem_)

def epoch_train():
    # extract directories
    dirs = ['text/{}'.format(item) for item in os.listdir('text')]
    paths = []
    for dir_ in dirs:   
        paths += ['{}/{}'.format(dir_, item) for item in os.listdir(dir_)]
    paths.sort()

    # init models
    model = load_model()

    for idx,path in enumerate(paths):
        if path.endswith('sampled'): continue
        if not path.split('/')[-1].startswith('sp_tokenized') : continue
        print('current path : {}'.format(path))
        now = datetime.datetime.now()
        save_path = 'kor2vec{:02d}{:02d}{:02d}{:02d}.checkpoint'.format(now.month,now.day, now.hour, now.minute)
        do_save_results= ((idx+1) % SAVE_INTERVAL == 0)
        print('idx : {} / do_save_results : {}'.format(idx, do_save_results))
        try: # pass problematic text file
            train_on(model, path, save_path, do_save_results)
        except:
            pass
        remove_checkpoints()
if __name__ == '__main__':
    for epoch in range(EPOCH):
        epoch_train()
        # break