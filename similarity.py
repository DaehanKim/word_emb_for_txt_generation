import numpy as np
from scipy.spatial.distance import cosine
import os

def get_latest_emb_and_word_list():
	emb_list, word_list = [], []
	for item in os.listdir('.'):
		if item.startswith('word_emb'):
			emb_list.append(item)
		elif item.startswith('word_list'):
			word_list.append(item)
	emb_list.sort();word_list.sort()
	return emb_list[-1], word_list[-1]

EMBEDDING_FILE, WORD_FILE = get_latest_emb_and_word_list()
K = 10

def cosine_sim(vec1, vec2):
	return 1-cosine(vec1, vec2)

def print_similar_words(words, similarity):
	for word, sim in zip(words, similarity):
		print('{} : {}'.format(word, sim))

# Load an embedding
print('Loading embedding...')
emb = np.genfromtxt(EMBEDDING_FILE)
with open(WORD_FILE,'rt',encoding='utf8') as f:
	words = [line.strip() for line in f.readlines()]

# User interactive query
input_ = ''
while input_ != 'quit':
	input_ = input('Query : ')
	if input_ in words:
		idx = words.index(input_)
		input_emb = emb[idx]
		sim_list = np.array([cosine_sim(input_emb, other_vec) for other_vec in emb])
		topk_idx = np.argsort(-sim_list)[:K]
		print_similar_words(np.array(words)[topk_idx], sim_list[topk_idx])
	else: 
		print('Not in a vocab!')