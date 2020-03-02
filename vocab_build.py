from collections import Counter
from tqdm import tqdm
import os
import datetime

# config
N = 5
corpus_contains_special_tokens = True

def extract_vocab(path):
	with open(path, 'rt',encoding='utf8') as f:
		tokens = f.read().split()
	return Counter(tokens)

def main():
	# extract path
	dirs = ['text/{}'.format(item) for item in os.listdir('text')]
	paths = []
	for dir_ in dirs:	
		paths += ['{}/{}'.format(dir_, item) for item in os.listdir(dir_)]

	# print(paths)
	# exit()

	# extract vocabs
	whole_vocab_count = Counter([])
	for path in tqdm(paths):
		if not path.split('/')[-1].startswith('sp_tokenized') or path.endswith('sampled'): continue # verify path
		vocab_count = extract_vocab(path) # returns a Counter object
		whole_vocab_count += vocab_count
		# break

	# print(whole_vocab_count)
	# collect all words that appear more than or equal to N times
	final_vocab_list = []
	for k,v in whole_vocab_count.items():
		if v >= N: final_vocab_list.append(k)

	# add special tokens
	if corpus_contains_special_tokens:
		final_vocab_list.remove('<sos>'); final_vocab_list.remove('<eos>')
	final_vocab_list = ['<sos>','<eos>'] + final_vocab_list


	# print(final_vocab_list)
	# save result
	now = datetime.datetime.now()
	with open('word_list_{:02d}{:02d}{:02d}{:02d}.txt'.format(now.month, now.day, now.hour, now.minute),'wt', encoding='utf8') as f:
		f.write('\n'.join(final_vocab_list)+'\n')
	print('result saved!')

if __name__ == '__main__':
	main()