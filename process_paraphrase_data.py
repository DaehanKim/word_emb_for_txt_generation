'''process labelled paraphrasing dataset for vocab building and word vector training purposes.'''

def main():
	# extract useful sentences from labelled paraphrasing corpus
	sent_list = []
	with open('paraphrasing data_DH - train_x_kss.tsv', 'rt', encoding='utf8') as f:
		for idx, line in enumerate(f):
			if idx < 2 : continue
			sent_list += line.split('\t')[:2]

	# remove empty strings
	while '' in sent_list:
		sent_list.remove('')

	# save results
	new_corpus = '\n'.join(sent_list)+'\n'
	with open('para_corpus','wt',encoding = 'utf8') as f:
		f.write(new_corpus)
	print('saved collected corpus!')


if __name__ == '__main__' : 
	main()

