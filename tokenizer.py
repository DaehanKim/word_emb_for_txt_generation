from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
import os
import re
from tqdm import tqdm

# config
PUT_SPECIAL_TOKEN = True

# load model
tok_path = get_tokenizer()
sp = SentencepieceTokenizer(tok_path)
tag = re.compile(r'<[/"=\w\s:?.]{1,200}>')

def test_tag_regex():
	print('Testing tag regex...')
	docs = ['<doc id="5" url="https://ko.wikipedia.org/wiki?curid=5" title="지미 카터">',
	'''</doc>
	<doc id="9" url="https://ko.wikipedia.org/wiki?curid=9" title="수학">''',
	'<doc id="70" url="https://ko.wikipedia.org/wiki?curid=70" title="일반 상대성이론">']

	for doc in docs:
		detect = tag.findall(doc)
		print(detect)
		print(clean_txt(doc))

def test_sentencepiece():
	print('Testing sentencepiece tokenizer...')
	SAMPLE_PATH = 'text/AA/wiki_00'

	# print(sp('한국어 모델을 공유합니다.'))
	new_lines = ""
	with open(SAMPLE_PATH,'rt',encoding='utf8') as f:
		for line in f:
			new_lines += ' '.join(sp(line))

	print(new_lines)

def clean_txt(txt):
	tag_list = tag.findall(txt)
	for tag_ in tag_list:
		txt = txt.replace(tag_, '')
	return txt

def sent_tokenize(txt): 
	# put <sos> and <eos> at each sentences
	# ◆: sos ◇: eos
	splitter = '.'
	splitted = [e+splitter for e in txt.split(splitter)]
	if splitted[-1] == '.': splitted = splitted[:-1]
	splitted = ['◆ ' + item +' ◇' for item in splitted]
	return '\n'.join(splitted)

def test_sent_tokenize(txt):
	a = sent_tokenize('안녕하세요. 반가워요. 잘있어요. 다시 만나요.')
	print(a)

def transform(path):
	new_lines = ""
	with open(path,'rt',encoding='utf8') as f:
		for line in f:
			if line.strip() == '': continue # if it's just a new line or tab or spaces, pass by.
			new_lines += (' '.join(sp(sent_tokenize(clean_txt(line))))+' ')

	new_lines = new_lines.replace('◆', '<sos>').replace('◇','<eos>')
	with open('{}/{}/sp_tokenized.{}'.format(path.split('/')[0],path.split('/')[1],path.split('/')[-1]),'wt', encoding='utf8') as f:
		f.write(new_lines) 

def transform_all():
	# extract directories
	dirs = ['text/{}'.format(item) for item in os.listdir('text')]
	paths = []
	for dir_ in dirs:	
		paths += ['{}/{}'.format(dir_, item) for item in os.listdir(dir_)]

	# transform files
	for path in tqdm(paths, desc='cleaning/tokenizing docs'):
		if path.split('/')[-1].startswith('sp_tokenized') or path.split('/')[-1].endswith('sampled'): 
			print('pass {}'.format(path))
			continue
		# print(path.split('/')[-1])
		transform(path)
		# break


if __name__ == '__main__':
	# transform_all()
# test_tag_regex()
	transform('text/AG/para_corpus')
	