# word_emb_for_txt_generation
Preprocessing corpus and train word embeddings for text generation


### File description


 - ```tokenizer.py```
 
   - clean xml tags from corpus 
   - apply sentencepiece tokenizer (trained by KoBERT team)
   - add <sos> and <eos> to each sentence
   - input : wiki corpus / output : sentence piece tokenized corpus(ready to use for kor2vec training) 
   
 - ```process_paraphrase_data.py```
 
   - make corpus for word vector training using hand labelled paraphrasing data(from .tsv format)
   - input : paraphrasing hand-labelled data of .tsv format / output : sentences collected for kor2vec training
 
 - ```vocab_build.py```
 
   - scan wiki corpus and paraphrasing corpus to build vocabulary table, which consists of words appearing more than 5 times across the whole corpus
   - input : sentence piece tokenized corpus / output : txt file with each vocab in each line
   
 - ```build_embedding.py```
 
   - make embeddings of each word based on kor2vec model
   - input : kor2vec checkpoint, vocab list / output : txt file with embedding vector seperated with spaces in each line
   
 - ```similarity.py```
 
   - interactively shows which vectors are closest to the given vector 
   - input : word embedding file / output : on the console
   
 - ```main.py```
 
   - trains kor2vec model 
   - input : sentence piece tokenized corpus / output : kor2vec checkpoint
   
