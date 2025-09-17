---
layout: post
tags: [nlp,dl,他山之石]
published : true
---
-https://radimrehurek.com/gensim/models/word2vec.html

学习：https://rare-technologies.com/word2vec-tutorial/  （来自gensim文档的官方推荐）
建议直接看原文。
gensim比用numpy普通实现快很多。

Starting from the beginning, gensim’s word2vec expects a sequence of sentences as its input. Each sentence a list of words (utf8 strings):    
```python
# import modules & set up logging
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)
```

>Keeping the input as a Python built-in list is convenient, but can use up a lot of RAM when the input is large.
Gensim only requires that the input must provide sentences sequentially, when iterated over. No need to keep everything in RAM: we can provide one sentence, process it, forget it, load another sentence…

通过迭代器来节省内存.

```python
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = MySentences('/some/directory') # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences)
```

>Say we want to further preprocess the words from the files — convert to unicode, lowercase, remove numbers, extract named entities… All of this can be done inside the MySentences iterator and word2vec doesn’t need to know. All that is required is that the input yields one sentence (list of utf8 words) after another.


>Note to advanced users: calling Word2Vec(sentences, iter=1) will run two passes over the sentences iterator (or, in general iter+1 passes; default iter=5). The first pass collects words and their frequencies to build an internal dictionary tree structure. The second and subsequent passes train the neural model. These two (or, iter+1) passes can also be initiated manually, in case your input stream is non-repeatable (you can only afford one pass), and you’re able to initialize the vocabulary some other way:

```python
model = gensim.models.Word2Vec(iter=1)  # an empty model, no training yet
model.build_vocab(some_sentences)  # can be a non-repeatable, 1-pass generator
model.train(other_sentences)  # can be a non-repeatable, 1-pass generator
```

>Word2vec accepts several parameters that affect both training speed and quality.
One of them is for pruning the internal dictionary. Words that appear only once or twice in a billion-word corpus are probably uninteresting typos and garbage. In addition, there’s not enough data to make any meaningful training on those words, so it’s best to ignore them:   
`model = Word2Vec(sentences, min_count=10)  # default value is 5`

min_count:词在预料中出现的最小次数，低于该次数则忽略
>A reasonable value for min_count is between 0-100, depending on the size of your dataset.
Another parameter is the size of the NN layers, which correspond to the “degrees” of freedom the training algorithm has:

`model = Word2Vec(sentences, size=200)  # default value is 100`  

>Bigger size values require more training data, but can lead to better (more accurate) models. Reasonable values are in the tens to hundreds.

这应该就是embedding layer中的hidden layer的单元数量  

并行度    
`model = Word2Vec(sentences, workers=4) # default = 1 worker = no parallelization`
>The workers parameter has only effect if you have Cython installed. Without Cython, you’ll only be able to use one core because of the GIL (and word2vec training will be miserably slow).

>At its core, word2vec model parameters are stored as matrices (NumPy arrays). Each array is #vocabulary (controlled by min_count parameter) times #size (size parameter) of floats (single precision aka 4 bytes).

>Three such matrices are held in RAM (work is underway to reduce that number to two, or even one). So if your input contains 100,000 unique words, and you asked for layer size=200, the model will require approx. 100,000*200*4*3 bytes = ~229MB.

>There’s a little extra memory needed for storing the vocabulary tree (100,000 words would take a few megabytes), but unless your words are extremely loooong strings, memory footprint will be dominated by the three matrices above.

>Google have released their testing set of about 20,000 syntactic and semantic test examples, following the “A is to B as C is to D” task: https://raw.githubusercontent.com/RaRe-Technologies/gensim/develop/gensim/test/test_data/questions-words.txt.

>Gensim support the same evaluation set, in exactly the same format:

```python
model.accuracy('/tmp/questions-words.txt')
2014-02-01 22:14:28,387 : INFO : family: 88.9% (304/342)
2014-02-01 22:29:24,006 : INFO : gram1-adjective-to-adverb: 32.4% (263/812)
2014-02-01 22:36:26,528 : INFO : gram2-opposite: 50.3% (191/380)
2014-02-01 23:00:52,406 : INFO : gram3-comparative: 91.7% (1222/1332)
2014-02-01 23:13:48,243 : INFO : gram4-superlative: 87.9% (617/702)
2014-02-01 23:29:52,268 : INFO : gram5-present-participle: 79.4% (691/870)
2014-02-01 23:57:04,965 : INFO : gram7-past-tense: 67.1% (995/1482)
2014-02-02 00:15:18,525 : INFO : gram8-plural: 89.6% (889/992)
2014-02-02 00:28:18,140 : INFO : gram9-plural-verbs: 68.7% (482/702)
2014-02-02 00:28:18,140 : INFO : total: 74.3% (5654/7614)
```

>Once again, good performance on this test set doesn’t mean word2vec will work well in your application, or vice versa. It’s always best to evaluate directly on your intended task.

```
model.save('/tmp/mymodel')
new_model = gensim.models.Word2Vec.load('/tmp/mymodel')
```

>which uses pickle internally, optionally mmap‘ing the model’s internal large NumPy matrices into virtual memory directly from disk files, for inter-process memory sharing.

>In addition, you can load models created by the original C tool, both using its text and binary formats:

文本文件或二进制(包括压缩文件)
```
model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)
# using gzipped/bz2 input works too, no need to unzip:
model = Word2Vec.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)
```

在线学习：
```
model = gensim.models.Word2Vec.load('/tmp/mymodel')
model.train(more_sentences)
```

直接从txt读入model，没法继续训练。所以还是用model.save比较好
>You may need to tweak the total_words parameter to train(), depending on what learning rate decay you want to simulate.

>Note that it’s not possible to resume training with models generated by the C tool, load_word2vec_format(). You can still use them for querying/similarity, but information vital for training (the vocab tree) is missing there.

>Word2vec training is an unsupervised task, there’s no good way to objectively evaluate the result. Evaluation depends on your end application.

>Google have released their testing set of about 20,000 syntactic and semantic test examples, following the “A is to B as C is to D” task:
https://raw.githubusercontent.com/RaRe-Technologies/gensim/develop/gensim/test/test_data/questions-words.txt.

>Word2vec supports several word similarity tasks out of the box:
```
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
[('queen', 0.50882536)]
model.doesnt_match("breakfast cereal dinner lunch";.split())
'cereal'
model.similarity('woman', 'man')
0.73723527
```


```
model.accuracy('/tmp/questions-words.txt')
2014-02-01 22:14:28,387 : INFO : family: 88.9% (304/342)
2014-02-01 22:29:24,006 : INFO : gram1-adjective-to-adverb: 32.4% (263/812)
2014-02-01 22:36:26,528 : INFO : gram2-opposite: 50.3% (191/380)
2014-02-01 23:00:52,406 : INFO : gram3-comparative: 91.7% (1222/1332)
2014-02-01 23:13:48,243 : INFO : gram4-superlative: 87.9% (617/702)
2014-02-01 23:29:52,268 : INFO : gram5-present-participle: 79.4% (691/870)
2014-02-01 23:57:04,965 : INFO : gram7-past-tense: 67.1% (995/1482)
2014-02-02 00:15:18,525 : INFO : gram8-plural: 89.6% (889/992)
2014-02-02 00:28:18,140 : INFO : gram9-plural-verbs: 68.7% (482/702)
2014-02-02 00:28:18,140 : INFO : total: 74.3% (5654/7614)

```
>This accuracy takes an optional parameter restrict_vocab which limits which test examples are to be considered.

>Once again, good performance on this test set doesn’t mean word2vec will work well in your application, or vice versa. It’s always best to evaluate directly on your intended task.


>If you need the raw output vectors in your application, you can access these either on a word-by-word basis
```
model['computer']  # raw NumPy vector of a word
array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)
```
>…or en-masse as a 2D NumPy matrix from model.syn0.  (新版本似乎失效了)
返回全部词向量  



>Note that the similarities were trained on a news dataset, and that Google did very little preprocessing there. So the phrases are case sensitive: watch out! Especially with proper nouns.

-https://github.com/RaRe-Technologies/w2v_server_googlenews

网页版word2vec。


>Note that there is a gensim.models.phrases module which lets you automatically detect phrases longer than one word. Using phrases, you can learn a word2vec model where “words” are actually multiword expressions, such as new_york_times or financial_crisis:

```
>>> bigram_transformer = gensim.models.Phrases(sentences)
>>> model = Word2Vec(bigram_transformer[sentences], size=100, ...)
```

英语中的有些词连起来才有意义。原始的输入只是单词的序列。最好要改成词组的序列。
