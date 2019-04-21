#coding:utf-8
'''
author:zrant
'''

#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from deepwalk import graph
from deepwalk import walks as serialized_walks
from gensim.models import Word2Vec
from deepwalk.skipgram import Skipgram

def main():
    parser = ArgumentParser("deepwalk",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                        help="drop a debugger if an exception is raised.")

    parser.add_argument('--format', default='adjlist',
                        help='File format of input file')
    """
    由于本程序只处理三种类型的输入（ 'adjlist','edgelist','mat'），因此这里也可以写成：
    parser.add_argument('--format', default='adjlist',choices=['adjlist','edgelist','mat'],
                        help='File format of input file')
    """

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')
    """
    这句话的意思是，你的命令行里必须有--input,至于说这个参数后面有没有文件名字无所谓，如果有文件路径，那么input就是
    这个文件；如果没有指定文件路径，那么input=None
    """

    parser.add_argument("-l", "--log", dest="log", default="INFO",
                        help="log verbosity level")

    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')
    """
    这是针对matlab文件.mat设置的，是这个文件内置的邻接矩阵名字。如果我们使用的是karate.adjlist形式的格式，这个地方就可忽略掉
    """

    parser.add_argument('--max-memory-data-size', default=10000, type=int,
                        help='Size to start dumping walks to disk, instead of keeping them in memory.')
    """
    max_memory_data_size在这里不是实际的内存大小，它只是节点使用总次数的计数。
    """




    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')
    """
    number-walks：相当于你算法里面的$\gamma$
    """

    parser.add_argument('--output', required=True,
                        help='Output representation file')

    parser.add_argument('--representation-size', default=64, type=int,
                        help='Number of latent dimensions to learn for each node.')

    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for random walk generator.')

    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')

    parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                        help='Use vertex degree to estimate the frequency of nodes '
                             'in the random walks. This option is faster than '
                             'calculating the vocabulary.')


    parser.add_argument('--walk-length', default=40, type=int,
                        help='Length of the random walk started at each node')
    """
    walk_length：相当于你算法里面的$\ell$
    """

    parser.add_argument('--window-size', default=5, type=int,
                        help='Window size of skipgram model.')
    """
      window_size：相当于你算法里面的context size $\omega$
    """

    parser.add_argument('--workers', default=1, type=int,
                        help='Number of parallel processes.')

    # args = parser.parse_args()
    args = parser.parse_args("--input ../example_graphs/karate.adjlist "
                             "--output ./output".split())
    print(args)

    # numeric_level = getattr(logging, args.log.upper(), None)
    # logging.basicConfig(format=LOGFORMAT)
    # logger.setLevel(numeric_level)
    #
    # if args.debug:
    #     sys.excepthook = debug

    process(args)

def process(args):
    """Section1：
      deepwalk的输入可以是三种格式的数据，因此针对这三种格式的数据分别构建图类对象G
      我们这里主要关心的是karae.adjlist和p2p-Gnutella08.edgelist两个文件对应的格式，
      这两个文件都在文件夹example_graphs里面
      对于.mat格式我们不关心，也不使用
    """
    if args.format == "adjlist":
        G = graph.load_adjacencylist(args.input, undirected=args.undirected)
    elif args.format == "edgelist":
        G = graph.load_edgelist(args.input, undirected=args.undirected)
    elif args.format == "mat":
        G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)



    """Section2：
    下面打印的都是一些基本信息
    """
    print("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * args.number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * args.walk_length

    print("Data size (walks*length): {}".format(data_size))



    """Section3：
      这里的data_size不是实际的内存大小，只是节点使用总次数的计数。
      在这个if-else代码段中，更常出现的情况是if，而else里的代码几乎从来不会出现，除非data_size超过了max_memory_data_size
      max_memory_data_size默认的是10亿次。
    """
    if data_size < args.max_memory_data_size:
        print("Walking...")
        walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                            path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
        """
        这里直接调用图对象里的build_deepwalk_corpus，它的主要作用是，对每一个点进行多次随机游走，最后的到全部的walks。
        相当于deepwalk论文里两层for循环randomwalk，
        也相当于node2vec里两层for循环node2vecWalk，
        也相当于你算法里的两层for循环下的adawalk
        num_paths=args.number_walks：是采样的轮数，即每一个起点采样的次数，相当于你论文里的$\gamma$
        path_length=args.walk_length, 
        alpha=0, 指的是以概率1-alpha从当前节点继续走下去(可以回到上一个节点)，或者以alpha的概率停止继续走，回到路径的起点重新走。
                 请注意：这里的随机游走路径未必是连续的，有可能是走着走着突然回到起点接着走
                 即便alpha=0，也不能保证以后的路径不会出现起点，因为有可能从起点开始走到第二个点时，这第二个点以1的概率继续走下去，
                 这时候它可能会回到起点。
                 但是如果alpha=1,那么就意味着它不可能继续走下去，只能每次都从起点开始重新走，所以这个时候打印出来的路径序列是一连串的起点。
        rand=random.Random(args.seed)
        """


        print("Training...")

        model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1,
                         workers=args.workers)
        """
        这段代码几乎不需要改动，它就是你算法里的SkipGram，你算发里有三个输入：
        random walks---->walks
        context size---->window=args.window_size
        embedding size---->size=args.representation_size
        如果你想对这个模型了解的更多，或者是做更加人性化的定制，需要继续了解from gensim.models import Word2Vec下的源码。
        我们会在下一篇博文中专门解读这个源码
        Word2Vec：

        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
            If you don't supply `sentences`, the model is left uninitialized -- use if you plan to initialize it
            in some other way.

        sg : int {1, 0}
            Defines the training algorithm. If 1, skip-gram is employed; otherwise, CBOW is used.
        size : int
            Dimensionality of the feature vectors.
        window : int
            The maximum distance between the current and predicted word within a sentence.
        alpha : float
            The initial learning rate.
        min_alpha : float
            Learning rate will linearly drop to `min_alpha` as training progresses.
        seed : int
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        min_count : int
            Ignores all words with total frequency lower than this.
        max_vocab_size : int
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        workers : int
            Use these many worker threads to train the model (=faster training with multicore machines).
        hs : int {1,0}
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        negative : int
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        cbow_mean : int {1,0}
            If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        hashfxn : function
            Hash function to use to randomly initialize weights, for increased training reproducibility.
        iter : int
            Number of iterations (epochs) over the corpus.
        trim_rule : function
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
            of the model.
        sorted_vocab : int {1,0}
            If 1, sort the vocabulary by descending frequency before assigning word indexes.
        batch_words : int
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
        compute_loss: bool
            If True, computes and stores loss value which can be retrieved using `model.get_latest_training_loss()`.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`
            List of callbacks that need to be executed/run at specific stages during training.

        Examples
        --------
        Initialize and train a `Word2Vec` model

        >>> from gensim.models import Word2Vec
        >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>>
        >>> model = Word2Vec(sentences, min_count=1)
        >>> say_vector = model['say']  # get vector for word
        """


    else:
        print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size,
                                                                                                             args.max_memory_data_size))
        print("Walking...")

        """
        在这种情况下，游走路径会被存入到一系列文件output.walks.x中，即如果你在命令行指定了--output file_path
        那么程序结束后会出现两个文件，一个是file_path，一个是file_path.walks.0,file_path.walks.1,...file_path.walks.n。
        file_path存的是各个节点的embedding，
        output.walks.x存的是采样的路径,x表示这个文件是第x个处理器存入的
        x的个数与处理器个数相同
        """
        walks_filebase = args.output + ".walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                                          path_length=args.walk_length, alpha=0,
                                                          rand=random.Random(args.seed),
                                                          num_workers=args.workers)
        """
        在目前的层次上，你只需要知道serialized_walks.write_walks_to_disk本质上也是在调用graph里的randwalk，只不过包装一下
        加入了并行化代码和写到磁盘的程序。
        这个函数具体的细节需要稍后剖析另一个文件walk.py
        函数里各个参数的含义可以参考if语句里面的代码
        """



        # print("Counting vertex frequency...")
        # if not args.vertex_freq_degree:
        #     """Use vertex degree to estimate the frequency of nodes
        #     in the random walks. This option is faster than
        #     calculating the vocabulary."""
        #     vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
        # else:
        #     # use degree distribution for frequency in tree
        #     vertex_counts = G.degree(nodes=G.iterkeys())
        #
        # print("Training...")
        # walks_corpus = serialized_walks.WalksCorpus(walk_files)
        # model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
        #                  size=args.representation_size,
        #                  window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)
        """
        上面注释掉的代码是原来的代码，这里没有使用Word2Vec而是使用了自己编写的Skipgram，
        但是Skipgram仅仅是对Word2Vec的输入参数做了一些保装，里面仍然在调用word2vec，
        word2vec本身也可以实现并行化运算，不知道这里为什么要这么做。
        需要说明的是：从现有的情况类看vertex_counts似乎并没什么卵用
        直接向下面这样写就可以：         
        """
        print("Training...")
        walks_corpus = serialized_walks.WalksCorpus(walk_files)
        model = Word2Vec(sentences=walks_corpus,
                         size=args.representation_size,
                         window=args.window_size,
                         min_count=0, sg=1, hs=1,
                         workers=args.workers)


    """Section？：
      训练完成之后，将模型输出
    """
    model.wv.save_word2vec_format(args.output)


#----------------------------代码简化------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#!usr/bin/env python
# -*- coding:utf-8 _*-
""" 
@project:deepwalk-master
@author:xiangguosun 
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: simple_main.py
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1 
@time: 2018/09/13 
"""

import sys
import random

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from deepwalk import graph
from deepwalk import walks as serialized_walks
from gensim.models import Word2Vec


def process(args):
    if args.format == "adjlist":
        G = graph.load_adjacencylist(args.input, undirected=args.undirected)
    elif args.format == "edgelist":
        G = graph.load_edgelist(args.input, undirected=args.undirected)
    elif args.format == "mat":
        G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

    print("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * args.number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * args.walk_length

    print("Data size (walks*length): {}".format(data_size))

    print("Walking...")
    walks_filebase = args.output + ".walks"
    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                                      path_length=args.walk_length, alpha=0,
                                                      rand=random.Random(args.seed),
                                                      num_workers=args.workers)
    walks = serialized_walks.WalksCorpus(walk_files)

    print("Training...")
    model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1,
                     workers=args.workers)

    model.wv.save_word2vec_format(args.output)


def main():
    parser = ArgumentParser("deepwalk",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                        help="drop a debugger if an exception is raised.")

    parser.add_argument('--format', default='adjlist', choices=['adjlist', 'edgelist', 'mat'],
                        help='File format of input file')

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')

    parser.add_argument("-l", "--log", dest="log", default="INFO",
                        help="log verbosity level")

    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')

    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')

    parser.add_argument('--output', required=True,
                        help='Output representation file')

    parser.add_argument('--representation-size', default=64, type=int,
                        help='Number of latent dimensions to learn for each node.')

    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for random walk generator.')

    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')

    parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                        help='Use vertex degree to estimate the frequency of nodes '
                             'in the random walks. This option is faster than '
                             'calculating the vocabulary.')

    parser.add_argument('--walk-length', default=40, type=int,
                        help='Length of the random walk started at each node')

    parser.add_argument('--window-size', default=5, type=int,
                        help='Window size of skipgram model.')

    parser.add_argument('--workers', default=1, type=int,
                        help='Number of parallel processes.')

    args = parser.parse_args("--input ../example_graphs/karate.adjlist "
                             "--output ./output".split())
    print(args)

    process(args)


if __name__ == "__main__":
    sys.exit(main())