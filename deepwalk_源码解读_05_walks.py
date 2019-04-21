#coding:utf-8
'''
author:zrant
'''


#!usr/bin/env python
# -*- coding:utf-8 _*-

# import logging
from io import open
from os import path
# from time import time
from multiprocessing import cpu_count
import random
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

from six.moves import zip

from deepwalk.myanalysis import graph

# logger = logging.getLogger("deepwalk")

__current_graph = None

# speed up the string encoding 然而这个参数在代码中并没有再次出现过
# __vertex2str = None

"""
count_words(file)和
count_textfiles(files, workers=1)的作用是
计算语料库中各个词的词频
"""


def count_words(file):
    """
    Counts the word frequences in a list of sentences.
    file里面是一系列walks，一个walk是一行，这个函数最终返回这个file中各个单词出现的次数。

    Note:
      This is a helper function for parallel execution of `Vocabulary.from_text`
      method.
    exmaple:

    file='./output.walks.0'
    print(count_words(file))

    输出:
    Counter({'34': 1446, '1': 1414, '33': 920, '3': 914, '2': 752, '4': 537,
            '32': 520, '9': 443, '14': 432, '24': 420, '6': 408, '7': 393,
            '28': 377, '31': 355, '8': 346, '30': 320, '25': 295, '11': 292,
            '5': 284, '29': 281, '20': 271, '26': 265, '17': 196, '10': 190,
            '18': 181, '15': 173, '21': 168, '13': 167, '19': 159, '22': 155,
            '27': 153, '16': 152, '23': 134, '12': 87})

  """
    c = Counter()
    with open(file, 'r') as f:
        for l in f:
            words = l.strip().split()
            c.update(words)
    return c


def count_textfiles(files, workers=1):
    """
    这个函数使用里多线程的技巧，我们首先需要了解ProcessPoolExecutor方法

    c = Counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for c_ in executor.map(count_words, files):
            c.update(c_)
    return c

    这段代码是标准使用模式，意思是：
    开启workers个进程处理器，将他们整体命名为executor，或者说executor是
    一个ProcessPoolExecutor实例化的对象的别名。


    executor.map(called_function, input_array)
    这句代码的意思是并行化的完成下面的功能:
    result=[]
    for item in input_array:
        re=called_function(item)
        result.append(re)
    不同的是，这段代码不能并行计算，而executor.map(called_function, input_array)
    可以实现并行计算，并且显然在代码上更加简洁


    count_words这个函数会返回一个Counter，因此
    c = Counter()
    for c_ in executor.map(count_words, files):
        c.update(c_)
    这段代码的意思是，每次得到一个Counter(即代码中的c_)，就用这个Counter去更新
    全局的Counter(即代码中的c)


    关于from collections import Counter的具体或更多用法：

    Counter是继承自Dict的一个子类，因此它本质上仍然是一个字典类型
    Dict subclass for counting hashable items.  Sometimes called a bag
    or multiset.  Elements are stored as dictionary keys and their counts
    are stored as dictionary values.

    ..>>> c = Counter('abcdeabcdabcaba')  # count elements from a string

    ..>>> c.most_common(3)                # three most common elements
    [('a', 5), ('b', 4), ('c', 3)]
    ..>>> sorted(c)                       # list all unique elements
    ['a', 'b', 'c', 'd', 'e']
    ..>>> ''.join(sorted(c.elements()))   # list elements with repetitions
    'aaaaabbbbcccdde'
    .>>> sum(c.values())                 # total of all counts
    15

    .>>> c['a']                          # count of letter 'a'
    5
    .>>> for elem in 'shazam':           # update counts from an iterable
    ...     c[elem] += 1                # by adding 1 to each element's count
    .>>> c['a']                          # now there are seven 'a'
    7
    .>>> del c['b']                      # remove all 'b'
    .>>> c['b']                          # now there are zero 'b'
    0

    .>>> d = Counter('simsalabim')       # make another counter
    .>>> c.update(d)                     # add in the second counter
    .>>> c['a']                          # now there are nine 'a'
    9

    .>>> c.clear()                       # empty the counter
    .>>> c
    Counter()
    Note:  If a count is set to zero or reduced to zero, it will remain
    in the counter until the entry is deleted or the counter is cleared:
    .>>> c = Counter('aaabbc')
    .>>> c['b'] -= 2                     # reduce the count of 'b' by two
    .>>> c.most_common()                 # 'b' is still in, but its count is zero
    [('a', 3), ('c', 1), ('b', 0)]

    """
    c = Counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for c_ in executor.map(count_words, files):
            c.update(c_)
    return c


def count_lines(f):
    if path.isfile(f):  # Test whether a path is a regular file
        num_lines = sum(1 for line in open(f))
        """
        上面这行代码相当于：
            a=[1 for line in open(f)] # a=[1,1,1,1,1,...,1]
            num_lines = sum(a)
        """
        return num_lines
    else:
        return 0


def _write_walks_to_disk(args):
    """
    :param args: ppw, path_length, alpha, random.Random(rand.randint(0, 2 ** 31)), file_
    ppw: paths number for a single worker
    path_length: 随机游走的路径长度
    alpha: 以alpha的概率停止继续走，回到路径的起点重新走
    random.Random(rand.randint(0, 2 ** 31)): 随机数种子，这个种子是在rand种子基础上建立的种子
    file_: './output.walks.0' 随机游走路径存储的文件
    :return:
    """
    num_paths, path_length, alpha, rand, f = args
    G = __current_graph
    # t_0 = time()
    with open(f, 'w') as fout:
        for walk in graph.build_deepwalk_corpus_iter(G=G, num_paths=num_paths, path_length=path_length,
                                                     alpha=alpha, rand=rand):
            fout.write(u"{}\n".format(u" ".join(v for v in walk)))
    # logger.debug("Generated new file {}, it took {} seconds".format(f, time() - t_0))
    return f


def write_walks_to_disk(G, filebase, num_paths, path_length, alpha=0, rand=random.Random(0), num_workers=cpu_count(),
                        always_rebuild=False):
    """
       :param G:
       :param filebase: './output.walks'（实际的文件是'./output.walks.x'）
       :param num_paths: 每一个起点走几轮
       :param path_length: 随机游走路径长度
       :param alpha:alpha=0, 指的是以概率1-alpha从当前节点继续走下去(可以回到上一个节点)，或者以alpha的概率停止继续走，回到路径的起点重新走。
                    请注意：这里的随机游走路径未必是连续的，有可能是走着走着突然回到起点接着走
                    即便alpha=0，也不能保证以后的路径不会出现起点，因为有可能从起点开始走到第二个点时，这第二个点以1的概率继续走下去，
                    这时候它可能会回到起点。
                    但是如果alpha=1,那么就意味着它不可能继续走下去，只能每次都从起点开始重新走，所以这个时候打印出来的路径序列是一连串的起点。
       :param rand: 随机数种子，确保随机过程可以复现
       :param num_workers: 进程处理器个数
       :param always_rebuild:
       :return:
       """
    global __current_graph
    """
    关于global变量的使用，请参考博客
    [python基础 - global关键字及全局变量的用法](https://blog.csdn.net/diaoxuesong/article/details/42552943)
    """

    __current_graph = G
    # files_list = ["{}.{}".format(filebase, str(x)) for x in list(range(num_paths))]
    """从代码分析来看，上面这句是写错了，应该这样写:"""
    files_list = ["{}.{}".format(filebase, str(x)) for x in list(range(num_workers))]

    print("num_paths is: ", num_paths)
    print([str(x) for x in list(range(num_paths))])
    print("file_lists:", files_list)
    """
    './output.walks.0'
    './output.walks.1'
    './output.walks.2'
    ...
    """
    expected_size = len(G)  # nodes number
    args_list = []  # ?
    files = []  # files_list只是一系列的文件名，files里才是真正存入采样路径后的文件

    if num_paths <= num_workers:
        paths_per_worker = [1 for x in range(num_paths)]
        """
        如果采样轮数小于处理器个数，
        那么其中一些处理器中，每一个处理器负责其中一轮采样，
        剩下一些处理器则没有任务
        """
    else:
        paths_per_worker = [len(list(filter(lambda z: z != None, [y for y in x])))
                            for x in graph.grouper(int(num_paths / num_workers) + 1, range(1, num_paths + 1))]
        """
        grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')
        第一个参数是整数int，第二个参数是可迭代对象，例如range(1,7+1) （返回1，2，3，4，5，6，7的迭代对象）
        如果第三个参数不写，则default为None,这样
        grouper(3, range(1,7+1))-->(1,2,3),(4,5,6),(7,None,None)

        假如现在有7轮，有3个workers那么每个worker至少需要处理7/3=2.3333...轮，为了保证所有轮数都被处理
        向上取整得3轮.即int(7/3)+1，

        轮数序列号为：1,2,3,4,5,6,7，所以每3个轮数被分成一组,得到：
        (1,2,3)(4,5,6)(7,None,None)

        filter(function, seq)可以从seq中筛选出是的function为True的元素
        filter(lambda z: z != None, [y for y in x])
        意思就是说，对于刚才得到的(1,2,3)(4,5,6)(7,None,None)中的任何一个进行筛选,例如
        若x=(7,None,None)则
        seq=[y for y in x]=[7,None,None]
        function相当于:
            fun(z):
                if z!=None:
                    return True
        这里的z是[7,None,None]中的每一个元素。

        这样通过filter之后，得到的结果为：
        [7]
        再将其进行len(list([7]))得到1，即这一个worker处理的轮数为1轮



        存在的问题：如果有6轮，3个处理器，那么按照它的方法，分配的结果是

        处理器1：1，2，3
        处理器2：4，5，6
        处理器3：None      
        """
    """
    其实上面的if-else完全可以合并：
    paths_per_worker = [len(list(filter(lambda z: z != None, [y for y in x])))
                        for x in graph.grouper(int(num_paths / num_workers) 
                                               if num_paths % num_workers == 0 
                                               else int(num_paths / num_workers) + 1, 
                                               range(1, num_paths + 1))]

    """

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for size, file_, ppw in zip(executor.map(count_lines, files_list), files_list, paths_per_worker):

            """
            假如有7轮，3个worker，按照原始代码，path_per_worker=[3,3,1]
            size    file_           ppw
            0   './output.walks.0'  3
            0   './output.walks.1'  3
            0   './output.walks.2'  1
            """
            if always_rebuild or size != (ppw * expected_size):
                args_list.append((ppw, path_length, alpha, random.Random(rand.randint(0, 2 ** 31)), file_))
                """
                random.Random(rand.randint(0, 2 ** 31))的意思是：

                首先rand在这里是外面传进来的随机数种子：rand=random.Random(0)
                然后用种子生成随机数，随机数大小介于0-2^31之间,即:random.Random(0).randint(0, 2 ** 31)
                得到的这个随机数仍然是一个整数，把这个整数作为新的种子,即:
                random.Random(random.Random(0).randint(0, 2 ** 31))
                """
            else:
                files.append(file_)
            """
            这里的if-else语句的意思是：
            假如我们现在有20个节点，7轮,3个处理器(各个处理器处理的轮数应为：3,3,1)，那么最终要生成三个文件应为：
            './output.walks.0'  3*20=60行
            './output.walks.1'  3*20=60行
            './output.walks.2'  1*20=20行

            从中可以看出ppw * expected_size表示当前文件理应存入获取的行数


            假如我们因为某些原因，已经生成好了'./output.walks.0'，但是半道停了下来，再次开始时，我们希望可以接着做，而不是从新生成
            './output.walks.0'文件，即我们希望always_rebuild=False
            那么这个时候我们就要判断一下已有的'./output.walks.0'文件是不是完成了，
            如若还没有完成，即size != (ppw * expected_size)，那么这个文件就重新开始。
            如果已经写完了，那么直接把这个文件添加到files里面就可以了。即files.append(file_)

            args_list实际生成了多少个参数组，就意味着我们有多少个文件需要完成任务，每一个参数组对应一个处理器因此后面的代码写的是：
            for file_ in executor.map(_write_walks_to_disk, args_list):
                files.append(file_)
            """


    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file_ in executor.map(_write_walks_to_disk, args_list):
            files.append(file_)

    return files


class WalksCorpus(object):
    def __init__(self, file_list):
        """

        :param file_list: 这是write_walks_to_disk写在本地里的一组文件列表
        """
        self.file_list = file_list

    def __iter__(self):
        """
        Python 中的顺序类型，都是可迭代的（list, tuple, string）。
        其余包括 dict, set, file 也是可迭代的。
        对于用户自己实现的类型，如果提供了 __iter__() 或者 __getitem__() 方法，
        那么该类的对象也是可迭代的。


        假如file_list=['output.walks.0','output.walks.1']其中
        'output.walks.0':
        8 2 4 8 2 31 34 28 24 28 25 28 3 10 3
        2 1 22 1 18 2 8 2 4 2 18 1 8 3 2 1
        'output.walks.1':
         32 25 26 32 29 3 33 31 9 33 16 34
         6 11 1 20 34 30 24 30 24 28 3 1 14

        那么这个函数返回后得到的就是：
        [   [8, 2, 4, 8, 2, 31, 34, 28, 24, 28, 25, 28, 3, 10, 3],
            [2, 1, 22, 1, 18, 2, 8, 2, 4, 2, 18, 1, 8, 3, 2, 1],
            [32, 25, 26, 32, 29, 3, 33, 31, 9, 33, 16, 34 ],
            [6, 11, 1, 20, 34, 30, 24, 30, 24, 28, 3, 1, 14]
        ]


        更多关于迭代和yield的内容，可以参考博文
        [Python 中的黑暗角落（一）：理解 yield 关键字](https://liam0205.me/2017/06/30/understanding-yield-in-python/)
        :return:
        """
        for file in self.file_list:
            with open(file, 'r') as f:
                for line in f:
                    yield line.split()

"""
下面这段代码是作者临时写的，忘记删除了，实际上根本没有第二次出现过
"""
def combine_files_iter(file_list):
    for file in file_list:
        with open(file, 'r') as f:
            for line in f:
                yield line.split()



#------------------------代码简化--------------------------------
#!usr/bin/env python
# -*- coding:utf-8 _*-
""" 
@project:deepwalk-master
@author:xiangguosun 
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: simple_walks.py 
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1 
@time: 2018/09/14 
"""

from io import open
from os import path
from multiprocessing import cpu_count
import random
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

from six.moves import zip

from deepwalk.myanalysis import graph

__current_graph = None


def count_words(file):
    c = Counter()
    with open(file, 'r') as f:
        for l in f:
            words = l.strip().split()
            c.update(words)
    return c


def count_textfiles(files, workers=1):
    c = Counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for c_ in executor.map(count_words, files):
            c.update(c_)
    return c


def count_lines(f):
    if path.isfile(f):  # Test whether a path is a regular file
        num_lines = sum(1 for line in open(f))
        return num_lines
    else:
        return 0


def _write_walks_to_disk(args):
    num_paths, path_length, alpha, rand, f = args
    G = __current_graph
    with open(f, 'w') as fout:
        for walk in graph.build_deepwalk_corpus_iter(G=G, num_paths=num_paths, path_length=path_length,
                                                     alpha=alpha, rand=rand):
            fout.write(u"{}\n".format(u" ".join(v for v in walk)))
    return f


def write_walks_to_disk(G, filebase, num_paths, path_length, alpha=0, rand=random.Random(0), num_workers=cpu_count(),
                        always_rebuild=False):
    global __current_graph

    __current_graph = G

    files_list = ["{}.{}".format(filebase, str(x)) for x in list(range(num_workers))]

    expected_size = len(G)
    args_list = []
    files = []

    paths_per_worker = [len(list(filter(lambda z: z != None, [y for y in x])))
                        for x in graph.grouper(int(num_paths / num_workers)
                                               if num_paths % num_workers == 0
                                               else int(num_paths / num_workers) + 1,
                                               range(1, num_paths + 1))]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for size, file_, ppw in zip(executor.map(count_lines, files_list), files_list, paths_per_worker):
            if always_rebuild or size != (ppw * expected_size):
                args_list.append((ppw, path_length, alpha, random.Random(rand.randint(0, 2 ** 31)), file_))
            else:
                files.append(file_)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file_ in executor.map(_write_walks_to_disk, args_list):
            files.append(file_)

    return files


class WalksCorpus(object):
    def __init__(self, file_list):
        self.file_list = file_list

    def __iter__(self):
        for file in self.file_list:
            with open(file, 'r') as f:
                for line in f:
                    yield line.split()