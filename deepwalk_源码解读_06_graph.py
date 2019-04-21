#coding:utf-8
'''
author:zrant
'''

"""Graph utilities."""

# import logging

from io import open
# from time import time
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable, Counter
import random
from itertools import permutations
from scipy.io import loadmat
from scipy.sparse import issparse

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import cpu_count


# logger = logging.getLogger("deepwalk")
#
# __author__ = "Bryan Perozzi"
# __email__ = "bperozzi@cs.stonybrook.edu"
#
# LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


class Graph(defaultdict):
    """
    defaultdict与dict用法几乎一样，只是比dict多了一些用法而已
    """
    """
    我们更愿意使用defaultdict而不是dict主要原因是defaultdict比dict更不容易报错例如
    >>>x_defaultdict=defaultdict(int)
    >>>print(x_defaultdict["key1"])
    程序输出：0

    >>>x_dict=dict()
    >>>print(x_dict["key1"])
    程序输出：KeyError: 'key1'

    对于dict类型，你可以为没有出现过的key进行赋值，但是不可以直接使用
    例如可以做 x_dict["key1"]=1
    但是不可以做 x_dict["key1"]+=1
    """

    def __init__(self):
        """
        这句话的意思是，构建的图是一个字典，key是节点，key对应的value是list，表示该节点所在边的另一端节点序列
        Graph的__init__直接订制于defaultdict的__init__(default_factory=None, **kwargs)
        其中default_factory在这里被制定为list，即字典的value类型全部为list
        事实上如果你构建了一个Graph实例然后print出来后，会得到类似的输出形式：

        defaultdict(
                        <class 'list'>,

                        {
                            1: [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 18, 20, 22, 32],
                            2: [1, 3, 4, 8, 14, 18, 20, 22, 31],
                            ……
                            34: [9, 10, 14, 15, 16, 19, 20, 21, 23, 24, 27, 28, 29, 30, 31, 32, 33]
                        }
                    )

        当第一次出现某一个key时，这里会默认设置这个key的value为[]
        """
        # super(Graph, self).__init__(list) # 这是python2.x的语法
        super().__init__(list)  # 这是python3.x的语法

    def nodes(self):
        """返回图中的所有节点
        下面这个是dict固有的方法
        """
        return self.keys()

    def adjacency_iter(self):
        """
        iteritems() 和 viewitems() 是python2.x中dict的函数
        Python 3.x 里面，iteritems() 和 viewitems() 这两个方法都已经废除了
        在Python2.x中，items( )用于返回一个字典的拷贝列表，占额外的内存。
        iteritems() 用于返回本身字典列表操作后的迭代，不占用额外的内存。
        Python 3.x 里面，items() 得到的结果是和 2.x 里面 viewitems() 一致的。
        在3.x 里 用 items()替换iteritems() ，可以用于 for 来循环遍历。
        """

        """
        dict_items([(1, [2, 4, 5, 6]), (2, [1, 3]),...,)
        返回节点和它的邻居
        """
        return self.items()  # 源代码这里写的是iteritems()

    def subgraph(self, nodes={}):
        """
        这个函数返回的是graph的子图
        :param nodes:
        :return:
        """
        subgraph = Graph()

        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]
        """
        这里self相当于字典，n in self可以理解为关键字是否在字典中
        self[n]是找字典中关键字为n的对应的value，上面一行的代码意思是把nodes中的节点和对应的边
        添加到子图subgraph中
        值得注意的是，虽然源代码中nodes默认是{}，但是这个函数中并没有使用key对应的value，
        因此这里直接传一个list也是可以的。
        """
        return subgraph

    def make_undirected(self):
        """
        将有向图变为无向图
        :return:
        """

        # t0 = time()

        for v in self.keys():
            for other in self[v]:
                if v != other:
                    """
                          考虑这样一种情形：
                          1:[2,3...]
                          2:[4,...]
                          当遍历1号节点时，发现2在1的list中，于是执行self[2].append(1)，这样变成了：
                          1:[2,3...]
                          2:[4,...,1]
                          现在遍历到了2号节点，发现1在2的list中，于是执行self[1].append(2),这样变成了：
                          1:[2,3...,2] 出现了两个2
                          2:[4,...,1]
                          """
                    # if v != other and v not in self[other]:
                    # #这行代码可以替代上一行，不需要下面的去重操作，但是需要排序
                    self[other].append(v)

        # t1 = time()
        # logger.info('make_directed: added missing edges {}s'.format(t1 - t0))

        self.make_consistent()
        """这个函数的作用是：
        1. 用来去除重复的，因为之前的工作会得到酱紫的输出：
            {1: [2, 5, 6, 2, 4, 5, 6], 2: [1, 3], ...}，每一对节点都保留了两条边
        2. 同时，这个函数还可以保证每一项的list是从小到达排序的。
        3. 最后这个函数内部调用了remove_self_loops()，可以去除掉自循环（自己到自己的边）
        """
        return self

    def make_consistent(self):
        """这个函数的作用是：
        1. 用来去除重复的，因为之前的工作会得到酱紫的输出：
            {1: [2, 5, 6, 2, 4, 5, 6], 2: [1, 3], ...}，每一对节点都保留了两条边
        2. 同时，这个函数还可以保证每一项的list是从小到达排序的。
        3. 最后这个函数内部调用了remove_self_loops()，可以去除掉自循环（自己到自己的边）
        """
        # t0 = time()
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))
            # 原来的每一项list self[k]先进行set函数去重，
            # 然后经过sorted函数进行排序，
            # 最后转化为list进行重新赋值
        # t1 = time()
        # logger.info('make_consistent: made consistent in {}s'.format(t1 - t0))

        # print("not remove_self_loops", self)
        self.remove_self_loops()
        # print("remove_self_loops:",self)

        return self

    def remove_self_loops(self):
        """
        去除掉节点到自己的自循环
        :return:
        """
        # removed = 0
        # t0 = time()

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                # removed += 1

        # t1 = time()
        #
        # logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1 - t0)))
        return self

    def check_self_loops(self):
        """
        检查是否含有自回路
        :return:
        """
        for x in self:
            for y in self[x]:
                if x == y:
                    return True

        return False

    def has_edge(self, v1, v2):
        """
        判断两个节点之间是否有边
        :param v1:
        :param v2:
        :return:
        """
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None):
        """
        这个函数用来求节点的度数对于有向图而言，求的是入度或初度（具体是哪个要看你到dic是根据什么构建的）
        输入的nodes参数可以是一个节点编号，也可以是一系列节点的可迭代对象如list。如果是单个节点，则直接返回
        节点的度，如果是一些列节点，则返回dic，key对应为目标节点，value为该节点对应的度数。
        :param nodes:
        :return:
        """
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    # def order(self):
    #     "Returns the number of nodes in the graph"
    #     """
    #     这段代码仅仅被下面的number_of_nodes函数调用，其实没必要在这里单独开一个函数，之所以这样写，是因为
    #     这套代码是deepwalk新版本的代码，为了向下兼容，保留了这个order的接口，在这里我们把这个接口直接删掉
    #     """
    #     return len(self)

    def number_of_edges(self):
        "Returns the number of nodes in the graph"
        """
        请注意，这里的边的个数对应的是无向图，如果是有向图，需要去掉除以2的操作
        """
        return sum([self.degree(x) for x in self.keys()]) / 2

    def number_of_nodes(self):
        "Returns the number of nodes in the graph"
        # return self.order()  # 这个源代码
        return len(self)  # 这个是我们删掉了order旧接口后的代码

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
            返回一些较短的随机游走路径
            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
            random.Random()：产生0-1之间的随机浮点数
            请注意：这里的随机游走路径未必是连续的，有可能是走着走着突然回到起点接着走
        """
        G = self
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:  # 当cur有邻居时
                if rand.random() >= alpha:  # 这是一个经典的概率编程的语句
                    """
                    这条语句成立的概率是1-alpha
                    """
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])  # 这里应该写的是path.append(cur)吧？
                    """
                    以概率1-alpha从当前节点继续向前走，或者以alpha的概率restart
                    """
            else:
                break
        return [str(node) for node in path]


def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
    """
    这个函数可以对一个图生成一个语料库
    :param G:
    :param num_paths:
    :param path_length:
    :param alpha:
    :param rand:
    :return:
    """
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        # 这条语句将下面的random walk过程重复了num_paths次，
        # 相当于每一个节点做num_paths次random walk
        rand.shuffle(nodes)
        for node in nodes:  # 这条语句对图所有的节点各自进行一次random walk
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))

    return walks


def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                               rand=random.Random(0)):
    """
    这个函数可以迭代地生成语料库
    :param G:
    :param num_paths:
    :param path_length:
    :param alpha:
    :param rand:
    :return:
    """

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def clique(size):
    """

    :param size:
    :return:
    """

    return from_adjlist(permutations(range(1, size + 1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def parse_adjacencylist(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            introw = [int(x) for x in l.strip().split()]
            row = [introw[0]]
            row.extend(set(sorted(introw[1:])))
            adjlist.extend([row])

    return adjlist


def parse_adjacencylist_unchecked(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            adjlist.extend([[int(x) for x in l.strip().split()]])

    return adjlist


def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):
    if unchecked:
        parse_func = parse_adjacencylist_unchecked
        convert_func = from_adjlist_unchecked
    else:
        parse_func = parse_adjacencylist
        convert_func = from_adjlist

    adjlist = []

    # t0 = time()

    with open(file_) as f:
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            total = 0
            for idx, adj_chunk in enumerate(executor.map(parse_func, grouper(int(chunksize), f))):
                adjlist.extend(adj_chunk)
                total += len(adj_chunk)

    # t1 = time()
    #
    # logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1 - t0))

    # t0 = time()
    G = convert_func(adjlist)
    # t1 = time()
    #
    # logger.info('Converted edges to graph in {}s'.format(t1 - t0))

    if undirected:
        # t0 = time()
        G = G.make_undirected()
        # t1 = time()
        # logger.info('Made graph undirected in {}s'.format(t1 - t0))

    return G


def load_edgelist(file_, undirected=True):
    G = Graph()
    with open(file_) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = int(x)
            y = int(y)
            G[x].append(y)
            if undirected:
                G[y].append(x)

    G.make_consistent()
    return G


def load_matfile(file_, variable_name="network", undirected=True):
    mat_varables = loadmat(file_)
    mat_matrix = mat_varables[variable_name]

    return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
        raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G

