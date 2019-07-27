#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import random, randint, choice
from copy import deepcopy
from math import log


# In[2]:


class fwrapper:
    def __init__(self, function, childcount, name):
        self.function = function
        self.childcount = childcount
        self.name = name


# In[3]:


class node:
    """ A node that store the class of operation unit
    
    Args:
        fw(fwrapper): the operation unit define by fwrapper
        children(int): the child count of fw in order to fit the operation necessary.
    """
    def __init__(self, fw, children):
        self.function = fw.function
        self.name = fw.name
        self.children = children
    def evaluate(self, inp): # inp is input
        results = [n.evaluate(inp) for n in self.children]
        return self.function(results)
    def display(self, indent = 0):
        print( ' '*indent + self.name)
        for c in self.children:
            c.display(indent + 1)


# In[4]:


class paramnode:
    """ A leaf node that store random value from input
    
    Args:
        idx(int): index that point to the target input 
    """
    def __init__(self, idx):
        self.idx = idx
    def evaluate(self, inp):
        return inp[self.idx]
    def display(self, indent = 0):
        print('%sp%d' % (' '*indent, self.idx))


# In[5]:


class constnode:
    """ A leaf node that store constant value
    
    Args:
        v(float): constant value
    """
    def __init__(self, v):
        self.v = v
    def evaluate(self, inp):
        return self.v
    def display(self, indent = 0):
        print('%s%d' % (' '*indent, self.v))


# In[6]:


addw = fwrapper(lambda I:I[0] + I[1], 2, 'add')
subw = fwrapper(lambda I:I[0] - I[1], 2, 'subtract')
mulw = fwrapper(lambda I:I[0] * I[1], 2, 'multiply')
def iffunc(I):
    if I[0] > 0:
        return I[1]
    else:
        return I[2]
ifw = fwrapper(iffunc, 3, 'if')
def isgreater(I):
    if I[0] > I[1]:
        return 1
    else:
        return 0
gtw = fwrapper(isgreater, 2, 'isgreater')


# In[7]:


flist = [addw, mulw, ifw, gtw, subw]


# In[8]:


def makerandomtree(pc, maxdepth = 4, fpr = 0.5, ppr = 0.6):
    """ build a random tree(model)
    
    Args:
        pc(int): the length of input vector
        fpr(float): the probability of new node belong to class node
        ppr(float): the probability of new node belong to class paramnode
        
    Return:
        list: a list of node represent the tree(model)
    """
    if random() < fpr and maxdepth > 0:
        f = choice(flist)
        children = [makerandomtree(pc, maxdepth-1, fpr, ppr) for i in range(f.childcount)]
        return node(f, children)
    elif random() < ppr:
        return paramnode(randint(0, pc-1))
    else:
        return constnode(randint(0,10))


# In[9]:


def scorefunction(tree, s):
    """ use Euclidean Distance to measure the distance
    
    Args:
        tree(list): a list of node that represent the tree(model)
        s(list) : the training dataset and ground truth, the length of list is 3 ,length input vector is 2 and output is 1
        
    Return:
        float: the Euclidean Distance between model predict output and ground truth
    """
    dif = 0
    for data in s:
        v = tree.evaluate([data[0], data[1]])
        dif += abs(v - data[2])
    return dif


# In[10]:


def mutate(t, pc, probchange = 0.1):
    """ mutate will change a bit of tree to randomtree according to the probchange
        
    Args:
        pc(int): the length of input
        probchange(float): the prob of a input tree to be change to random tree
        
    Return:
        list: a list of node represent the tree(model)
    """
    if random() < probchange:
        return makerandomtree(pc)
    else:
        result = deepcopy(t)
        # if t is a node(tree), then we test every child of t.
        if isinstance(t, node):
            result.children = [mutate(c,pc,probchange) for c in t.children]
        return result


# In[11]:


def crossover(t1, t2, probswap=0.7, top=1):
    """ to swap the subtree of two different tree according to the probswap
    
    Args:
        t1(list): a list of node that represent the tree(model)
        t2(list): a list of node that represent the tree(model), which is different from t1
        probswap(float): the prob of swapping
        top(bool): a flag that represent if top or not
    """
    if random() < probswap and not top:
        return deepcopy(t2)
    else:
        result = deepcopy(t1)
        if isinstance(t1, node) and isinstance(t2, node):
            # child of t1 crossover with child of t2
            result.children = [crossover(c, choice(t2.children), probswap, 0) for c in t1.children]
        return result


# In[12]:


def getrankfunction(dataset):
    """ initialize training dataset and groundtruth from dataset,
        then we take several trees as input to get a list of scores from scorefunction,
        and return a sort score list
    """
    def rankfunction(population):
        scores = [(scorefunction(t, dataset),t) for t in population]
        # we sort only the first num, otherwise sort() will compare second num if first nums are the same
        scores.sort(key=lambda x: x[0])
        return scores
    return rankfunction


# In[13]:


def evolve(pc, popsize, rankfunction, maxgen=500, mutationrate=0.1, breedingrate=0.4, pexp=0.7, pnew=0.05):
    """ the whole process of genetic programming including make random tree, mutation and crossover to select winners from every generation.
    """
    def selectindex():
        # smaller pexp will return a samller random value
        return int(log(random())/log(pexp))
    # make random trees
    population = [makerandomtree(pc) for i in range(popsize)]
    for i in range(maxgen):
        scores = rankfunction(population)
        # score[i][0] indicate the score of ith tree, score[i][1] is the tree itself
        print(scores[0][0])
        if scores[0][0] == 0:
            break
        #make sure newpop always have the top 2, and select the rest by chance(depend on selectindex)
        newpop = [scores[0][1], scores[1][1]]
        while len(newpop) < popsize:
            if random() > pnew:
                newpop.append(mutate(crossover(scores[selectindex()][1], 
                                                scores[selectindex()][1],
                                              probswap=breedingrate),
                                    pc, probchange=mutationrate))
            else:
                newpop.append(makerandomtree(pc))
        population = newpop
    scores[0][1].display()
    return scores[0][1]

