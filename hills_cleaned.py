#!/usr/bin/env python
import os
import pathlib
import glob
import argparse
import random
import sys
import time
import pandas as pd
import numpy as np
import collections
import itertools
import datetime
import pickle
import signal
#from queue import PriorityQueue
import heapq
from typing import Callable, Sequence
from tqdm import tqdm

import utils
import evalmetricp as metric

'''
nth try at optimizing word order
uses LKH ideas
use a heapq to find best next node to explore
use Node class
We generate the list of permutations to search over as an NxM array of permutations
Use recursion to do a depth first search of nodes, taking only improvements, with a max number.
If no improvements found, we can decide how many if any should be explored 
Using recursion means we do not need a priority queue and have to remember what iteration
we left off at to continue the search; with a random search order, this is more difficult (we could
keep a list of the remaining perm indexes to search, or a seed into the random permutation generator
with an index into the number last done, but these seem more difficult.
The queue would allow the search to be stopped and saved, and also allows us to see what 
the status of the search is more clearly.

* do a "kick" when the search fails, which is some randomization, keeping some of what is good, hopefully

Need to set BEST_SUB for correct file for --file best to work

Large Parts stolen from teammate @Kaizaburo Chubachi - Thank you!
* added --opt = (a,b,c,d)  for enumerate_kopt(a,b) + enumerate_kopt(b,d)
* --phrase forces use of hardcoded phrases by problem index
* --beam
* --maxb
* --worse: number of levels to go before must see an improvement over the last improved node
hillsv5
* back to delins (rotate) operations
parameters are
total_branches = 100?
beam = 2   #will be improvements if possible, else up to 2 of other ones (random or next best?)
* Writes out pickle file every round with prior text->perp calculations
* reads in pickle file and uses that for shortcutting calculations
hillsv6
* Went back to using a priority queue
* store the perm array and the last index that was searched at the node
  so that the search can continue at that node.  Also need to store the_nodes
  best N scores so far (and their perm indexes) so they can be recreated if we need
  to go uphill 
'''
parser = argparse.ArgumentParser(
    description="search program 001",
    epilog='''
brute force searching''')

parser.add_argument("index", type=int, default=0, help='which problem to work on')
parser.add_argument("--rounds", type=int, default=None, help='how many rounds to do')
parser.add_argument("--rand", action='store_true', help='shuffle starting point each round')
parser.add_argument("--file", type=str, default=None, help='file to start with in results dir')
parser.add_argument("--state", type=str, default=None, help='read state from file')
parser.add_argument("--batch", type=int, default=None, help='batch size')
parser.add_argument("--seed", type=int, default=None, help='seed to use for all rands')
parser.add_argument("--time", type=float, default=None, help='time in hours to quit')
parser.add_argument("--rtime", type=float, default=None, help='time in hours for each round')
parser.add_argument("--cpu", action='store_true', help='force cpu usage')
parser.add_argument("--phrase", action='store_true', help='force phrases')
parser.add_argument("--swaps", action='store_true', help='add word swaps near the end')
parser.add_argument("--stdout", action='store_true', help='use stdout instead of log file')
parser.add_argument("--log", type=str, default='hv6', help='string to use in log file name')
parser.add_argument("--test", action='store_true', help='no writing of soln, stdout, other test stuff')
parser.add_argument("--rstart", action='store_true', help='start next round with random soln from previous round')
parser.add_argument("--gen", action='store_true', help='generate output file with all 2 step perturbations')
parser.add_argument("--skip-first-kick", action='store_true', help='do that')
parser.add_argument("--opt", type=int, nargs='+', default=None, help='(a,b), (c,d) args to use in enumerate_kopts, def (3,5,4,1)')
parser.add_argument("--first", type=str, default=None, help='lock first word to this')
#search parameters:
#At each node:
parser.add_argument("--nimp", type=int, default=2, help='max number of improvements (stop search after this number)')
parser.add_argument("--nuphill", type=int, default=1, help='max number uphill to explore')
parser.add_argument("--beam", type=int, default=2, help='max number total to explore')
parser.add_argument("--maxd", type=int, default=100, help='max depth of nodes to explore at')
#overall
parser.add_argument("--nworse", type=int, default=5, help='max worse in depth')
options = parser.parse_args()
device = 'cpu' if options.cpu else None
DEPTH_FIRST=False
#Change this as needed, used for --file best
BEST_SUB = 'subs/best_submission.csv'
#Change batch sizes
batch_size = utils.get_batch_size(options.index)
if options.batch is not None:
    batch_size=options.batch

print(f'batch_size = {batch_size}')    
write_threshold={0: 500, 1:450, 2:335, 3:230, 4:90, 5:40}
scorer = None
#Make results directory
#results_dir = f'/mnt/speedy/santa2024/results/{options.index}'
results_dir = f'results/{options.index}'
p = pathlib.Path(results_dir)
p.mkdir(exist_ok=True, parents=True)
#Make state directory
state_dir = f'./state'
p = pathlib.Path(state_dir)
p.mkdir(exist_ok=True, parents=True)
best_node = None

if options.seed is not None:
    seed = options.seed
else:
    seed = random.randint(0,100000)
print(f'using random seed {seed}')
random.seed(seed)
np.random.seed(seed)
all_prior_explores=[]  #list of dicts containing prior round exporations

## Set constraint phrases, if needed
constraint_phrases=[]
if options.phrase:
    if options.index==3:
        constraint_phrases=['gifts of the magi']
    elif options.index==4:
        constraint_phrases=[]
    elif options.index==5:
        constraint_phrases=[]
#constraint_first_words=['magi', 'jingle', 'sleigh', 'chimney', 'stocking', 'ornament', 'decorations']
constraint_first_words=[]
fixing_intervals=None
if options.first:
    constraint_first_words=[options.first]
    print(f'locking first word to {options.first}')
    fixing_intervals=[0,1]   #do not move the first word
    print(f'{constraint_first_words=}')
print(f'{constraint_phrases=}')
the_nodes=[]

#use these to get perplexity
#NOTE: get_perps get a list of texts with batch size, get_score gets a single one
#Both will generate scorer when needed
def get_score(text: str) -> float:
    #batch size of 1
    global scorer
    if scorer is None:
        scorer = metric.PerplexityCalculator(load_in_8bit=False, device=device)
        #scorer.clear_gpu_memory(tokenizer=False, model=False)
    perps = scorer.get_perplexity([text], batch_size=1)
    return perps[0]

def get_perps(trials: list[str]) -> list[float]:
    #get a perps for a list of texts
    global scorer
    if scorer is None:
        scorer = metric.PerplexityCalculator(load_in_8bit=False, device=device)        
    perps = scorer.get_perplexity(trials, batch_size=batch_size)
    return perps

def join(w):
    return ' '.join(w)

all_prior_trials=[]       #list[dict]
all_prior_solutions=[]    #list[dict]
def write_state(all_trials: dict, all_solutions: dict, round: int):
    #all_trials is all evaluations done -> perp
    #all_solutions is all explored text -> level
    #will not capture the last expansion done
    state = {'all_trials': all_trials, 'all_solutions': all_solutions}
    file = f'state/trialsv6-{options.index}-{dbg.code}-R{round}.pickle'
    with open(file, 'wb') as fp:
        pickle.dump(state, fp)
    print(f'wrote state file {file}')
    #we append the new info to the current lists in memory since it is a new round finished
    all_prior_trials.append(all_trials)
    all_prior_solutions.append(all_solutions)

def del_from_dict(d, limit):
    #remove items from d whose value is above limit
    for k in list(d.keys()):
        if d[k] > limit:
            del d[k]
    
def read_states():
    mem_per_entry={0:200, 1:250, 2:254, 3:300, 4:400, 5:500}[options.index]
    global state
    c=0
    delta = 300
    for file in glob.glob(f'state/trialsv?-{options.index}-*.pickle'):
        with open(file, 'rb') as fp:
            state = pickle.load(fp)
        thresh = delta + write_threshold[options.index]
        del_from_dict(state['all_trials'], thresh)
        all_prior_trials.append(state['all_trials'])
        all_prior_solutions.append(state['all_solutions'])
        c+=1
    print(f'read in {c} state files, deleted above thresh {delta=}')
    tot_trials = sum([len(x) for x in all_prior_trials])
    tot_soln = sum([len(x) for x in all_prior_solutions])
    print(f'all trials:     len {tot_trials}')
    print(f'all solutions:  len {tot_soln}')
    MEM_DESIRED=6_000_000_000
    delta = int(delta*.80)
    while tot_trials > MEM_DESIRED/mem_per_entry:
        limit = delta + write_threshold[options.index]
        tot = 0
        for trials in all_prior_trials:
            del_from_dict(trials, limit)
            tot += len(trials)
        print(f'after deleting with delta {delta} limit {limit} there are {tot} entries in all trials')
        delta = int(delta * .80)
        tot_trials = tot
        if delta < 10:
            print(f'cannot delete enough trials')
            sys.exit(1)
    #now merge into a single dictionary for speed in lookup
    new_prior_trials={}
    for d in all_prior_trials:
        new_prior_trials.update(d)
    all_prior_trials.clear()  #keep global variable
    all_prior_trials.append(new_prior_trials)
                

def write_soln(node, round=None, iter=0, end=False):
    #end means best for this round
    if end:
        level='Y' if node.truncated else 'Z'
    else:
        level = str(iter)
    file = f'prob-{options.index}-v6-{node.perp:08.3f}-R{round}_{level}-{dbg.code}.words'
    if options.test:
        print(f'Wrote - not really, test mode {file} H{node.hash} {node.perp:08.3f}')
        return
    if not end and options.index in write_threshold and node.perp > write_threshold[options.index]:
        print(f'NotWrote solution #{options.index} {node}')
        return
    with open(os.path.join(results_dir,file), 'wt') as fp:
        fp.write(node.text)
    print(f'Wrote solution {file} H{node.hash} {node.perp:08.3f}')

def a_prior_solution(text):
    for d in all_prior_solutions:
        if text in d:
            return True
    return False

def look_up_text(text):
    for d in all_prior_trials:
        try:
            return d[text]
        except KeyError:
            continue
    raise KeyError('not found in all_prior_trials' )

def signal_handler(sig, frame):
    #write state and exit
    #global quit
    #file = os.path.join(state_dir, f'{dbg.code}-{options.index}-hillsv4.state')
    #write_state(file)
    #print(f'wrote state to {file}, quitting')
    print('  Ctrl+C...quitting...')
    sys.exit(1)
    
signal.signal(signal.SIGINT, signal_handler)

def check_phrases(words, phrases):
    #make sure all the phrases are present.  Should we cache some of this?
    for phrase in phrases:
        w = phrase.split()
        m = len(w)
        k = words.index(w[0])
        if words[k:k+m] != w:
            return False
    return True
    
def check_first(words, first_words):
    if first_words:
        return words[0] in first_words
    return True

def check_constraints(words):
    #return True if this is an OK sequence
    return all([
        check_phrases(words, constraint_phrases),
        check_first(words, constraint_first_words)
        ])


#get a repeatable hash - seems to be around 1usec.
import hashlib
def xhash(s, length=6):
    #hash of string s
    h=hashlib.new('md5')
    h.update(s.encode())
    return h.hexdigest()[-8:]


class Node():
    """
    For phrases, we will consider each locked phrase as a word.  So node.words will 
    return a list like ["chimney", "gifts of the magi", "of", "the"] etc.
    node.len will return the smaller len(words) where a phrase is considered a word
    """
    problem_words = None
    problem_num_words=0
    @classmethod
    def set_words(cls, words: list[str], phrases: list[str]) -> None:
        #words: the full set of words
        #phrases: ['gifts of the magi', 'chocolate milk']
        Node.problem_all_words=words.copy()
        Node.phrases = phrases.copy()
        #now make the list of words with phrases intact
        words = words.copy()
        for phrase in phrases:
            for w in phrase.split():
                words.remove(w)
        #words are now what is left over when phrases taken out
        for phrase in phrases:
            words.append(phrase)
        Node.problem_words = words
        Node.problem_num_words = len(Node.problem_words)
        Node.problem_num_all_words=len(Node.problem_all_words)
    @classmethod                
    def get_phrases(cls, words: list[str]) -> list[str]:
        #return a list of phrases from the list of original words, we use this
        #to get started.  Thereafter, the words will be shuffled by k-opt/kick, etc.
        words = words.copy()
        for phrase in cls.phrases:
            words_in_phrase = phrase.split()
            phrase_len = len(words_in_phrase)
            x = words.index(words_in_phrase[0])
            #print(f'{x=} {phrase=} {words_in_phrase=} {words=}')
            assert words[x:x+phrase_len] ==  words_in_phrase
            #replace the first word of the phrase with the phrase, delete the rest
            words[x]=phrase
            del words[x+1:x+phrase_len]
        return words
    def __init__(self, words_or_phrases, perp=0, level=0,
                 last_improved_level=None,
                 last_improved_perp=None):
        #words or text can be used for init
        #we cannot use split to go back to words because of phrases
        #needs to be words and phrases
        assert isinstance(words_or_phrases, list)
        assert len(words_or_phrases) == Node.problem_num_words
        self._words=words_or_phrases.copy()
        self.perp = perp
        self._level = level
        self.truncated=False   #whether search stopped early
        if last_improved_level is None:
            self.last_improved_level=level
        else:
            self.last_improved_level=last_improved_level
        if last_improved_perp is None:
            self.last_improved_perp=perp
        else:
            self.last_improved_perp=last_improved_perp
        #self.perm_array=None
        #self.last_index_searched=None
    @property
    def len(self):
        return len(self.words)
    def all_len(self):
        return Node.problem_num_all_words
    @property
    def hash(self):
        return xhash(' '.join(self._words))
    @property
    def level(self):
        return self._level
    @property
    def words(self):
        return self._words
    @property
    def text(self):
        return ' '.join(self._words)
    #Define comparison operators so heapq will be happy
    def __lt__(self, other):
        return self.perp < other.perp

    def __gt__(self, other):
        return self.perp > other.perp
    
    def __eq__(self, other):
        #do we need text too?
        return self.perp == other.perp
    @property
    def shortdesc(self):
        return f'{self.perp:8.3f} H{self.hash} L{self._level} [..{text[-40:]}]'
    
    def __str__(self):
        text = self.text
        return f'{self.perp:8.3f} H{self.hash} L{self._level}:{self.last_improved_level} [{text}]'
    
    def str(self):
        #see phrases, separate phrases with '/'
        t = '/'.join(self._words)
        return f'{self.perp:8.3f} [{t}]'

#Find a sequence to start with
file='data/sample_submission.csv'
if options.file == 'best':
    file=BEST_SUB
elif options.file is not None:
    file=options.file
print(f'reading from {file}')        
words = utils.read_words(file, options.index, results_dir=f'results/{options.index}')
num_words=len(words)
print(f'num words (before phrasing) {num_words} batch_size {batch_size}')

#Fix up any constraints on phrases

def swap(words, a, b):
    words[a],words[b]=words[b],words[a]

def check_and_fix_constraints(word):
    if check_constraints(words):
        return
    #modify words list to fix constraints.  This is words before phrases converted
    #Just doing 1 phrase constraint for now
    #note a problem if first word is in constraint phrase
    print(f'trying to fix constraints')
    if not check_phrases(words, constraint_phrases):
        for i, w in enumerate(constraint_phrases[0].split()):
            k = words.index(w)
            swap(words, k, i+1)
    if constraint_first_words:
        print(f'fixing first word to required {constraint_first_words[0]}')
        k = words.index(constraint_first_words[0])
        swap(words, k, 0)
    assert check_constraints(words)

#iterator for r points
#do dumb way first, return tuple of points
def gen_points(width, r):
    if r==3:
        for i in range(0, width-1):
            for j in range(i+1,width):
                for k in range(j+1,width+1):
                    yield (i, j, k)
    elif r==4:
        for i in range(0, width-2):
            for j in range(i+1,width-1):
                for k in range(j+1,width):
                    for q in range(k+1,width+1):
                        yield (i, j, k, q)
    else:
        raise ValueError(f'gen_points r={r} not implemented yet')



def double_bridge_kick(words: list[str]) -> str:
    #return a single random Double Bridge Kick: ABCDE => ADCBE
    #each piece other than 1st and last must be size >= 2
    print("Double Bridge Kick")
    while True:
        A, C, E, G = sorted(np.random.choice(len(words)+1, size=4, replace=False))
        B = A + 1
        D = C + 1
        F = E + 1
        if A < B < C < D < E < F < G < len(words):
            break
    #same order    [words[:A]+words[A:C]+words[C:E]+words[E:G]+words[G:]
    new_words = words[:A] + words[E:G] + words[C:E] + words[A:C] + words[G:]
    assert len(new_words) == len(words)
    return new_words

def subsequence_random_permutation(base_words: list[str], n: int = 5) -> str:
    print(f'Subsequence Random Permutation {n=}')
    words = np.asarray(base_words)
    assert len(words) >= n
    indices = np.random.choice(len(words), n, replace=False)
    subsequence = words[indices].copy()
    np.random.shuffle(subsequence)
    words[indices] = subsequence
    #print('subseq', words)
    #print(type(words))  #np.ndarray, elements np.str_
    #need to convert ndarray back into a list of words (str)
    words=[str(w) for w in words]
    return words

def big_kick(words: list[str], k: int = 6) -> str:
    #return a single random Kick, with k points, and k+1 pieces randomly deranged
    #with k points
    while True:
        pts = sorted(np.random.choice(len(words)+1, size=k, replace=False))
        mn_space = min( [b-a for a,b in zip(pts, pts[1:])])
        if mn_space >= 1:
            break
    pieces = [words[:pts[0]]] + [words[a:b] for a,b in zip(pts, pts[1:])] + [words[pts[-1]:]]
    #print(f'{pieces=}')
    #We don't want a derangement, where every piece moves, we want to make sure that
    #each actually moves, so perm must not have j,j+1 sequential
    while True:
        perm = np.random.permutation(k+1)
        if not any([perm[i]+1 == perm[i+1] for i in range(k)]):
        #if not any([perm[i]==i for i in perm]):   # a derangement
            break
    print(f'Big Kick k={k} points {[int(p) for p in pts]} perm {perm}')
    new_pieces = [pieces[i] for i in perm]
    new_words = list(itertools.chain.from_iterable(new_pieces))
    #print(f'{new_words=}')
    #same order    [words[:A]+words[A:C]+words[C:E]+words[E:G]+words[G:]
    #new_words = words[:A] + words[E:G] + words[C:E] + words[A:C] + words[G:]
    assert len(new_words) == len(words)
    return new_words

def kick_word(node: Node,
              fixing_intervals: list[int],
              kicks: list[tuple[float, Callable]]):
    #randomly perturb base_text by list of kicks (with certain prob)
    #Changed: only take the first one
    #fixing_intervals: sequence of indexes, each pair represents range of words to not move
    assert len(fixing_intervals) % 2 == 0
    assert fixing_intervals == sorted(fixing_intervals)
    words = node.words
    sections = [0] + fixing_intervals + [len(words)]

    moving_words = []
    for i, (s1, s2) in enumerate(zip(sections[:-1], sections[1:], strict=True)):
        if i % 2 == 0:
            moving_words.extend(words[s1:s2])

    moving_text = " ".join(moving_words)
    moved_words = moving_words
    #Changing to apply each one with certain prob, so can have more than 1 applied
    #must have the words changed, though
    while ' '.join(moved_words) == moving_text:
        for p, kick_func in kicks:
            if np.random.rand() < p:
                moved_words = kick_func(moved_words)
                break  #only do 1 kick in the list

    moved_index = 0
    new_words = []
    for i, (s1, s2) in enumerate(zip(sections[:-1], sections[1:], strict=True)):
        if i % 2 == 0:
            new_words.extend(moved_words[moved_index : moved_index + s2 - s1])
            moved_index += s2 - s1
        else:
            new_words.extend(words[s1:s2])
    assert moved_index == len(moved_words)
    assert sorted(words) == sorted(new_words)
    return Node(new_words, perp=0)  # perplexity needs to be filled in by caller

def shuffle_word(node: Node):
    #shuffle words in Node and return a new node
    print("Shuffle Words")
    words = node.words
    new_words=words.copy()
    random.shuffle(new_words)
    return Node(new_words, perp=0)  # perplexity needs to be filled in by caller

def enumerate_kopt(num_words: int, k: int, max_moving_size_threshold: int | None = None)-> np.ndarray:
    #the pertubations now return an NxM array of permutations
    #k=3 means 3 points giving pieces ABCD -> ACBD
    #k=4 means 4 points giving pieces ABCDE -> ADCBE
    #the end pieces can be size 0, the other cannot
    #for points a,b,c the pieces are [:a],[a:b],[b:c],[c:];  a can be 0, c can be num_words
    assert k in (3,4)
    if max_moving_size_threshold==0:
        #dont do anything
        return np.zeros((0,num_words), dtype=np.int8)  # so it can be concatenated
    orders=[]
    for points in gen_points(num_words, k):
        if (max_moving_size_threshold is not None) and (
                max_moving_size_threshold < min(s2 - s1 for s1, s2 in zip(points[:-1], points[1:], strict=True)) ):
            continue
        order = list(range(num_words))
        if k==3:
            #no perm would be:  order[:a] + order[a:b] + order[b:c] + order[c:]
            a,b,c=points
            order = order[:a] + order[b:c] + order[a:b] + order[c:]
        elif k==4:
            a,b,c,d=points
            #order = order[:a] + order[a:b] + order[b:c] + order[c:d] + order[d:]
            order = order[:a] + order[c:d] + order[b:c] + order[a:b] + order[d:]
        else:
            ValueError(f'enumerate_kopt, bad k={k}')
        orders.append(order)
    return np.asarray(orders, dtype=np.int8)       

def rotater(x):
    #rotate right
    return x[-1:] + x[:-1]
def rotatel(x):
    #rotate left
    return x[1:] + x[:1]

def enumerate_cycles(num_words: int, max_d: int | None = None, swaps: bool=False)-> np.ndarray:
    #the pertubations now return an NxM array of permutations
    #returns pertubations for cyles left and right, max_d is size of maximum rotated piece
    if max_d==0:
        #dont do anything
        return np.zeros((0,num_words), dtype=np.int8)  # so it can be concatenated
    orders=[]
    if max_d is None:
        max_d = num_words-1
    for d in range(2,max_d+1):
        cycle_a=0
        cycle_b=num_words+1-d
        order = list(range(num_words))
        for i in range(cycle_a,cycle_b):
            new_order = order[:i] + rotater(order[i:i+d]) + order[i+d:]
            assert len(new_order) == num_words
            orders.append(new_order)
            if d>2:
                new_order = order[:i] + rotatel(order[i:i+d]) + order[i+d:]
                assert len(new_order) == num_words
                orders.append(new_order)
            if swaps and 2<d and d<6 and num_words-i-d <= 2:
                #add in swaps for cycle must end within 2 of the end
                new_order=order.copy()
                new_order[i],new_order[i+d-1] = new_order[i+d-1],new_order[i]
                orders.append(new_order)
    return np.asarray(orders, dtype=np.int8)       


all_solutions_this_round={}  #contains a dict of text->score
all_trials={}   # for this round 
def explore_node(node: Node,
                 permutations: np.ndarray,
                 nimp: int = 2,
                 nuphill: int = 1,
                 beam: int = 2,
                 nworse:int = 5,
                 write_threshold: float=None,   #call write_soln if beat this (usually best so far?)
                 ) -> tuple[bool, Node]:
    #explore seaches from current node
    #returns terminate round (bool), best node it found (possibly 'node') 
    #all trials is a dict of all the scores we have done - we can write to a state file for later use
    #all trials makes sure we do not explore the same node twice
    num_words = node.len
    #print(f'{num_words=} {node.str()}')
    num_perm = permutations.shape[0]
    #does keeping a list of trials use too much memory for recursion?
    best_trial=None
    num_improvements=0  #start counting
    assert permutations.shape[1] == num_words
    test_order = np.random.permutation(num_perm)
    words = node.words
    assert len(words) == num_words
    best_node = node
    #print(f'test_all permutations {num_perm}')
    trials={}     #all trials->perm index
    do_trials={}  #ones to evaluate
    total_improvements=0  # for improvements
    total_explores=0      #for all things
    map_perm_index_to_score={}  # use so we can go back and find best non-improved scores
    known_trials_score={}  #this will be the ones we already know from the state
    #if a_prior_solution(node.text):
    #    print(f'this node was found in a prior solution, ending')
    #    #How do we communicate this to main program.  For now, not doing
    #    #could have a terminate search return value
    #    node.truncated=True
    #    return True, best_node    
    for perm_index in range(num_perm):
        counter=0
        perm = permutations[test_order[perm_index]]
        #print(f'order={test_order[perm_index]} {perm=}')
        #print(f'AA {words=}')
        new_words=[words[i] for i in perm]
        new_text = join(new_words)
        if new_text in all_trials:
            #we have seen it this round, so skip 
            if options.test:
                print(f'{perm_index=} order={test_order[perm_index]} found in all_trials H{xhash(new_text)} {new_text}')
            continue
        if new_text in do_trials:
            if options.test:
                print(f'{perm_index=} order={test_order[perm_index]} found in do_trials H{xhash(new_text)} {new_text}')
            #do not repeat any texts 
            continue
        elif new_text==node.text:
            #same as original
            continue
        trials[new_text]=perm_index
        try:
            known_trials_score[new_text]=look_up_text(new_text)
            continue
        except KeyError:
            #I guess we have to compute it
            pass
        #remember the perm_index so we can recreate the words list; using a dict to guarantee unqiue
        do_trials[new_text]=perm_index
        #perm_index_list.append(perm_index)  # 
        #print(f'{len(trials)=}')
        if len(do_trials) >= batch_size or perm_index == num_perm-1:
            if len(known_trials_score)>0:
                print(f'  got batch of {len(do_trials)} prev known {len(known_trials_score)}')
            #last one or a batch to do
            trials_list = list(do_trials.keys())
            perps = get_perps(trials_list)
            #num_improvements = len([1 for x in perps if x < node.perp])
            #print(f'found at i=[{1+perm_index-len(trials)}-{perm_index}] {num_improvements} improvement')
            for t,p in known_trials_score.items():
                trials_list.append(t)
                perps.append(p)
            all_trials.update({t:p for t,p in zip(trials_list, perps)})
            #trials_list now has all the trials, including ones previously known
            #we should score all the ones we evaluated, even though we would terminate on nimp
            for j, (text, score) in enumerate(zip(trials_list, perps, strict=True)):
                map_perm_index_to_score[trials[text]] = score
                if score < node.perp:
                    total_improvements+=1
            if total_improvements >= nimp:
                break
            do_trials.clear()
            known_trials_score.clear()
    #Now go through all the nodes and figure out which to place on the queue
    #tuples will be (perm_index, score) in order
    print(f'  got {total_improvements} improvements out of {len(map_perm_index_to_score)}')
    tuples=sorted(list(map_perm_index_to_score.items()), key=lambda x:x[1])
    count_nimp=0
    count_tot=0
    count_uphill=0
    best_found_node = node
    if options.test:
        print(f'  scored {len(map_perm_index_to_score)} nodes')
    uphill_ok = node.last_improved_level + nworse >= node.level+1
    #print(tuples)
    if not uphill_ok:
        print(f'  no uphills allowed, nworse exceeded')
    for p_index, score in tuples:
        pushit = False
        t='N/A'
        if count_tot >= beam:
            break
        if score < node.perp and count_nimp < nimp:
            pushit = True
            t='IMP   '
            count_nimp+=1
            count_tot+=1
        elif score < node.perp:
            pass
        elif count_uphill < nuphill and uphill_ok:
            t='UPHILL'
            pushit = True
            count_uphill+=1
            count_tot+=1
        else:
            break
        if pushit:
            new_words=[words[i] for i in permutations[test_order[p_index]]]
            level = node.level + 1
            last_improved_level=node.last_improved_level
            last_improved_perp = node.last_improved_perp
            if score < node.last_improved_perp:
                last_improved_level = level
                last_improved_perp = score
            new_node = Node(new_words, perp=score, level=level,
                            last_improved_level=last_improved_level,
                            last_improved_perp=last_improved_perp)
            print(f'  adding new to heap {t} {new_node}')
            heapq.heappush(the_nodes, new_node)
            if best_found_node is None or score < best_found_node.perp:
                best_found_node = new_node
    return False, best_found_node

def gen_perturbations(node, permutations):
    print(f'taken out')
    pass
            

print(f'before constraints [{join(words)}]')
check_and_fix_constraints(words)    
text1 = ' '.join(words)
print(f'after constraints with [{text1}]')
#map words into phrases
#Set up the words/phrases and initial node
Node.set_words(words, constraint_phrases)
words = Node.get_phrases(words)
num_words = Node.problem_num_words
#make sure we did not mess anything up
#print(f'{words=}')
#print(f'{text1=}')
assert ' '.join(words) == text1
perp = get_score(text1)
#starting node
best_node = Node(words, perp=perp)
print(f'starting node {best_node.str()}')
overall_best_node = best_node
orig_start_node = best_node
#get all the prior solutions and trials
read_states()

start_time = time.time()
round=0
#generate length of each iterator
skip_first_kick=options.skip_first_kick
skip_kick=skip_first_kick
#can be e.g.  [2,10,14,16] = do not move words 2:10 and 14:16
#dur = time.time() - start_time
#Do we want to write out every improved solution, or just the best at each exploration?
a,b=3,5
c,d=4,0
if options.opt:
    a,b,c,d=options.opt
    print(f'using kopt arguments {a=} {b=} {c=} {d=}')
    p1 = enumerate_kopt(num_words, a, b)
    p2 = enumerate_kopt(num_words, c, d)
    all_permutations = np.concatenate([p1, p2], axis=0)
else:    
    p3 = enumerate_cycles(num_words, None, True)
    #all_permutations = np.concatenate([p1, p2], axis=0)
    all_permutations = p3
#unique sorts the elements!
all_permutations = np.unique(all_permutations, axis=0)
#print(f'got permutations {len(p1)}, {len(p2)} combined {len(all_permutations)}')
print(f'got permutations combined {len(all_permutations)}')
#remove permutations that alter the fixing intervals
if fixing_intervals is not None:
    #NOTE: these need to be intervals after considering mapping of words->phrases, so might
    #not be so useful
    assert len(fixing_intervals) % 2 == 0
    assert fixing_intervals == sorted(fixing_intervals)
    for start, end in zip(fixing_intervals[::2], fixing_intervals[1::2], strict=True):
        all_permutations = all_permutations[(all_permutations[:, start:end] == np.arange(start, end)).all(axis=1)]
else:
    fixing_intervals = []
##############
print(f'after removing fixing intervals {len(all_permutations)}')
if options.gen:
    gen_perturbations(best_node, all_permutations)
    sys.exit(1)
#############
#Need to be careful about single batch score and node.perp !
#We keep 2 nodes:  best_node is current best after each round, overall_best is best one ever
    
kicks_to_use=[#(0.5, double_bridge_kick),
              (0.5, lambda x: big_kick(x, k=6)),
              (0.5, subsequence_random_permutation),
              ]
print(f'using kicks:')
for p, kick in kicks_to_use:
    print(f'prob {p:8.3f}  {kick.__name__}')

#start    
#overall_best_node is overall best node
#best_node is best node this round
while options.time is None or time.time()-start_time < options.time*3600:
    print(f'=====Starting round {round} {best_node}')
    if options.rstart:
        if len(all_solutions_this_round) > 1:
            print(f'picking a random solution from last round to start out of {len(all_solutions_this_round)}')
            best_node = random.choice(all_solutions_this_round)
            print(f'choose {best_node}')
    if not skip_kick:
        if options.rand:
            new_node = shuffle_word(best_node)
        else:
            new_node = kick_word(best_node,
                                 [],
                                 kicks_to_use)
        new_node.perp = get_score(new_node.text)
        new_node.last_improved_perp=new_node.perp
    else:
        new_node = best_node
    best_node = new_node  #really the best this round, not counting the pre-kick one
    skip_kick = False
    the_nodes=[new_node]
    print(f'starting round with {new_node}')
    heapq.heapify(the_nodes)  #don't need this but just for clarity
    #figure out how long we should do each test - time, score after a certain time, etc.
    time_for_round=options.rtime  #will be in hours limit for this round
    round_start=time.time()
    explore_count=0
    #reset everything for starting a new round
    all_solutions_this_round.clear()
    all_trials={}   #text we have computed this round
    explored={}     #text we we have explored this round
    #START A ROUND
    nodes_this_round=0
    while len(the_nodes) > 0 and (time_for_round is None or time.time()-round_start < time_for_round*3600):
        best_perp = best_node.perp
        print(f'==iter {nodes_this_round} best {best_perp:9.3f} H{best_node.hash} Q{len(the_nodes)} X{len(explored)} calcs {len(all_trials)}')
        node = heapq.heappop(the_nodes)
        print(f'popped {node}')
        if node.text in explored:
            print(f'  explored already')
            continue
        explored[node.text]=1
        if node.level >= options.maxd:
            print(f'  node at max depth, skipping explore')
            continue
        term, new_node = explore_node(node,
                                      all_permutations,
                                      nimp=options.nimp,
                                      nuphill=options.nuphill,
                                      beam=options.beam,
                                      nworse = options.nworse)
        if new_node.perp < best_node.perp:
            best_node = new_node
            write_soln(best_node, round=round, iter=nodes_this_round, end=False)
            all_solutions_this_round[new_node.text]=new_node.perp
        if best_node.perp < overall_best_node.perp:
            overall_best_node = best_node
        if term:
            break
        nodes_this_round+=1
    t1='IMPSTART' if best_node.perp < orig_start_node.perp else 'SAMESTART'
    t2='IMPBEST' if best_node.perp < overall_best_node.perp else 'SAMEPERP'
    t3='DIFFTEXT' if best_node.text != orig_start_node.text else 'SAMETEXT'
    print(f'ROUND {t1} {t2} {t3} R{round} over found {best_node} ')
    print(f'OVERALL R{round} best {overall_best_node}')
    #write the state and also append to prior data
    write_state(all_trials, all_solutions_this_round, round=round)
    explore_count += 1
    sys.stdout.flush()
    round+=1
print('Done')    
print(f'BEST {best_node}')
