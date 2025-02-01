#2nd place solution (@solverworld's part)

This was a very enjoyable Santa competition.  I must thank Kaggle for putting on the
competition and my two teammates @danielphalen and @zaburo for the tremendous team work
and ingenuity.  I learned a lot from them and from the other Kagglers offering to share their
insights and comments.

Our teammate @zaburo found the final best solution to id3 and id5, so his post will 
discuss the solution to those 2 problems.  

For many of the problems, we shuffled solutions back and forth between our various programs.
Because we used different algorithms from each other, when one program got stuck
in a local minima, it seemed like another one could improve the solution a little bit
(by finding a newer minima close by) or a lot (by jumping to a completely different
solution).

I will explain a lot of things, a few that worked, and (a lot more) that didn't work.

First, it is important to understand what is a local minima.  In the classical 
concept of minimizing a function f(x), you say you have a local minima at x0 when f only
increases when you move away from x0 a small amount (a small perturbation).  There is the usual epsilon-delta
type of argumnet, which I won't bore you with.  The problem in the Sanata case is "what
is a small perturbation"?  If you say it is swapping pairs of sequential words, you get one 
sort of neighborhood for a local minima, but if it is a k-opt swap, you get a different neighborhood.
In order for any kind of local search to work, you require some concept of smoothness
in f, that is, f(x) doesn't change that much when you are in a neighborhood of x.  So
you want to find neighborhoods that result in a "smooth" perplexity function with
regard to that neighborhood.

We spent some time trying differnt neighborhood methods:

* swapping consecutive words
* swapping any 2 words
* cycler - rotating a subsequence left or right, or equivalently deleting a word and inserting it
  elsewhere
* 3-opt:  ABCD=>ACBD for 4 subsequences (possibly null) ABCD; equivalently pick 3 points (the 3-opt) 
          that define the boundaries of AB, BC, CD
* 4-opt:  ABCDE=>ADCBE for 5 subsequences
  
@danielphalen is the one who theorized that a cycler move would disrupt perplexity less
that other moves.
Here is a plot that shows that <deltacompare.png>.  The distance is the number of 
perturbations applied to a starting point, which was one of our early submissions to problem
id1.  The 3 perturbations types are swap (any 2 words), cycler, and 3-opt.
You can see that the cycler tended to have less variation than swap.  So you would
think that cyler would be a better way to search for nearby solutions that did 
not perturb the perplexity that much.  3-opt might be better for searching things farther
away.  And swap is probably the worst because it gave answers unrelated to the 
original perplexity.

I will now describe various algorithms that I used.  As I mentioned before, we
originally found answers to id1,2 and 4 through passing solutions back between 
algorithms, but I believe that several of the programs I am about to describe will
find the solutions to those problems on their own quite readily.

The algorithms all have the same basic framework.
First my definitions:
A Node an object representing a particular solution (words in order).
A Priority Queue is data structure that allows you to insert Nodes and you can 
remove them in order of lowest cost (perplexity).

Add some starting node to the queue
while True:
  node A <- best (lowest) cost node off the queue
  Generate nodes in some neighborhood of A, calculating their perplexity, and putting 
  them back into the queue.
  keep track of the best score and words achieved so far
  stop when you get tired
  
There are several decisions to make in this algorithm:
1. The starting node.  You can choose the initial words given in the problem,
   a random shuffle of same, or a "kick", which is a perturbation bigger than 
   any neighborhood you have been using but smaller than a random shuffle.
2. Neighborhood generation: is it cycler, 3-opt, swaps, etc.
2. When you generate nodes, do you 
  (a) generate all nodes in a neighborhood, (b) stop at the first one that improves
      the score, or (c) something in between
3. How many of the nodes generated do you keep?  (a) all of them, or (b) the best k
4. What if there are no improved nodes?  Do you place any nodes that are worse on the
   queue or not?
5. Do you discard nodes that show no improvement from a best ancestor after some
   number of tries?
6. When do you give up and pick a new starting point?

As an example, I ran program hills_cleaned.py id2 on a 4090 with a 15 minute limit
for each ROUND with the following parameters:
1. look for 2 improvements at each neighborhood.  This was a compromise between 
   not spending too much time searching in the beginning, when many
   perturbations are an improvement, and allowing for a slight better one
   to be found when they are scarce.
2. Place up to 2 nodes on the queue at each point.  If there are not 2 improved nodes,
   use up to 1 worse node.
3. If a particular node has not improved over its best ancestor in 5 moves, discard it.   
4. Stop each ROUND after 15 min.
5. Kicks are either a 5 word random subsequence shuffle or a 6-opt kick 
   (meaning take 6 random points in the sequence and shuffle the resulting subsequences)
This program running for 24 hours on id2 will find the 298.9 solution several times.
One speed up the program used was storing low scoring solutions and their scores 
in a local pickled file.  Then I could check if I had previously calculated a
score before.  This saved a little time when restarting a search near a previous one.
You can see the file <kjc-350-hil-3_graph.png> for the type of progress the program makes
on id3.  It sadly does not find the solution to id3.  You need to see my teammates @kaburo
solution for that.

Another issue we ran into was the difference between scores on our GPUs and the
Kaggle environment with T4s.  To guard against misfortune, we took our best 
solutions and all their neigboring perturbations, and analyzed them in a Kaggle
notebook (with competition metric) to ensure that no nearby solutions were better.
We did find a few word sequences that had inverted polarity on T4 vs 4090, but they
were not close to the best ones found.


Things we tried that did not work but seemed like they might at the time
1. Branch-and-bound algorithms.  Once the perplexity of some quite good solutions
   became known on the discussion board but were better than our best solutions, this
   approach became possible.  Note that the nature of perplexity is that it is
   the sum of losses for each word, divided by the number of tokens, with an exp()
   thrown in at the end.  Ignoring the exp and assuming the number of tokens is fixed (more on that
   later), it becomes an additive loss problem, where the words after word k do
   not affect the losses of the first k words.  So we could perform a tree search,
   where each node is a partial solution (first N words).  We know the losses for 
   the first N words, as the words coming later do not affect them.  If we can estimate
   the loss for future words in some way that is optimistic, when can then
   prune away any searches that result in losses higher than our best known perplexity.
   One 100x speedup comes from realizing that you can calculate the loss added
   by the next word from the logits output by the Gemma LLM, without doing
   any more LLM evaluations.  In the case of id5, that is 100x savings.
   Alas, this estimate is far too optimistic; it estimates the losses for all
   the remaining words as just the loss for the next word.  So the search space 
   does not get pruned enough, and the search time is not practial.
2. Generating and scoring 14 million random solutions to id5.  I originally thought
   this would be useful at some point, but it turned out not to be.  It also
   did not find any good solutions.  I think the best score it found was 500 (vs the best
   of 28.5). :(
3. Trying to use the structure of the perplexity calculation to help with 
   heuristics for searching.  In TSP algorithms, you have the ability to assess how
   much swapping links could change things, so I thought that maybe the logistics
   arrays could help.  While you can predict how much the loss will be changed
   at ONE point by moving other words there, the other links being changed cannot
   be estimated.  Those logistic losses did not seem to be helpful in predicting
   perplexity changes.
   
   For amusements sake there is a long printout (losses_best5.txt) showing the losses for 
   each word in the best known solution for id5, along with the predicted next
   word for each slot.  It also shows what Gemma thinks is the most 
   likely word from all words (tokens), not just ones we are allowed to use.
4. We tried freezing phrases when moving subsequences around, but it never worked.
   For example, the prior printout showed that magi is a high loss word and that
   maybe it could be put in a better place.  "gifts of the magi" is one of the
   best scoring 4 word phrases, so I tried locking that in place while searching.
   It turns out the best known solution for id3 starts with "magi is of the grinch"


