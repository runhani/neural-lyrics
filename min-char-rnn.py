"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import pdb

# data I/O
# data = open('data/korean_lyrics.txt', 'r').read() # should be simple plain text file
pdb.set_trace()
data = open('data/hello.txt', 'r').read() # should be simple plain text file
# (Pdb) data[0:30] = 'hello hello hello hello hello '

chars = list(set(data))
# (Pdb) chars = ['h', 'e', 'l', 'o', ' ']

data_size, vocab_size = len(data), len(chars)
# (Pdb) data_size = 30000
# (Pdb) vocab_size = 5

print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
# (Pdb) char_to_ix = {'h': 0, 'e': 1, 'l': 2, 'o': 3, ' ': 4}

ix_to_char = { i:ch for i,ch in enumerate(chars) }
# (Pdb) ix_to_char = {0: 'h', 1: 'e', 2: 'l', 3: 'o', 4: ' '}

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

# (Pdb) Wxh.shape = (100, 5)
# (Pdb) Whh.shape = (100, 100)
# (Pdb) Why.shape = (5, 100)
# (Pdb) bh.shape = (100, 1)
# (Pdb) by.shape = (5, 1)

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    # (Pdb) xs[t].shape = (5, 1)
    # (Pdb) xs[t]
    # array([[ 1.],
    #        [ 0.],
    #        [ 0.],
    #        [ 0.],
    #        [ 0.]])


    # ht = tanh( Wxh * xt + Whh * ht-1 + bh )
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    # (Pdb) hs[t].shape = (100, 1)

    # yt = Why * ht + by
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    # (Pdb) ys[t].shape = (5, 1)

    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    # (Pdb) ps[t].shape = (5, 1)
    # array([[ 0.20000739],
    #        [ 0.19994888],
    #        [ 0.20008367],
    #        [ 0.19986411],
    #        [ 0.20009595]])

    # cross-entropy loss
    # L = - Sigma ( yk * log pk )
    # Lk = - log (pk) = - log (softmax(fk)) 
    # (Pdb) ps[t].shape = (5, 1)
    # (Pdb) targets = [1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4, 0, 1]
    # (Pdb) targets[0] = 1, targets[1] = 2, targets[2] = 2, ... ...  

    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    pdb.set_trace()
    dy = np.copy(ps[t])
    # dL / dfk = pk - 1
    dy[targets[t]] -= 1 
    # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    # also see http://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
    #####################
    # yt = Why * ht + by 
    #####################
    # dyt / dWhy = ht
    # dyt / dby = 1
    # dyt / dht = Why
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    
    # see https://theclevermachine.wordpress.com/2014/09/08/derivation-derivatives-for-common-neural-network-activation-functions/
    ################################################### 
    # ht = tanh( Wxh * xt + Whh * ht-1 + bh ) = tanh(z)
    ###################################################
    # dtanh(z) / dz = 1 - tanh(z) * tank(z)
    # dht / dbh = tanh'(z) * 1
    # dht / dWxh = tanh'(z) * xt
    # dht / dWhh = tanh'(z) * ht-1
    # dht / dht-1 = tanh'(z) * Whh

    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
  # (Pdb) data[0:25] = 'hello hello hello hello h'
  # (Pdb) data[1:26] = 'ello hello hello hello he'

  # sample from the model now and then
  if n % 100 == 0:
    pdb.set_trace()
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

   # n == 0 
   # l  h h lheloeoe lhlo e loe hehele o o hel lell oloheeheh  e oeollo  eo el h ee
   # ho lh  eoeoehlohoe lhhelee e he oeloeeh oooeollhhlee le hooehho le hlooeoeeheoe looehoeloelohho l oheehhohhee lllollh 

   # n == 100
   #   hello heolo hello hello heoloehello hello hello heloo ollo hlleo hloholholl 
   # llheloo he lo helloohello hello hello hello helll hello hello hello l llo hollllhlo o hello  helo lelloollolo hello hello h 

   # n == 300 
   #  hello hello hello hello hello hello "heloo" hello hello hello hello hello hello 
   #  hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello h

   # n == 500
   #  ello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hellolhello hello 
   # hello hello hello hello hello hello hello hello hello hello hello hello hello hel
 

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 