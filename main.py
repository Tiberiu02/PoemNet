from os import listdir
import codecs

EOS = '#'
dataset = sorted([open("dataset/" + n, "r").read().decode('utf-8') for n in listdir("dataset")], key=len)
vocab = sorted(list(set(sum([list(poem) for poem in dataset], []))) + [EOS])

vocab_size = len(vocab)
print "dataset size: %d\nvocabulary size: %d" % (sum(len(poem) for poem in dataset), vocab_size)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import BatchNorm1d
import math

use_cuda, reset_model = True, True
hidden_size, num_layers, bsize = 512, 1, 20520

batches = [[]]
for b in dataset:
	if (len(batches[-1]) + 1) * len(b) < bsize:
		batches[-1].append(b)
	else:
		batches.append([b])
print "Batch sizes: " + str([len(b) for b in batches])

print "\n"

def to_tensors(batch):
	slen = len(max(batch, key=len))
	i = [torch.zeros(len(batch), vocab_size) for k in range(slen)]
	t = [torch.zeros(len(batch)).type(torch.LongTensor) for k in range(slen)]
	for s_ix in range(len(batch)):
		s = batch[s_ix]
		s += EOS * (slen - len(s) + 1)
		for ch_ix in range(slen):
			i[ch_ix][s_ix, vocab.index(s[ch_ix])] = 1.0
			t[ch_ix][s_ix] = vocab.index(s[ch_ix + 1])
	return i, t

class MGUCell(nn.Module):
	def __init__(self, input_size, hidden_size, dropout=0.25):
		super(MGUCell, self).__init__()

		self.Wx = Parameter(torch.Tensor(input_size, hidden_size * 2))
		self.Wh = Parameter(torch.Tensor(hidden_size, hidden_size * 2))
		self.b = Parameter(torch.Tensor(hidden_size * 2))
		self.a = Parameter(torch.Tensor(hidden_size * 2))
		
		self.dropout = dropout

		self.bn_x = []
		self.bn_h = []

		self.reset_parameters()
		self.reset_state()
	
	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.Wx.size(0) + self.Wh.size(0))
		self.Wx.data.uniform_(-stdv, stdv)
		self.Wh.data.uniform_(-stdv, stdv)
		self.b.data.fill_(0.0)
		self.a.data.fill_(0.1)

	def reset_state(self):
		self.h = None
		self.time_step = 0

	def forward(self, x):
		if self.h is None:
			self.h = Variable(torch.zeros(x.size(0), self.b.size(0) / 2).type_as(self.b.data), requires_grad=False)
			if self.dropout > 0:
				self.dp = Variable((torch.rand(x.size(0), self.b.size(0) / 2) >= self.dropout).type_as(self.b.data), requires_grad=False)

		if len(self.bn_x) <= self.time_step and self.time_step < 250 and self.training:
			self.bn_x.append(BatchNorm1d(self.b.size(0), affine=False).cuda() if self.b.data.type() == 'torch.cuda.FloatTensor' else BatchNorm1d(self.b.size(0), affine=False))
			self.bn_h.append(BatchNorm1d(self.b.size(0), affine=False).cuda() if self.b.data.type() == 'torch.cuda.FloatTensor' else BatchNorm1d(self.b.size(0), affine=False))
			self.add_module('bnx_' + str(self.time_step), self.bn_x[-1])
			self.add_module('bnh_' + str(self.time_step), self.bn_h[-1])
		

		g = self.bn_x[min(49, self.time_step)](torch.mm(x, self.Wx)) + self.bn_h[min(49, self.time_step)](torch.mm(self.h, self.Wh))
		g = g * self.a.expand(x.size(0), self.b.size(0))
		g = g + self.b.expand(x.size(0), self.b.size(0))
		#g = torch.mm(x, self.Wx) + torch.mm(self.h, self.Wh) + self.b.expand(x.size(0), self.b.size(0))
		f = F.sigmoid(g[:, :self.b.size(0) / 2])
		z = F.tanh(g[:, self.b.size(0) / 2:])
		self.h = self.h * f + z * (1.0 - f)
		if self.training and self.dropout > 0:
			self.h = self.h * self.dp
		self.time_step += 1
		
		return self.h
		

class LM(nn.Module):
	def __init__(self, vocab_size, hidden_size):
		super(LM, self).__init__()

		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.l1 = MGUCell(vocab_size, hidden_size)
		self.l2 = MGUCell(hidden_size, hidden_size)
		self.l3 = MGUCell(hidden_size, hidden_size)
		self.l4 = MGUCell(hidden_size, hidden_size)
		self.l5 = nn.Linear(hidden_size, vocab_size)

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()
		self.l3.reset_state()
		self.l4.reset_state()

	def transfer_state(self, m2):
		self.l1.h = m2.l1.h
		self.l2.h = m2.l2.h
		self.l3.h = m2.l3.h
		self.l4.h = m2.l4.h

		self.l1.dp = m2.l1.dp
		self.l2.dp = m2.l2.dp
		self.l3.dp = m2.l3.dp
		self.l4.dp = m2.l4.dp

	def forward(self, x):
		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = self.l4(x)
		x = self.l5(x)
		x = F.softmax(x)

		return x

def sample(model, seq_len=100, init_text="Doi copii"):
	model.reset_state()
	model.eval()
	for ch in init_text:
		x = torch.zeros(1, vocab_size).cuda() if use_cuda else torch.zeros(1, vocab_size)
		x[0, vocab.index(ch)] = 1.0
		y = model(Variable(x))
	for ix in range(len(init_text), seq_len):
		ch = 0
		for ix in range(1, vocab_size):
			if y.data[0, ix] > y.data[0, ch]:
				ch = ix
		init_text = init_text + vocab[ch]
		y = model(y)
	return init_text

model = LM(vocab_size, hidden_size) if reset_model else torch.load("model")
#decoder = LM(vocab_size, hidden_size) if reset_model else torch.load("decoder")

if use_cuda: model.cuda()

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(size_average=False)

for e in range(1, 1001):
	for ix in range(len(batches)):
		model.train()
		model.reset_state()

		loss = 0.0
		for x, y in zip(*to_tensors(batches[ix])):
			if use_cuda:
				x, y = x.cuda(), y.cuda()
			x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)
			y_pred = model(x)
		
			loss += loss_fn(y_pred, y)

		print "epoch=%f, loss=%f, poems_in_batch=%d" % (e + 1.0 * ix / len(batches), loss.data[0] / bsize / vocab_size, len(batches[ix]))
		#print sample(model)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	if e % 10 == 0:
		torch.save(model, "model")
		
