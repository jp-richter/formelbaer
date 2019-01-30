import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

# model:      LSTM, RNN_TANH, RNN_RELU, GRU
# emsize:     size of word embeddings
# nhid:       number of hidden units per layer
# nlayers:    number of layers
# lr:         initial learning rate
# clip:       gradient clipping
# epochs:     upper epoch limit
# batch_size: batch size
# bptt:       sequence length
# dropout:    dropout applied to layers (0 = no dropout)
# tied:       tie the word embedding and the softmax weights
# seed:       random seed
# cuda:       gpu support

MODEL        = 'LSTM'
EMSIZE       = 0
NHID         = 1000
NLAYERS      = 10
LR           = 0.01
CLIP         = 0.5
EPOCHS       = 10
BATCH_SIZE   = 32
BPTT         = 16
DROPOUT      = 0.5
TIED         = False
SEED         = 0
CUDA         = False
DATAPATH     = ""
LOG_INTERVAL = 100
SAVE         = "best epoch save path"         

# Set the random seed manually for reproducibility.
torch.manual_seed(SEED)

device = torch.device("cuda" if CUDA else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(DATAPATH)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data  = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data  = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data      = batchify(corpus.train, BATCH_SIZE)
val_data        = batchify(corpus.valid, eval_batch_size)
test_data       = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens   = len(corpus.dictionary)
model     = model.RNNModel(MODEL, ntokens, EMSIZE, NHID, NLAYERS, DROPOUT, TIED).to(device)
criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length BPTT.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data    = source[i:i+seq_len]
    target  = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0
    ntokens    = len(corpus.dictionary)
    hidden     = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)

    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()

    total_loss = 0.
    start_time = time.time()
    ntokens    = len(corpus.dictionary)
    hidden     = model.init_hidden(BATCH_SIZE)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, BPTT)):

        data, targets = get_batch(train_data, i)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.

        hidden         = repackage_hidden(hidden)
        model.zero_grad()

        output, hidden = model(data, hidden)
        loss           = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % LOG_INTERVAL == 0 and batch > 0:

            cur_loss = total_loss / args.log_interval
            elapsed  = time.time() - start_time

            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))

            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):

    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))

    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden      = model.init_hidden(batch_size)

    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, EPOCHS+1):
        epoch_start_time = time.time()

        train()

        val_loss = evaluate(val_data)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            LR /= 4.0

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(SAVE, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)

print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=BPTT)