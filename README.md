# TSAI_END2.0-Assignment-8

### Assignment Instructions - 

Refer to this Repo [here](https://github.com/bentrevett/pytorch-seq2seq). 

* You are going to refactor this repo in the next 3 sessions. In the current assignment, change the 2 and 3 (optional 4, 500 additional points) such that
is uses none of the legacy stuff
* It MUST use Multi30k dataset from torchtext
* uses yield_token, and other code that we wrote
* Once done, proceed to answer questions in the Assignment-Submission Page. 

## Modern Way of Training NLP Models Using Pytorch

### Loading the Dataset -

PyTorch provides two data primitives:  `torch.utils.data.DataLoader` and `torch.utils.data.Dataset` that allows us to use preloaded datasets as well as your own data. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

**But What are pytorch Datasets?**

PyTorch Datasets are just things that have a length and are indexable so that len(dataset) will work and dataset[index] will return a tuple of (x,y).

Pytorch’s data sets have "dunder/magic methods" `__getitem__` (for `dataset[index]` functionality) and `__len__` (for `len(dataset)` functionality).

We can also create our own custom dataset class, our dataset class must have implementation of __getitem__ and __len__ to work with map-style datasets.

Let's write a custom dataset class to load our question/answer dataset - 

```python
class CustomTextDataset(Dataset):

  "Custom Dataset class to load QnA data"

  def __init__(self,src, tgt,path):

      self.df = pd.read_csv(path,sep='\t',encoding='iso-8859-1')   # read the data
      self.df = self.df[['Question','Answer']].dropna().reset_index(drop=True)   # drop the null values from the dataset

      self.src = self.df[src]   # source column or Questions
      self.tgt = self.df[tgt]   # target column or Answers

  def __len__(self):
      return len(self.tgt)   # return the length of dataset

  def __getitem__(self, idx):  # returns a dictionary of questions and answers
      src = self.src[idx]
      tgt = self.tgt[idx]
      sample = {"SRC":src,"TGT" :tgt}   
      return sample

data = CustomTextDataset('Question','Answer','/content/full_question_answer_data.txt')
next(iter(data))
```

```
output : {'SRC': 'Was Abraham Lincoln the sixteenth President of the United States?',
          'TGT': 'yes'}
```
### DataLoader

This is main vehicle to help us to sample data from our data source, with my limited understanding, these are the key points:
* Manage multi-process fetching
* Sample data from dataset as small batches
* transform data with collate_fn()
* pin memory (for GPU memory performance)

**How does DataLoader sample data?**

High level idea is, it check what style of dataset (iterator / map) and iterate through calling __iter__() (for iterator style dataset) or sample a set of index and query the __getitem__() (for map style dataset).

Then we just wrap the data in a DataLoader and we can iterate it but now they're magically tensors and we can use DataLoaders handy configurations like shuffling, batching, multi-processing, etc.:

For Instance on our QnA dataset if we use DataLoader to load the dataset let's see what is the output 
```python
list(DataLoader(data))[:11]  # only looking at first 11 examples
```
```
####################################################################################

[{'SRC': ['Was Abraham Lincoln the sixteenth President of the United States?'],
  'TGT': ['yes']},
 {'SRC': ['Was Abraham Lincoln the sixteenth President of the United States?'],
  'TGT': ['Yes.']},
 {'SRC': ['Did Lincoln sign the National Banking Act of 1863?'],
  'TGT': ['yes']},
 {'SRC': ['Did Lincoln sign the National Banking Act of 1863?'],
  'TGT': ['Yes.']},
 {'SRC': ['Did his mother die of pneumonia?'], 'TGT': ['no']},
 {'SRC': ['Did his mother die of pneumonia?'], 'TGT': ['No.']},
 {'SRC': ["How many long was Lincoln's formal education?"],
  'TGT': ['18 months']},
 {'SRC': ["How many long was Lincoln's formal education?"],
  'TGT': ['18 months.']},
 {'SRC': ['When did Lincoln begin his political career?'], 'TGT': ['1832']},
 {'SRC': ['When did Lincoln begin his political career?'], 'TGT': ['1832.']},
 {'SRC': ['What did The Legal Tender Act of 1862 establish?'],
  'TGT': ['the United States Note, the first paper currency in United States history']}]
 
```

DataLoaders can also load the data in batches let's try to load the data with batch size of 2.

```python
bat_size = 2
DL_DS = DataLoader(data, batch_size=bat_size)

# loop through each batch in the DataLoader object
for (idx,batch) in enumerate(DL_DS):

    # Print the 'SRC' data of the batch
     print(idx, 'SRC data: ', batch['SRC'], '\n')

    # Print the 'TGT'  data of batch
     print(idx, 'TGT data: ', batch['TGT'], '\n')

     if idx == 5:
       break
```

```
0 SRC data:  ['Was Abraham Lincoln the sixteenth President of the United States?', 'Was Abraham Lincoln the sixteenth President of the United States?'] 

0 TGT data:  ['yes', 'Yes.'] 

1 SRC data:  ['Did Lincoln sign the National Banking Act of 1863?', 'Did Lincoln sign the National Banking Act of 1863?'] 

1 TGT data:  ['yes', 'Yes.'] 

2 SRC data:  ['Did his mother die of pneumonia?', 'Did his mother die of pneumonia?'] 

2 TGT data:  ['no', 'No.'] 

3 SRC data:  ["How many long was Lincoln's formal education?", "How many long was Lincoln's formal education?"] 

3 TGT data:  ['18 months', '18 months.'] 

4 SRC data:  ['When did Lincoln begin his political career?', 'When did Lincoln begin his political career?'] 

4 TGT data:  ['1832', '1832.'] 

5 SRC data:  ['What did The Legal Tender Act of 1862 establish?', 'What did The Legal Tender Act of 1862 establish?'] 

5 TGT data:  ['the United States Note, the first paper currency in United States history', 'The United States Note, the first paper currency in United States history.'] 
```

In machine learning or deep learning text needs to be cleaned and turned in to vectors prior to training. DataLoader has a handy parameter called collate_fn. This parameter allows you to create separate data processing functions and will apply the processing within that function to the data before it is output.

### Preprocessing the dataset - 

**Tokenization**

Tokenization is a common task in Natural Language Processing (NLP), Tokenization is a way of separating a piece of text into smaller units called tokens. Here, tokens can be either words, characters, or subwords.

Let's look at an example - 
```python
import torchtext
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer("You can now install TorchText using pip!")
tokens
#['you', 'can', 'now', 'install', 'torchtext', 'using', 'pip', '!']
```

**Building Vocab**

The set of unique words used in the text corpus is referred to as the vocabulary. When processing raw text for NLP, everything is done around the vocabulary.

The first step is to build a vocabulary with the raw training dataset. Here we use built in factory function `build_vocab_from_iterator` which accepts iterator that yield list or iterator of tokens. Users can also pass any special symbols to be added to the vocabulary.

we can use `build_vocab_from_iterator` from `torchtext.vocab` to build our vocab, the first argument to this is an iterator it must yield list or iterator of tokens.

Let's use demonstrate this with an example on AG_NEWS dataset.

```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS

tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

vocab(['hello','we','are','languagecorpus'])

# Output - [12544, 507, 42, 0]
```

### Collate function - 

What are they ? 🙄😖

They are our friend not foe 😁

Let's understand what actually collate function does -

This is where transform of data take place, normally one does not need to bother with this because there is a default implementation that work for straight forward dataset like list.

You can read the default implementation here [here](https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py)

Let's have a look at what collate function does - 

**Example - 1**
```python
item_list = [1,2,3,4,5]
default_collate(item_list)

# output - tensor([1, 2, 3, 4, 5])
```

Here the collate function has converted our inputs into tensors.

**Example - 2**
```python
item_list = ([1,2,3,4,5],[6,7,8,9,10])  #item
default_collate(item_list)

# output - [tensor([1, 6]),
            tensor([2, 7]),
            tensor([3, 8]),
            tensor([4, 9]),
            tensor([ 5, 10])]
```
For example 2, the batch is a tuple of 2 lists, and it return a list of tensor, which each tensor get 1 item from each list in original tuple.


Example - 3
```
item_list = [(1,2),(3,4),(5,6),(7,8)]
default_collate(item_list)

# output - [tensor([1, 3, 5, 7]), tensor([2, 4, 6, 8])]

```

Example - 4
```
item_list = [[1,2,3],[3,4,5],[5,6,7],[7,8,9]]
default_collate(item_list)

# output - [tensor([1, 3, 5, 7]), tensor([2, 4, 6, 8]), tensor([3, 5, 7, 9])]
```

For Example 3 and 4, the input look like typical data form that have multiple attributes. Consider case 4, if 3rd element per record is the label and first 2 elements are input data attributes, the return list of tensors is not directly usable by the model, in which the preferable return could be:

`[tensor([[1,2], [3,4], [5,6], [7,8]]), tensor([3,5,7,9])]`

The first example “collating along a dimension other than the first”, our interpretation is when we want the batch data being grouped differently compare to default collate function.

```
Preferred: 
[tensor([[1,2], [3,4], [5,6], [7,8]]), tensor([3,5,7,9])]
v.s.
Default:
[tensor([1,3,5,7]), tensor([2,4,6,8]), tensor([3,5,7,9])]

```

Let's Write our own collate function for that - 
```python
import torch
def our_own_collate(data):

 # from [[x1,x2,y],[...]] to [tensor([[x1,x2],.....]) tensor([y,.....])] 

 xs = [[data_item[0],data_item[1]] for data_item in data]
 y = [data_item[2] for data_item in data]

 return torch.tensor(xs),torch.tensor(y)

item_list = [[1,2,3],[3,4,5],[5,6,7],[7,8,9]]
our_own_collate(item_list)

```
```
# output
(tensor([[1, 2],
         [3, 4],
         [5, 6],
         [7, 8]]), tensor([3, 5, 7, 9]))
```

In our case working with textual data or in NLP we do padding of sequences, one of the use case is RNN/LSTM model for NLP. For a batch of sentence, when we sample randomly, we would get batches of sentence with different length, and because we are performing batch operation, we would need to pad the shorter sequences to the longest one. One option is to pad to a pre-defined maximum length, it should be the case for Transformer models.

Collate function used in our case is - 
```python
######################################################################
# Collation
# ---------
#   
# As seen in the ``Data Sourcing and Processing`` section, our data iterator yields a pair of raw strings. 
# We need to convert these string pairs into the batched tensors that can be processed by our ``Seq2Seq`` network 
# defined previously. Below we define our collate function that convert batch of raw strings into batch tensors that
# can be fed directly into our model.   
#


from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch
```


## Refrences - 
* https://pytorch.org/text/stable/index.html
* https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
* https://pytorch.org/docs/stable/data.html#map-style-datasets
* https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
* https://medium.com/geekculture/pytorch-datasets-dataloader-samplers-and-the-collat-fn-bbfc7c527cf1
