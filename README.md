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

```class CustomTextDataset(Dataset):

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

For Instance on our QnA dataset if we use DataLoader to load the dataset let's see what is the output 
```
list(DataLoader(data))[:11]

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

```
bat_size = 2
DL_DS = DataLoader(data, batch_size=bat_size)

# loop through each batch in the DataLoader object
for (idx,batch) in enumerate(DL_DS):

    # Print the 'text' data of the batch
     print(idx, 'SRC data: ', batch['SRC'], '\n')

    # Print the 'class' data of batch
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
```
import torchtext
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer("You can now install TorchText using pip!")
tokens
#['you', 'can', 'now', 'install', 'torchtext', 'using', 'pip', '!']
```

**Building Vocab**

The set of unique words used in the text corpus is referred to as the vocabulary. When processing raw text for NLP, everything is done around the vocabulary.

```
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

# vocab(['hello','we','are','languagecorpus'])
```
