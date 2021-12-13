import argparse
import logging
import pickle
from os import mkdir
from os.path import exists, join

from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Iterable, TypeVar
import regex
import nltk

T = TypeVar('T')

def flatten_list(iterable_of_lists: Iterable[Iterable[T]]) -> List[T]:
  """Unpack lists into a single list."""
  return [x for sublist in iterable_of_lists for x in sublist]

# class Tokenizer(Configured):
#   def tokenize(self, text: str) -> List[str]:
#     raise NotImplementedError()

#   def tokenize_with_inverse(self, text: str) -> Tuple[List[str], np.ndarray]:
#     """Tokenize the text, and return start/end character mapping of each token within `text`"""
#     raise NotImplementedError()


_double_quote_re = regex.compile(u"\"|``|''")


def convert_to_spans(raw_text: str, text: List[str]) -> np.ndarray:
  """ Convert a tokenized version of `raw_text` into a series character
  spans referencing the `raw_text` """
  cur_idx = 0
  all_spans = np.zeros((len(text), 2), dtype=np.int32)
  for i, token in enumerate(text):
    if _double_quote_re.match(token):
      span = _double_quote_re.search(raw_text[cur_idx:])
      tmp = cur_idx + span.start()
      l = span.end() - span.start()
    else:
      tmp = raw_text.find(token, cur_idx)
      l = len(token)

    if tmp < cur_idx:
      raise ValueError(token)
    cur_idx = tmp
    all_spans[i] = (cur_idx, cur_idx + l)
    cur_idx += l
  return all_spans


class NltkAndPunctTokenizer():
  """Tokenize ntlk, but additionally split on most punctuations symbols"""

  def __init__(self, split_dash=True, split_single_quote=False, split_period=False, split_comma=False):
    self.split_dash = split_dash
    self.split_single_quote = split_single_quote
    self.split_period = split_period
    self.split_comma = split_comma

    # Unix character classes to split on
    resplit = r"\p{Pd}\p{Po}\p{Pe}\p{S}\p{Pc}"

    # A list of optional exceptions, will we trust nltk to split them correctly
    # unless otherwise specified by the ini arguments
    dont_split = ""
    if not split_dash:
      dont_split += "\-"
    if not split_single_quote:
      dont_split += "'"
    if not split_period:
      dont_split += "\."
    if not split_comma:
      dont_split += ","

    resplit = "([" + resplit + "]|'')"
    if len(dont_split) > 0:
      split_regex = r"(?![" + dont_split + "])" + resplit
    else:
      split_regex = resplit

    self.split_regex = regex.compile(split_regex)
    try:
      self.sent_tokenzier = nltk.load('tokenizers/punkt/english.pickle')
    except LookupError:
      logging.info("Downloading NLTK punkt tokenizer")
      nltk.download('punkt')
      self.sent_tokenzier = nltk.load('tokenizers/punkt/english.pickle')

    self.word_tokenizer = nltk.TreebankWordTokenizer()

  def retokenize(self, x):
    if _double_quote_re.match(x):
      # Never split isolated double quotes(TODO Just integrate this into the regex?)
      return (x, )
    return (x.strip() for x in self.split_regex.split(x) if len(x) > 0)

  def tokenize(self, text: str) -> List[str]:
    out = []
    for s in self.sent_tokenzier.tokenize(text):
      out += flatten_list(self.retokenize(w) for w in self.word_tokenizer.tokenize(s))
    return out

  def tokenize_with_inverse(self, paragraph: str):
    text = self.tokenize(paragraph)
    inv = convert_to_spans(paragraph, text)
    return text, inv


STOP_WORDS = frozenset([
  'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
  'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
  'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
  'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
  'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
  'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
  'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
  'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
  'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
  'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
  'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
  've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
  'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
   "many", "how", "de"
])


def is_subseq(needle, haystack):
  l = len(needle)
  if l > len(haystack):
    return False
  else:
    return any(haystack[i:i+l] == needle for i in range(len(haystack)-l + 1))


def build_mnli_bias_only(out_dir):
  """Builds our bias-only MNLI model and saves its predictions
  :param out_dir: Directory to save the predictions
  :param cache_examples: Cache examples to this file
  :param w2v_cache: Cache w2v features to this file
  """

  tok = NltkAndPunctTokenizer()

  def read_tsv(path, quotechar=None):
      import csv
      with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

  def _create_mnli_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    labelmap = {"contradiction":0, "entailment":1, "neutral":2}
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = line[0]
      text_a = line[8]
      text_b = line[9]
      if set_type == "test":
        label = "contradiction"
      else:
        label = line[-1]
      examples.append([guid, text_a, text_b, label])
    return examples
  def _create_qqp_examples(lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = i - 1
      if set_type.startswith("test"):
        label =  line[3]
        text_a = line[1]
        text_b = line[2]
      else:
        text_a = line[3]
        text_b = line[4] 
        label =  line[5]
      examples.append([guid, text_a, text_b, label])
    return examples

  dataset_to_examples = {}
  QQP_training = read_tsv("/home/data/QQP/train.tsv")
  dataset_to_examples["qqp_train"] = _create_qqp_examples(QQP_training, "train")
  QQP_training = read_tsv("/home/data/QQP/dev.tsv")
  dataset_to_examples["qqp_dev"] = _create_qqp_examples(QQP_training, "dev")
  # MNLI_training = read_tsv("/home/data/MNLI/dev.tsv")
  # dataset_to_examples["mnli_dev"] = _create_mnli_examples(MNLI_training, "dev")


  # Our models will only distinguish entailment vs (neutral/contradict)
  for examples in dataset_to_examples.values():
    for i, ex in enumerate(examples):
      if ex[3] == 2:
        examples[i][3] = 0
  negations =  ["not", "no", "n't", "never", "nothing", "none", "nobody", "nowhere", "neither"]

  # Build the features, store as a pandas dataset
  dataset_to_features = {}
  for name, examples in dataset_to_examples.items():
    tf.logging.info("Building features for %s.." % name)
    features = []
    for example in examples:
      h = [x.lower() for x in tok.tokenize(example[2])]
      p = [x.lower() for x in tok.tokenize(example[1])]
      p_words = set(p)
      neg_in_h = sum(x in h for x in negations)
      n_words_in_p = sum(x in p_words for x in h)
      fe = {
        "h-is-subseq": 1 if is_subseq(h, p) else 0,
        "all-in-p": 1 if n_words_in_p == len(h) else 0,
        "percent-in-p": n_words_in_p / len(h),
        "log-len-diff": np.log(max(len(p) - len(h), 1)),
        "neg-in-h": 1 if neg_in_h > 0 else 0,
        "label": example[-1],
      }

    #   h_vecs = [w2v[w] for w in example.hypothesis if w in w2v]
    #   p_vecs = [w2v[w] for w in example.premise if w in w2v]
    #   if len(h_vecs) > 0 and len(p_vecs) > 0:
    #     h_vecs = np.stack(h_vecs, 0)
    #     p_vecs = np.stack(p_vecs, 0)
    #     # [h_size, p_size]
    #     similarities = np.matmul(h_vecs, p_vecs.T)
    #     # [h_size]
    #     similarities = np.max(similarities, 1)
    #     similarities.sort()
    #     fe["average-sim"] = similarities.sum() / len(h)
    #     fe["min-similarity"] = similarities[0]
    #     if len(similarities) > 1:
    #       fe["min2-similarity"] = similarities[1]
      fe["average-sim"] = fe["min-similarity"] = fe["min2-similarity"] = 0
      features.append(fe) #for the wordemb similarity; now only use the psuedo number

    dataset_to_features[name] = pd.DataFrame(features)
    dataset_to_features[name].fillna(0.0, inplace=True)
    if "mnli" in name:
        dataset_to_features[name].to_csv("/home/data/MNLI/dev_bias_features.csv")
    else:
        dataset_to_features[name].to_csv(f"/home/data/QQP/{name}_bias_features.csv") 

#   # Train the model
#   tf.logging.info("Fitting...")
#   train_df = dataset_to_features["train"]
#   feature_cols = [x for x in train_df.columns if x != "label"]

#   # class_weight='balanced' will weight the entailemnt/non-entailment examples equally
#   # C=100 means no regularization
#   lr = LogisticRegression(multi_class="auto", solver="liblinear",
#                           class_weight='balanced', C=100)
#   lr.fit(train_df[feature_cols].values, train_df.label.values)

#   # Save the model predictions
#   if not exists(out_dir):
#     mkdir(out_dir)

#   for name, ds in dataset_to_features.items():
#     tf.logging.info("Predicting for %s" % name)
#     examples = dataset_to_examples[name]
#     pred = lr.predict_log_proba(ds[feature_cols].values).astype(np.float32)
#     y = ds.label.values

#     bias = {}
#     for i in range(len(pred)):
#       if examples[i].id in bias:
#         raise RuntimeError("non-unique IDs?")
#       bias[examples[i].id] = pred[i]

#     acc = np.mean(y == np.argmax(pred, 1))
#     print("%s two-class accuracy: %.4f (size=%d)" % (name, acc, len(examples)))

#     with open(join(out_dir, "%s.pkl" % name), "wb") as f:
#       pickle.dump(bias, f)


def main():
  parser = argparse.ArgumentParser("Train our MNLI bias-only model")
#   parser.add_argument("output_dir", help="Directory to store the bias-only predictions")
#   parser.add_argument("--cache_examples")
#   parser.add_argument("--cache_w2v_features")
#   args = parser.parse_args()

  build_mnli_bias_only("null")


if __name__ == "__main__":
  main()