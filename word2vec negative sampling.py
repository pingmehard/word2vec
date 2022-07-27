import sys
import ast
import numpy as np
import random


def parse_array(s):
    return np.array(ast.literal_eval(s))

def read_array():
    return parse_array(sys.stdin.readline())

def write_array(arr):
    print(repr(arr.tolist()))

def define_random_int(text, window_step, window_size, vocab_text_disjunction):
    
    a = list(text[window_step + (window_size - 1) // 2 + 1 :])
    a.extend(vocab_text_disjunction)

    return a[np.random.randint(0,len(a))]


def generate_w2v_sgns_samples(text, window_size, vocab_size, ns_rate):
    """
    text - list of integer numbers - ids of tokens in text
    window_size - odd integer - width of window
    vocab_size - positive integer - number of tokens in vocabulary
    ns_rate - positive integer - number of negative tokens to sample per one positive sample

    returns list of training samples (CenterWord, CtxWord, Label)
    """
    tokens_list = []
    text_set = set(text)
    vocab_set = set(range(vocab_size))
    text_len = len(text)
    window_step_range = [int(i) for i in list((np.array(list(range(window_size))) - np.array(list(range(window_size)))[::-1]) / 2)]

    # difference between vocab and text vocab
    vocab_text_disjunction = list(vocab_set.difference(text_set))

    for i in range(text_len):
        for j in window_step_range:
            if i + j >= 0 and i + j <= text_len - 1 and j != 0:

                tokens_list.append([text[i], text[i + j], 1])
                print([text[i], text[i + j], 1])
                for r in range(ns_rate):
                    
                    tokens_list.append([text[i], define_random_int(text, i, window_size, vocab_text_disjunction), 0])
                    print([text[i], define_random_int(text, i, window_size, vocab_text_disjunction), 0])
        print('next step')

    return tokens_list


text = read_array()
window_size = int(sys.stdin.readline().strip())
vocab_size = int(sys.stdin.readline().strip())
ns_rate = int(sys.stdin.readline().strip())

result = generate_w2v_sgns_samples(text, window_size, vocab_size, ns_rate)

write_array(np.array(result))