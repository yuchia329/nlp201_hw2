import json
import math
import functools
from collections import Counter


def loadDataAppendStartStop(file):
    with open(file, 'r') as f:
        output = f.read()
    lines = output.split("\n")
    lines = lines[:-1]
    lines = ["<START> " + line + " <STOP>" for line in lines]
    return lines


def writeFile(file_path, content, JSON=False):
    with open(file_path, 'w') as f:
        json.dump(content, f) if JSON else f.write(content)


def train(filename):
    file = f'A2-Data/1b_benchmark.{filename}.tokens'
    lines = loadDataAppendStartStop(file)
    unigrams = []
    for line in lines:
        tokens = line.split(" ")
        unigrams.extend(tokens)

    unigram_freq = Counter(unigrams)
    tokenFreq = {}
    for token, count in unigram_freq.items():
        tokenFreq[token] = count
    tokenFreq = convertInfreqnetTokens(tokenFreq)

    bigrams = []
    trigrams = []
    for line in lines:
        tokens = line.split(" ")
        tokens = [token if tokenFreq.get(
            token, None) else '<UNK>' for token in tokens]
        bigrams.extend([(tokens[i], tokens[i+1])
                       for i in range(len(tokens)-1)])
        trigrams.extend([(tokens[i], tokens[i+1], tokens[i+2])
                        for i in range((len(tokens)-2))])

    bigram_freq = Counter(bigrams)
    trigram_freq = Counter(trigrams)

    unigram_probs = {}
    tokenCounts = sum(tokenFreq.values()) - tokenFreq['<START>']
    for token in tokenFreq.keys():
        unigram_probs[token] = tokenFreq[token] / tokenCounts
    unigram_probs['<START>'] = 1

    bigram_freq_JSON = {}
    bigram_probs = {}
    for (w1, w2), count in bigram_freq.items():
        if bigram_probs.get(w1, None) is None:
            bigram_probs[w1] = {}
            bigram_freq_JSON[w1] = {}
        bigram_probs[w1][w2] = count / tokenFreq.get(w1, tokenFreq['<START>'])
        bigram_freq_JSON[w1][w2] = count

    trigram_probs = {}
    for (w1, w2, w3), count in trigram_freq.items():
        if trigram_probs.get(w1, None) is None:
            trigram_probs[w1] = {}
        if trigram_probs[w1].get(w2, None) is None:
            trigram_probs[w1][w2] = {}
        trigram_probs[w1][w2][w3] = count / bigram_freq_JSON[w1][w2]

    # writeFile(f'perplex/unigram_{filename}_token_freq.json',
    #           unigram_probs, JSON=True)
    # writeFile(f'perplex/bigram_{filename}_token_freq.json',
    #           bigram_probs, JSON=True)
    # writeFile(f'perplex/trigram_{filename}_token_freq.json',
    #           trigram_probs, JSON=True)
    return unigram_probs, bigram_probs, trigram_probs


def convertInfreqnetTokens(tokenFreq):
    tokenFreq['<UNK>'] = 0
    removeKeys = []
    for key in tokenFreq.keys():
        if tokenFreq[key] < 3:
            tokenFreq['<UNK>'] += tokenFreq[key]
            removeKeys.append(key)
    [tokenFreq.pop(key) for key in removeKeys]
    print(len(tokenFreq.keys()))
    return tokenFreq


def evalUnigramSentence(tokens, unigram_probs):
    sum_log_prob = functools.reduce(
        lambda a, b: a + math.log2(unigram_probs.get(b, unigram_probs.get('<UNK>'))), tokens, 0)
    prob = 2 ** (-sum_log_prob/(len(tokens)-1))
    if " ".join(tokens) == '<START> HDTV . <STOP>':
        print(prob)
    return str(sum_log_prob)


def evalBigramSentence(tokens, unigram_probs, bigram_probs):
    sum_log_prob = 0
    pre_token = ''
    for index, token in enumerate(tokens):
        if index == 0:
            token_prob = unigram_probs.get(token)
            sum_log_prob += math.log2(token_prob)
            pre_token = token
        else:
            pre_token_prob = bigram_probs.get(
                pre_token, bigram_probs.get('<UNK>'))
            cur_token_prob = pre_token_prob.get(
                token, pre_token_prob.get('<UNK>'))
            if cur_token_prob is None:
                sum_log_prob = "INF"
                break
            else:
                sum_log_prob += math.log2(cur_token_prob)
            pre_token = token
    prob = 2 ** (-sum_log_prob/(len(tokens)-1)
                 ) if sum_log_prob != 'INF' else 'INF'
    if " ".join(tokens) == '<START> HDTV . <STOP>':
        print(prob)
    return str(prob)


def evalTrigramSentence(tokens, unigram_probs, bigram_probs, trigram_probs):
    sum_log_prob = 0
    pre_token = ''
    pre_pre_token = ''
    for index, token in enumerate(tokens):
        if index == 0:
            token_prob = unigram_probs.get(token)
            sum_log_prob += math.log2(token_prob)
            pre_token = token
        elif index == 1:
            pre_token_prob = bigram_probs.get(
                pre_token, bigram_probs.get('<UNK>'))
            cur_token_prob = pre_token_prob.get(
                token, pre_token_prob.get('<UNK>'))
            if cur_token_prob is None:
                sum_log_prob = "INF"
                break
            else:
                sum_log_prob += math.log2(cur_token_prob)
            pre_pre_token = pre_token
            pre_token = token
        else:
            pre_pre_token_prob = trigram_probs.get(
                pre_pre_token, trigram_probs.get('<UNK>'))
            pre_token_prob = pre_pre_token_prob.get(
                pre_token, pre_pre_token_prob.get('<UNK>'))
            if pre_token_prob is None:
                sum_log_prob = "INF"
                break
            cur_token_prob = pre_token_prob.get(
                token, pre_token_prob.get('<UNK>'))
            if cur_token_prob is None:
                sum_log_prob = "INF"
                break
            else:
                sum_log_prob += math.log2(cur_token_prob)
            pre_pre_token = pre_token
            pre_token = token
    prob = 2 ** (-sum_log_prob/(len(tokens)-1)
                 ) if sum_log_prob != 'INF' else 'INF'
    if " ".join(tokens) == '<START> HDTV . <STOP>':
        print(prob)
    return str(prob)


def eval(unigram_probs, bigram_probs, trigram_probs, weight=[1, 0, 0]):
    filenames = ['train', 'dev', 'test']
    for filename in filenames:
        file = f'A2-Data/1b_benchmark.{filename}.tokens'
        lines = loadDataAppendStartStop(file)
        unigram_score = []
        bigram_score = []
        trigram_score = []
        for line in lines:
            tokens = line.split(" ")

            # unigram
            prob = evalUnigramSentence(tokens, unigram_probs)
            unigram_score.append(prob)

            # bigram
            prob = evalBigramSentence(tokens, unigram_probs, bigram_probs)
            bigram_score.append(prob)

            # trigram
            prob = evalTrigramSentence(
                tokens, unigram_probs, bigram_probs, trigram_probs)
            trigram_score.append(prob)

        unigram_score_str = "\n".join(unigram_score)
        bigram_score_str = "\n".join(bigram_score)
        trigram_score_str = "\n".join(trigram_score)
        writeFile(f'perplex/unigram_{filename}_score.txt',
                  unigram_score_str, False)
        writeFile(f'perplex/bigram_{filename}_score.txt',
                  bigram_score_str, False)
        writeFile(f'perplex/trigram_{filename}_score.txt',
                  trigram_score_str, False)


def smoothing():
    prob = {
        "unigram": {},
        "bigram": {},
        "trigram": {},
    }
    for key in prob.keys():
        filename = f'{key}_train_token_freq.json'
        with open(filename, 'r') as f:
            output = json.load(f)
            prob[key] = output

    weights = [[0.3, 0.3, 0.4]]


def main():
    filename = 'train'
    unigram_probs, bigram_probs, trigram_probs = train(filename)
    eval(unigram_probs, bigram_probs, trigram_probs)


if __name__ == '__main__':
    main()
