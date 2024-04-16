import nltk
import math
from collections import Counter, namedtuple

Score = namedtuple('SARI', 'SARI, F_keep, P_del, F_add, D_SARI, D_keep, D_del, D_add')

def f1(precision, recall):
    if precision > 0 and recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0

def keep(sgramcounter_rep, cgramcounter_rep, rgramcounter):
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keeptmpscore1 = 0
    keeptmpscore2 = 0
    for keepgram in keepgramcountergood_rep:
        keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        keeptmpscore2 += keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]

    keepscore_precision = 0
    if len(keepgramcounter_rep) > 0:
        keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)

    keepscore_recall = 0
    if len(keepgramcounterall_rep) > 0:
        keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)

    return f1(keepscore_precision, keepscore_recall)

def delete(sgramcounter_rep, cgramcounter_rep, rgramcounter):
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    delgramcounterall_rep = sgramcounter_rep - rgramcounter

    deltmpscore1 = 0
    deltmpscore2 = 0
    for delgram in delgramcountergood_rep:
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
        deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]

    delscore_precision = 0
    if len(delgramcounter_rep) > 0:
        delscore_precision = deltmpscore1 / len(delgramcounter_rep)

    # delscore_recall = 0
    # if len(delgramcounterall_rep) > 0:
    #     delscore_recall = deltmpscore1 / len(delgramcounterall_rep)

    return delscore_precision # instead of f1(delscore_precision, delscore_recall)

def add(sgramcounter, cgramcounter, rgramcounter):

    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)

    addtmpscore = len(addgramcountergood)

    addscore_precision = 0
    if len(addgramcounter) > 0:
        addscore_precision = addtmpscore / len(addgramcounter)

    addscore_recall = 0
    if len(addgramcounterall) > 0:
        addscore_recall = addtmpscore / len(addgramcounterall)

    return f1(addscore_precision, addscore_recall)

def D_SARIngram(sgrams, cgrams, rgramslist, numref):

    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter()
    for sgram, scount in sgramcounter.items():
        sgramcounter_rep[sgram] = scount * numref

    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter()
    for cgram, ccount in cgramcounter.items():
        cgramcounter_rep[cgram] = ccount * numref

    rgramcounter = Counter(rgram for rgrams in rgramslist for rgram in rgrams)

    keepscore  =   keep(sgramcounter_rep, cgramcounter_rep, rgramcounter)
    delscore   = delete(sgramcounter_rep, cgramcounter_rep, rgramcounter)
    addscore   =    add(sgramcounter,     cgramcounter,     rgramcounter)

    return (keepscore, delscore, addscore)

def count_length(ssent, csent, rsents):
    input_length  = len(ssent.split(" "))
    output_length = len(csent.split(" "))
    reference_length = sum(len(rsent.split(" ")) for rsent in rsents) // len(rsents)
    return input_length, reference_length, output_length

def sentence_number(csent, rsents):
    c_sentence_number = len(nltk.sent_tokenize(csent))
    r_sentence_number = sum(len(nltk.sent_tokenize(rsent)) for rsent in rsents) // len(rsents)
    return c_sentence_number, r_sentence_number

def make_ngram(unigrams, *ngrams):
    n_token = len(unigrams)
    n_gram  = len(ngrams)
    for i in range(n_token - 1):
        for j in range(n_gram):
            k = i + j + 2
            if k <= n_token:
                ngrams[j].append(tuple(unigrams[i:k]))

def D_SARIsent(ssent, csent, rsents):

    r1gramslist = []
    r2gramslist = []
    r3gramslist = []
    r4gramslist = []

    for rsent in rsents:

        r1grams = rsent.lower().split(" ")
        r2grams = []
        r3grams = []
        r4grams = []
        make_ngram(r1grams, r2grams, r3grams, r4grams)
        r1gramslist.append(r1grams)
        r2gramslist.append(r2grams)
        r3gramslist.append(r3grams)
        r4gramslist.append(r4grams)


    s1grams = ssent.lower().split(" ")
    s2grams = []
    s3grams = []
    s4grams = []
    c1grams = csent.lower().split(" ")
    c2grams = []
    c3grams = []
    c4grams = []
    make_ngram(c1grams, c2grams, c3grams, c4grams)
    make_ngram(s1grams, s2grams, s3grams, s4grams)

    numref = len(rsents)
    (keep1score, del1score, add1score) = D_SARIngram(s1grams, c1grams, r1gramslist, numref)
    (keep2score, del2score, add2score) = D_SARIngram(s2grams, c2grams, r2gramslist, numref)
    (keep3score, del3score, add3score) = D_SARIngram(s3grams, c3grams, r3gramslist, numref)
    (keep4score, del4score, add4score) = D_SARIngram(s4grams, c4grams, r4gramslist, numref)

    avgkeepscore = sum([keep1score, keep2score, keep3score, keep4score]) / 4
    avgdelscore = sum([del1score, del2score, del3score, del4score]) / 4
    avgaddscore = sum([add1score, add2score, add3score, add4score]) / 4
    # D-SARI
    finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3

    # SARI
    input_length, reference_length, output_length = count_length(ssent, csent, rsents)
    output_sentence_number, reference_sentence_number = sentence_number(csent, rsents)
    
    if output_length >= reference_length:
        LP_1 = 1
    else:
        LP_1 = math.exp((output_length - reference_length) / output_length)

    if output_length > reference_length:
        LP_2 = math.exp((reference_length - output_length) / max(input_length - reference_length, 1))
    else:
        LP_2 = 1

    SLP = math.exp(-abs(reference_sentence_number - output_sentence_number) / max(reference_sentence_number,
                                                                                  output_sentence_number))
    d_avgkeepscore = avgkeepscore * LP_2 * SLP
    d_avgaddscore = avgaddscore * LP_1
    d_avgdelscore = avgdelscore * LP_2
    d_finalscore = (d_avgkeepscore + d_avgdelscore + d_avgaddscore) / 3

    return Score(finalscore, avgkeepscore, avgdelscore, avgaddscore, d_finalscore, d_avgkeepscore, d_avgdelscore, d_avgaddscore)

def main():

    ssent = "marengo is a town in and the county seat of iowa county , iowa , united states . it has served as the county seat since august 1845 , even though it was not incorporated until july 1859 . the population was 2,528 in the 2010 census , a decline from 2,535 in 2000 ."

    csent1 = "in the US . 2,528 in 2010 ."
    csent2 = "marengo is a city in iowa , the US . it has served as the county seat since august 1845 , even though it was not incorporated . the population was 2,528 in the 2010 census , a decline from 2,535 in 2010 ."
    csent3 = "marengo is a town in iowa . marengo is a town in the US . in the US . the population was 2,528 . the population in the 2010 census ."
    csent4 = "marengo is a town in iowa , united states . in 2010 , the population was 2,528 ."
    rsents = ["marengo is a city in iowa in the US . the population was 2,528 in 2010 ."]
    
    print(D_SARIsent(ssent, csent1, rsents))
    print(D_SARIsent(ssent, csent2, rsents))
    print(D_SARIsent(ssent, csent3, rsents))
    print(D_SARIsent(ssent, csent4, rsents))
    
if __name__ == '__main__':
    main()