#checks word start is upper or not
import csv
import re

#from geotext.geotext import GeoText
from nltk.corpus import names


def feature_Cap(word):
    if word[0][0].isupper():
        return 1
    else:
        return 0

#checks if word is name or location
def feature_nameList(word):
    if word[0] in names.words('male.txt'):
        return 1
    elif word[0] in names.words('female.txt'):
        return 1
    #elif len(GeoText(word[0]).cities)>0 or len(GeoText(word[0]).countries)>0:
     #   return 1
    else:
        return 0


def get_lex(words):
    lex = set()
    for line in words:
        if len(line.split())>1:
            splitted = line.split()
            lex.add(splitted[0])
    return list(lex)
lexical = get_lex(open("Dutch/ned.train").readlines())
def features_lexical(words, lexicall):
    len_lex = len(lexicall)
    score = [0]*len_lex

    if str(words.split(" ")[0]) in lexicall:
        ind = lexicall.index(words.split(" ")[0])
        score[ind] = 1
    return score


#prefix and suffix
def feature_prefix(word):
    return word[0:2]


def feature_suffix(word):
    return word[-2:]

def suf_fixes(words):
    suffixes = set()
    for line in words:
        if len(line.split(" "))>1:
            suffixes.add(feature_suffix(line.split(" ")[0]))
    suffixes = list(suffixes)
    return suffixes


def pre_fixes(words):
    prefixes = set()
    for line in words:
        if len(line.split(" "))>1:
            prefixes.add(feature_prefix(line.split(" ")[0]))
    prefixes = list(prefixes)
    return prefixes
pre_fix = pre_fixes(open("Dutch/ned.train").readlines())
def feature_prefixxx(words, prefixs):
    len_pre = len(prefixs)
    score = [0] * len_pre
    if len(words.split(" "))>1:
        if words.split(" ")[0][0:2] in prefixs:
            ind = prefixs.index(words.split(" ")[0][0:2])
            score[ind] = 1
    return score

suf_fix = suf_fixes(open("Dutch/ned.train").readlines())
def feature_suffixxx(words, suffixs):
    len_suf = len(suffixs)
    score = [0] * len_suf
    if len(words.split(" "))>1:
        if words.split(" ")[0][-2:] in suffixs:
            ind = suffixs.index(words.split(" ")[0][-2:])
            score[ind] = 1
    return score

#class of previous word
def get_classes(words):
    classes = set()
    for line in words:
        if len(line.split(" "))>1:
            splitted = line.split(" ")
            classes.add(splitted[0])
    return list(classes)
classes = get_classes(open("Dutch/ned.train").readlines())
prevClas = []
def feature_class(words):
    len_classes = len(classes)
    score = [0] * len_classes

    if len(prevClas) >0:
        ind = classes.index(prevClas[len(prevClas)-1])
        score[ind] = 1
        prevClas.append(words.split(" ")[1])
    return score

def feature_AllCap(word):
    if word[0].isupper():
        return 1
    else:
        return 0
def feature_containDigits(word):
    if any(i.isdigit() for i in word[0]):
        return 1
    else:
        return 0
def feature_hyphened(word):
    hyphenated = re.findall(r'\w+-\w+[-\w+]*', word[0])
    if len(hyphenated)>1:
        return 1
    else:
        return 0


def scoring(fileT):


    input = open(fileT).readlines()
    prefixs = pre_fixes(input[1:])
    suffxs = suf_fixes(input[1:])
    lexical = get_lex(input[1:])

    writer = csv.writer(open("scoring.csv", 'a'))

    for line in input[1:]:
        if len(line.split(" ")) > 1:
            print(line)
            score = []
            scoreCap = feature_Cap(line)
            score.append(int(scoreCap))
            scoreList = feature_nameList(line)
            score.append(scoreList)
            scoreAllCap = feature_AllCap(line)
            score.append(scoreAllCap)
            scoreDig = feature_containDigits(line)
            score.append(scoreDig)
            scorehyp = feature_hyphened(line)
            score.append(scorehyp)
            #scoreLex = features_lexical(line, lexical)
            #score.append(scoreLex)
            #scoreSuf = feature_suffixxx(line, suffxs)
            #score.append(scoreSuf)
            #scorePre = feature_prefixxx(line, prefixs)
            #score.append(scorePre)
            #scoreClas = feature_class(line)
            #score.append(scoreClas)
            score.append(line.split()[2])
            writer.writerow(score)
    prevClas = []
scoring("Dutch/ned.testa")

