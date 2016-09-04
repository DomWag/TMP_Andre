from nltk import treetransforms
from nltk.corpus import treebank
from nltk.tree import *
import nltk
from nltk.grammar import *
import numpy

treebank.ensure_loaded()

# building the grammar and test set
tbank_productions = treebank.parsed_sents()
grammar_used = tbank_productions[:int(len(tbank_productions) * 0.8)]

# normalize the c structures
for t in grammar_used:
    t.chomsky_normal_form()
tbank_productions2 = list(treebank.sents())
test_part = tbank_productions2[int(len(tbank_productions) * 0.8):]


# prodcutions
productions = []
for t in grammar_used:
    productions += Tree.productions(t)

# induce PCFG
S = nltk.Nonterminal("S")
grammar = nltk.induce_pcfg(S, productions)
prod = grammar.productions()

#helping function to get the probability of a production
def findProb(lhsa, rhsa, prod):
    for p in prod:
        if p.lhs() == lhsa and p.rhs() == rhsa:
            return (p.prob())


def CKY(words, grammar):
    nonterms = set()
    for g in grammar.productions():
        nonterms.add(g.lhs())
    triples =[]
    lenwords = len(words)
    nonterms = list(nonterms)
    nontermsc = len(nonterms)
    score = numpy.zeros((lenwords+1,lenwords+1,nontermsc))
    back = numpy.zeros((lenwords+1,lenwords+1,nontermsc))
    for i in range(0, lenwords):
        for A in nonterms:
            pp = grammar.productions(lhs=A, rhs=words[i])
            if len(pp)>0 :

                score[i][i + 1][nonterms.index(A)] = pp[0].prob()

        # handle unaries
        added = True
        while added:
            added = False
            for A in nonterms:
                for B in nonterms:

                    pp = grammar.productions(lhs=A, rhs=B)
                    if score[i][i + 1][nonterms.index(B)] > 0 and len(pp) >0:
                        prob = pp[0].prob() * score[i][i + 1][nonterms.index(B)]

                        if prob > score[i][i + 1][nonterms.index(A)]:
                            score[i][i + 1][nonterms.index(A)] = prob
                            back[i][i + 1][nonterms.index(A)] = nonterms.index(B)
                            added = True
    for span in range(2, lenwords):
        for begin in range(0, lenwords - span):
            end = begin + span
            for splint in range(begin + 1, end - 1):
                for A in nonterms:
                    for B in nonterms:
                        for C in nonterms:
                            val = str(B)+" "+str(C)
                            valII = Nonterminal(val)
                            pp = grammar.productions(lhs=A, rhs=valII)
                            if len(pp)>0:
                                print("find something")
                                prob = score[begin][splint][nonterms.index(B)] * score[splint][end][nonterms.index(C)] * pp[0].prob()
                                if prob > score[begin][end][nonterms.index(A)]:
                                    score[begin][end][nonterms.index(A)] = prob
                                    tripleV = (splint, B, C)
                                    triples.append(tripleV)
                                    back[begin][end][nonterms.index(A)] = triples.index(tripleV)+len(nonterms)
                        # handle unaries
            added = True
            while added:
                added = False
                for A in nonterms:
                    for B in nonterms:
                        pp = grammar.productions(lhs=A, rhs=B)
                        if len(pp) > 0:
                            prob = pp[0].prob() * score[begin][end][nonterms.index(B)];
                            if prob > score[begin][end][nonterms.index(A)]:
                                score[begin][end][nonterms.index(A)] = prob
                                back[begin][end][nonterms.index(A)] = nonterms.index(B)
                                added = True
    return buildTree(score, back, nonterms, triples)

def buildTree(score, back, nonterms, tripples):
    probs = 0
    start_element = 0

    for i in range(0, len(score[0][len(score)])):
        s = Nonterminal(u'S')
        if back[0][len(score)][i] == nonterms.index(s):
            probs = (score[0][len(score)][i])
            if i >= len(nonterms):
                start_element = tripples[i - len(nonterms)]
            else:
                start_element = nonterms[i]


    return probs

exits = False
while(not exits):
        s = raw_input("What do you want to do?")
        ss = s.split(" ")
        if ss[0] == "cky_parser":
            if len(ss) == 2:
                print("You should enter at least one argument")
            elif ss[1] =="-p":
                print(CKY(ss[2], grammar))
            elif len(ss) == 4:
                print("No visualization yet implemented")
                print(CKY(ss[3], grammar))
            elif ss[1] =="-v":
                print("No visualization yet implemented")
        elif ss[0] == "evaluate":
            print("Not implemented")
        elif ss[0] == "exit":
            exits = True
        else:
            print("You did something wrong.")

#print(CKY(test_part[1], grammar))
#print(grammar.productions()[0].prob())
#print(len(grammar.productions()))

