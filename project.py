from nltk import treetransforms
from nltk.corpus import treebank
from nltk.tree import *
import nltk
from nltk.grammar import *
import numpy

treebank.ensure_loaded()

# building the grammar and test set
#TODO here we reduced the tbank productions
tbank_productions = treebank.parsed_sents()[0:8]
grammar_used = tbank_productions[:int(len(tbank_productions) * 0.8)]

# normalize the c structures
for t in grammar_used:
    #treetransforms.chomsky_normal_form(t)
    t.chomsky_normal_form()
#print(round(len(tbank_productions) * 0.8))
tbank_productions2 = list(treebank.sents()[0:8])
test_part = tbank_productions2[int(len(tbank_productions) * 0.8):]
# normalize the c structures
#for tt in test_part:
    #treetransforms.chomsky_normal_form(tt)
#    tt.chomsky_normal_form()


# prodcutions
productions = []
for t in grammar_used:
    productions += Tree.productions(t)

# induce PCFG
S = nltk.Nonterminal("S")
grammar = nltk.induce_pcfg(S, productions)
#TODO here the grammar is not in chomsky normal form
#print(grammar.is_chomsky_normal_form())
prod = grammar.productions()


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
    #print(lenwords)
    score = numpy.zeros((lenwords+1,lenwords+1,nontermsc))
    back = numpy.zeros((lenwords+1,lenwords+1,nontermsc))
    for i in range(0, lenwords):
        for A in nonterms:
            pp = grammar.productions(lhs=A, rhs=words[i])
            if len(pp)>0 :

                score[i][i + 1][nonterms.index(A)] = pp[0].prob()
                #print(pp[0].prob())

        # handle unaries
        #print("uni")
        added = True
        while added:
            added = False
            for A in nonterms:
                for B in nonterms:

                    pp = grammar.productions(lhs=A, rhs=B)
                    if score[i][i + 1][nonterms.index(B)] > 0 and len(pp) >0:
                        # if score[i][i+1][B] > 0 && A->B in grammar
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
                #for A, B, C in nonterms:
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
                                    back[begin][end][nonterms.index(A)] = triples.index(tripleV)
                        # handle unaries
            added = True
            while added:
                added = False
                for A in nonterms:
                    for B in nonterms:
                # for A, B in nonterms:
                        pp = grammar.productions(lhs=A, rhs=B)
                        if len(pp) > 0:
                            prob = pp[0].prob() * score[begin][end][nonterms.index(B)];
                            if prob > score[begin][end][nonterms.index(A)]:
                                score[begin][end][nonterms.index(A)] = prob
                                back[begin][end][nonterms.index(A)] = nonterms.index(B)
                                added = True
    return buildTree(score, back)

#def buildTree(score, back):
#    print(score)
 #   print(back)
print(len(test_part[1]))
print(test_part[1])
print(CKY(test_part[1], grammar))
#findProb('NNS', 'appliances', prod)
# CYK(grammar, test_part[1])
print(grammar.productions()[0].prob())
print(len(grammar.productions()))
#print(grammar)
# print(len(productions))
# print(len(tbank_productions))
# print(len(grammar_used))
# print(len(test_part))
