#from geotext.geotext import GeoText


def end_word_extractor(document):
    tokens = document.split()
    first_word, last_word = tokens[0], tokens[-1]
    feats = {}
    feats["first({0})".format(first_word)] = True
    feats["last({0})".format(last_word)] = False
    return feats
features = end_word_extractor("I'm excited to try my new classifier.")
#def feature_nameList(word):
    #places = GeoText(word)
    #if len(places.cities)>0 or len(places.countries)>0:
     #   return 1
    #else:
     #   return 0
#print(feature_nameList("Bavaria is cool."))

def feature_prefix(word):
    return word[0:3]
#print(feature_prefix("Dominik"))

def feature_Cap(word):
    if word[0].isupper():
        return 1
    else:
        return 0

txt = open("Dutch/ned.testb").readlines()
for line in txt:
    if len(line.split())>1:
        if feature_Cap(line.split()[0]):
            #print("1"+line.split()[0])
            0
        else:
            #print(0)
            0
#print("\n" is "\n")

def get_lex(words):
    lex = set()
    for line in words:
        if len(line.split())>1:
            splitted = line.split()
            lex.add(splitted[0])
    return list(lex)
lexical = get_lex(open("Dutch/ned.train").readlines())
def features_lexical(words):
    # sentence = list()
    # sent = []
    # for line in words:
    #     if line is "\n":
    #         sentence.append(sent)
    #         sent = []
    #     else:
    #         sent.append(line.split()[0])
    # for s in sentence:
    #     if len(s)>1:
    #         for ss in s:
    len_lex = len(lexical)
    score = [0]*len_lex

    if words.split()[0] in lexical:
        ind = lexical.index(words.split()[0])
        score[ind] = 1


    return score


#for line in open("Dutch/ned.testa").readlines():
 #   if len(line.split()) > 1:
  #      print(features_lexical(line))

def get_classes(words):
    classes = set()
    for line in words:
        if len(line.split())>1:
            splitted = line.split()
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
        prevClas.append(words.split()[1])
    return score
print("Dominik".isupper())