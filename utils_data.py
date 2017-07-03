import fire

def build_dictionary(srcFile=""):
    dictA = {}
    fp = open(srcFile,"r").readlines()
    for line in fp:
        tokens = line.strip().split(" ")
        for token in tokens:
            if dictA.has_key(token):
                dictA[token] += 1
            else:
                dictA[token] = 1
    TokenList = ["BOS","EOS","UNK"] + dictA.keys()
    return TokenList

def save_dictionary(TokenList,filename):
    fo = open(filename,"w")
    for i,k in enumerate(TokenList):
        fo.writelines(str(i) + "\t" + k + "\n")
    fo.close()

def load_dictionary(srcVocab):
    fp = open(srcVocab,"r")
    vocabD = {}
    line = fp.readline()
    while line:
        item = line.strip().split("\t")
        vocabD[item[1]] = int(item[0])
        line = fp.readline()
    return vocabD

def load_corpus(srcFile,trgFile,srcVocab,trgVocab):
    srcVocabD = load_dictionary(srcVocab)
    trgVocabD = load_dictionary(trgVocab)
    
    Fsrc = open(srcFile,"r")
    line = Fsrc.readline()
    srcL = []
    while line:
        item = line.strip().split()
        item = ["BOS"] + item + ["EOS"]
        ids = [srcVocabD[x] for x in item]
        line = Fsrc.readline()
        srcL += [ids]

    Ftrg = open(trgFile,"r")
    line = Ftrg.readline()
    trgL = []
    while line:
        item = line.strip().split()
        item = ["BOS"] + item + ["EOS"]
        ids = [trgVocabD[x] for x in item]
        line = Ftrg.readline()
        trgL += [ids]

    return srcL,trgL

class Utility:
    def build_vocab(self,srcFile,vocabFile):
        token = build_dictionary(srcFile)
        save_dictionary(token,vocabFile)

if __name__ == "__main__":
    fire.Fire(Utility)
