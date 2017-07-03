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

class Utility:
    def build_vocab(self,srcFile,vocabFile):
        token = build_dictionary(srcFile)
        save_dictionary(token,vocabFile)

if __name__ == "__main__":
    fire.Fire(Utility)
