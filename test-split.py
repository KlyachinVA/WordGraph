import re

def test():
    txt = "Первое, предложение; конец. Второе: предложение? Третье -- предложение!"
    rg = re.compile(r"[.?!]")
    sents = rg.split(txt)
    print(sents)
    rx = re.compile(r"[-:;,]")
    txt2 = rx.sub("",txt)
    print(txt2)


    sentences = []
    for sent in sents:
        sent = rx.sub("", sent)
        ws = sent.split()
        sentences.append(ws)

    print(sentences)


if __name__ == "__main__":
    test()