import torch
import  sys
import  argparse


def fillCountMatrix(N: torch.tensor, words ,stoi):
    for word in words:
        temp = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(temp, temp[1:]):
            idx1 = stoi[ch1]
            idx2 = stoi[ch2]
            N[idx1, idx2] += 1
    return N

def charCount(words):
    count = {}
    for word in words:
        temp = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(temp, temp[1:]):
            char = (ch1, ch2)
            count[char] = count.get(char, 0) + 1

    return count


def train(path='names.txt'):

    try:
        words = open(path, 'r').read().splitlines()
    except FileNotFoundError:
        print("Error: The file does not exist")
        exit(-1)

    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {item[1]: item[0] for item in stoi.items()}
    # stoi is for string to integer
    # in our case stoi if for mapping
    # character to index
    # `.` is a special character for start
    # and end of a word
    # so stoi['a'] will give we the index of 'a'

    # map using stoi and itos
    # i can use N[stio['a],stoi['a']]
    # to give the ccount of `a` followed by `a`
    count = charCount(words)
    # row is first character and colum is second character
    N = torch.zeros((27, 27))
    N = fillCountMatrix(N=N, words=words,stoi=stoi)
    # Converting It to probability matrix:
    P = N / N.sum(1, keepdim=True)

    return P,itos,stoi

# Now accessing it

SEED = 42
P,itos,stoi = train()

def generateNames(num: int = 10, g: torch.Generator = torch.Generator().manual_seed(SEED)):
    for _ in range(num):
        name = []
        idx = 0
        while True:
            p = P[idx]
            idx = torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
            name.append(itos[idx])
            if idx == 0:
                break
            
        print(''.join(name[:-1]))

parser = argparse.ArgumentParser(description="Generate Random names using Bi model")
parser.add_argument('-d','--data',type=str,help='Path of own name files')
parser.add_argument('arg1', type=int, help='First positional argument')
parser.add_argument('arg2', type=int, help='Second positional argument')


if __name__=="__main__":
    args = parser.parse_args()
    N = 10
    g = torch.Generator().manual_seed(10)
    if len(sys.argv)==3:
        N = args.arg1
        arg2 = args.arg2
        g= torch.Generator().manual_seed(arg2)
    if len(sys.argv)==4:
        N = args.arg1
        arg2 = args.arg2
        g = torch.Generator().manual_seed(arg2)

    generateNames(N,g)
