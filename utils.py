
def build_encode_decode(chars):
    stoi = {c:i for i,c in enumerate(chars)}
    itos = {i:c for i,c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: [itos[i] for i in l]
    return encode, decode

def print_banner(string):
    print ('#' * (len(string) + 8))
    print (f'### {string} ###')
    print ('#' * (len(string) + 8))
