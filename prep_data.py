import re
from sklearn.datasets import fetch_20newsgroups


# load ascii text and covert to lowercase,
#filename = "word-rnn/input.txt"
#raw_text = open(filename).read()
#Categories: ['alt.atheism',
# 'comp.graphics',
# 'comp.os.ms-windows.misc',
# 'comp.sys.ibm.pc.hardware',
# 'comp.sys.mac.hardware',
# 'comp.windows.x',
# 'misc.forsale',
# 'rec.autos',
# 'rec.motorcycles',
# 'rec.sport.baseball',
# 'rec.sport.hockey',
# 'sci.crypt',
# 'sci.electronics',
# 'sci.med',
# 'sci.space',
# 'soc.religion.christian',
# 'talk.politics.guns',
# 'talk.politics.mideast',
# 'talk.politics.misc',
# 'talk.religion.misc']


def save_doc(text, fname):
    file = open(fname, 'w')
    file.write(text)
    file.close()


def clean_text(raw_text):
    #string with all ok non-alpha chars
    ok_nonalpha_chars = '.,!?;: '
    raw_text = raw_text.lower()
    raw_text = raw_text.replace('...', ' ')
    raw_text = raw_text.replace('\r', '\n')

    # pad all punctuation so they show up as unique words
    raw_text = re.sub('([.,!?;:])', r' \1 ', raw_text)
    raw_text = re.sub('\s{2,}', ' ', raw_text)

    # get rid of all new lines
    raw_text = ' '.join(raw_text.split())
    # get rid of all non alpha
    raw_text = ''.join(c for c in raw_text if (c.isalpha() or c in ok_nonalpha_chars))
    return raw_text


out_filename = 'sequences/sci-space-sequences-25.txt'

# get raw data
categories = ['sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'), categories=categories)

data = ""

for v in newsgroups_train.data:
    data_string = str(v)
    # clean text then turn into a list of words
    text_list = clean_text(data_string).split()

    # prepare the data set of input to output pairs encoded as integers
    seq_length = 25 + 1

    sequences = list()
    for i in range(seq_length, len(text_list)):
        # select sequence of tokens
        seq = text_list[i - seq_length:i]
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)

    print('Total Sequences: %d' % len(sequences))

    # save sequences to file
    data += '\n'.join(sequences)
    data += '\n'


data = re.sub(r'\n+', '\n',data)
data = data.strip()
save_doc(data, out_filename)

