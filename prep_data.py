import re
from sklearn.datasets import fetch_20newsgroups
import os

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

out_dir = 'sequences/'
out_filename = 'guns.txt'

def save_doc(text, fname):
    file = open(fname, 'w')
    file.write(text)
    file.close()

def clean_text(raw_text):
    #string with all ok non-alpha chars
    ok_nonalpha_chars = '1234567890.,!?;:\' '
    raw_text = raw_text.lower()
    raw_text = raw_text.replace('...', ' ')
    raw_text = raw_text.replace('\r', '\n')

    # pad all punctuation so they show up as unique words
    raw_text = re.sub('([.,!?;:])', r' \1 ', raw_text)

    # get rid of all new lines
    raw_text = ' '.join(raw_text.split())
    # get rid of all non alpha
    raw_text = ''.join(c if (c.isalpha() or c in ok_nonalpha_chars) else ' ' for c in raw_text)
    raw_text = re.sub('\s{2,}', ' ', raw_text)
    return raw_text



data = ""

### Uncomment the following if you want to use your own text ###
#filename = "input.txt"
#raw_text = open(filename).read()

# text_list = clean_text(raw_text).split()
#
# # prepare the data set of input to output pairs encoded as integers
# seq_length = 25 + 1
#
# sequences = list()
# for i in range(seq_length, len(text_list)):
#     # select sequence of tokens
#     seq = text_list[i - seq_length:i]
#     # convert into a line
#     line = ' '.join(seq)
#     # store
#     sequences.append(line)
#
# print('Total Sequences: %d' % len(sequences))
#
# # save sequences to file
# data += '\n'.join(sequences)
# data += '\n'
### STOP COMMENTING HERE

### Comment the following if you want to use your own text
# get raw data
categories = ['talk.politics.guns']
newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'), categories=categories)

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
### Stop commenting here

data = re.sub(r'\n+', '\n',data)
data = data.strip()

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

save_doc(data, out_dir + out_filename)

