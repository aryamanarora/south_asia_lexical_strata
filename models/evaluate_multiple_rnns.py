import torch
import numpy as np
from scipy.stats import norm
from statsmodels.stats.weightstats import CompareMeans
from training_multiple_rnns import compute_perplexity, all_perplexities

def get_probs(input_file, rnns, phone2ix, out_filename, device):
    print(rnns)
    inp_file = open(input_file, 'r',encoding='UTF-8')
    out_file = open(out_filename,'w',encoding='UTF-8')
    data_tens = []
    as_strings = []
    for line in inp_file:
        line = line.rstrip()
        as_strings.append(line)
        line = line.split(' ')
        line = ['<s>'] + line + ['<e>']
        line_as_tensor = torch.LongTensor([phone2ix[p] for p in line])
        data_tens.append(line_as_tensor)

    num_points = len(data_tens)

    for i,word in enumerate(data_tens):
        curr_string = as_strings[i]
        res = all_perplexities(word.unsqueeze(0), rnns, device)
        out_file.write(curr_string + '\t' + '\t'.join([str(x.numpy()) for x in res]) + '\n')
    
    inp_file.close()
    out_file.close()
