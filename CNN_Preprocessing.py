import numpy as np 
from Bio import SeqIO
import pickle as pkl

# SRR accession of interest
file = open("Downloaded_fix.txt", "r")
read_file = file.read()
srr_accession = read_file.split("\n")
print(len(srr_accession))

# Initialize an empty list to hold the 3D arrays for each sample
sample_arrays = []

# Path to multi-FASTA file
fasta_files = ["rpoB.fasta", "fabG_inhA.fasta", "katG.fasta", "ahpC.fasta", "pncA.fasta", "embB.fasta"] 

# Define a mapping of nucleotides to integers
nucleotide_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}

def one_hot_encode(seq):
    """
    One-hot encode a DNA sequence with a gap character.
    """
    # Convert sequence string to uppercase
    seq = seq.upper()

    # Initialize an empty numpy array of zeros with shape (length of sequence, 5)
    one_hot = np.zeros((len(seq), 5))

    # Encode each nucleotide in the sequence
    for i, nucleotide in enumerate(seq):
        if nucleotide in nucleotide_dict:
            one_hot[i, nucleotide_dict[nucleotide]] = 1
        else:
            one_hot[i, 4] = 1 # use gap character if nucleotide not found in dictionary

    return one_hot

data = {}

for accession in srr_accession:
    sequences = []
    encoded_sequence = []
    sequence = ""
    for fasta_file in fasta_files:
        with open(fasta_file) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                if accession in record.id:
                    sequence += str(record.seq)
                    sequences.append(sequence)
                    sequence = ""
                    for encoded in sequences :
                        if "c" in encoded and len(encoded) < 3760:
                            encoded += "-"
                        while len(encoded) < 3760:
                            encoded += "-"
                            if len(encoded) == 3760:
                                break
                        print (sequences)
                        if len(sequences) == 6:
                            one_encoded = one_hot_encode(encoded)   
                            encoded_sequence.append(one_encoded)   
                            print (encoded_sequence)
                            sample_array = np.stack(encoded_sequence)
                            print (sample_array)
                            sample_arrays.append(sample_array)
                        else :
                            pass
print (sample_arrays)

selected_size = (6, 3760, 5)

filter_samples = np.where([samples.shape == selected_size for samples in sample_arrays])[0]
filtered_list = [sample_arrays[i] for i in filter_samples]
print (filtered_list)

sample_4d_array = np.stack(filtered_list)
sample_4d_array_shape = np.stack(filtered_list).shape

print (sample_4d_array)
print (sample_4d_array_shape)

reshaped_4d_array = sample_4d_array.transpose(0, 3, 2, 1).reshape((12180, 5, 3760, 6))
reshaped_4d_array_shape = sample_4d_array.transpose(0, 3, 2, 1).reshape((12180, 5, 3760, 6)).shape
print (reshaped_4d_array)
print (reshaped_4d_array_shape)

with open('CNN_input.pickle', 'wb') as f:
    pkl.dump(reshaped_4d_array, f)

with open('CNN_input.pickle', 'rb') as f:
    loaded_data = pkl.load(f)

print(np.array_equal(reshaped_4d_array, loaded_data))
