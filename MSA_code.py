from Bio import SeqIO, AlignIO
from Bio.Align import MultipleSeqAlignment
import os

# Define the region of interest
rpoB_start = 759609
rpoB_end = 763369

fabG_inhA_start = 1673300
fabG_inhA_end = 1675016

katG_start = 2153235
katG_end = 2156706

ahpC_start = 2726087
ahpC_end = 2726805

pncA_start = 2288681
pncA_end = 2289282

embB_start = 4246514
embB_end = 4249878

file = open("Downloaded_fix.txt", "r")
read_file = file.read()
srr_accession = read_file.split("\n")
print(len(srr_accession))

import os
path = "consensus"
os.chdir(path)

def MSA(start, end):
    seq_records = []
    for accession in srr_accession:
        if accession : 
            file = str(dir) + accession + ".fa"
            seq = next(SeqIO.parse(file, "fasta")).seq[start:end]
            seq_records.append(SeqIO.SeqRecord(seq, id=accession, name="", description=""))
    alignment = MultipleSeqAlignment(seq_records)
    print (seq_records)
    return alignment

import os
path = "/home/jupyter-albertw17"
os.chdir(path)

alignment = MSA(rpoB_start, rpoB_end)
AlignIO.write(alignment, "rpoB.fasta", "fasta")

alignment = MSA(fabG_inhA_start, fabG_inhA_end)
AlignIO.write(alignment, "fabG_inhA.fasta", "fasta")

alignment = MSA(katG_start, katG_end)
AlignIO.write(alignment, "katG.fasta", "fasta")

alignment = MSA(ahpC_start, ahpC_end)
AlignIO.write(alignment, "ahpC.fasta", "fasta")

alignment = MSA(pncA_start, pncA_end)
AlignIO.write(alignment, "pncA.fasta", "fasta")

alignment = MSA(embB_start, embB_end)
AlignIO.write(alignment, "embB.fasta", "fasta")
