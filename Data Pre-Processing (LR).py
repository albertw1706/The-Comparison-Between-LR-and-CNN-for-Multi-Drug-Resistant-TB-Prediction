#VCF dataframe
from pdbio.vcfdataframe import VcfDataFrame
import pandas as pd
import os

# Read the csv file filled with the feature columns
DF = pd.read_csv("LR_DataFrame.csv")
print (DF)

# Create an empty list
samples = []

# Create a list of desired SRR accessions
file = open("Downloaded_fix.txt", "r")
read_file = file.read()
lists = read_file.split("\n")
print(lists)
string = '.targets.csq.vcf.gz'
new_list = [x + string for x in lists]
print (len(new_list))

# Create a new DataFrame
new = pd.DataFrame()

# Convert the feature columns to a list
f = DF.columns.to_numpy()
fs = f.tolist()
del fs[0]

# Detect the presence of the features in each VCF files
for i in new_list :    
    vcf_path = str(dir) + i
    vcfdf = VcfDataFrame(path=vcf_path)
    print (vcfdf)
    words = vcf_path.replace(".targets.csq.vcf.gz","")
    ERR = words.replace(str(dir),"")
    samples.append(ERR)

    vcfdf.sort()   
    dfdf = vcfdf.df.astype({'POS':'str'})
    print(dfdf)
    print (vcfdf.df [["POS","REF", "ALT"]])

    new["sample"] = dfdf["POS"] + " " + dfdf["REF"] + "/" + dfdf["ALT"]

    print(DF.columns)

    g = new["sample"].to_numpy()
    gs = g.tolist()

    for x in fs :   
        if x in gs:
            samples.append(1) 
        else : 
            samples.append(0)
    DF.loc[len(DF)] = samples
    samples.clear()
print (DF)

# Save the DataFrame into a CSV file
DF.to_csv('Model.csv')
