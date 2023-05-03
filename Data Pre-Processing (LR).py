#VCF dataframe
from pdbio.vcfdataframe import VcfDataFrame
import pandas as pd
import os

DF = pd.read_csv("LR_DataFrame.csv")
print (DF)

a = []

dir = os.environ['VCF']

file = open("Downloaded_fix.txt", "r")
read_file = file.read()
lists = read_file.split("\n")
print(lists)
string = '.targets.csq.vcf.gz'
new_list = [x + string for x in lists]
print (new_list)

new = pd.DataFrame()

f = DF.columns.to_numpy()
fs = f.tolist()
del fs[0]
print (fs)

for i in new_list :    
    vcf_path = str(dir) + i
    vcfdf = VcfDataFrame(path=vcf_path)
    print (vcfdf)
    words = vcf_path.replace(".targets.csq.vcf.gz","")
    ERR = words.replace(str(dir),"")
    a.append(ERR)

    vcfdf.sort()   
    dfdf = vcfdf.df.astype({'POS':'str'})
    print(dfdf)
    print (vcfdf.df [["POS","REF", "ALT"]])

    new["sample"] = dfdf["POS"] + " " + dfdf["REF"] + "/" + dfdf["ALT"]

    print(DF.columns)

    g = new["sample"].to_numpy()
    gs = g.tolist()
    print (gs)

    for x in fs :   
        if x in gs:
            a.append(1) 
        else : 
            a.append(0)
    DF.loc[len(DF)] = a
    print (a)
    a.clear()
print (DF)

DF.to_csv('Model.csv')
