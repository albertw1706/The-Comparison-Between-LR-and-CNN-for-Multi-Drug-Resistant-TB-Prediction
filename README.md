# The Comparison Between LR and CNN for Multi-Drug Resistant TB Prediction


Reproduced a Convolutional Neural Network model with an architecture from Green et al. (2022) and also created a Logistic Regression model to predict drug resistance in multi-drug resistant TB. 

-----------------

The input for the CNN was a 4D tensor of one-hot encoded selected genes extracted from MSA files obtained from Green et al. (2022), while the input for LR was VCF files from preprocessed FASTQ files (using TB-Profiler from Phelan et al. (2019)).

-----------------

The samples used were 12,179 TB samples selected from the fact that all of them had the Isoniazid, Pyrazinamide, Ethambutol, and Rifampicin (First-Line Drugs). The sample accessions can be found in the CSV file.

-----------------

References : 

- Green, A. G., Yoon, C. H., Chen, M. L., Ektefaie, Y., Fina, M., Freschi, L., Gröschel, M. I., Kohane, I., Beam, A., & Farhat, M. (2022). A convolutional neural network highlights mutations relevant to antimicrobial resistance in Mycobacterium tuberculosis. Nature Communications, 13(1). https://doi.org/10.1038/s41467-022-31236-0 [Link](https://www.nature.com/articles/s41467-022-31236-0)
- Phelan, J. E., O’Sullivan, D. M., Machado, D., Ramos, J., Oppong, Y. E. A., Campino, S., O’Grady, J., McNerney, R., Hibberd, M. L., Viveiros, M., Huggett, J. F., & Clark, T. G. (2019). Integrating informatics tools and portable sequencing technology for rapid detection of resistance to anti-tuberculous drugs. Genome Medicine, 11(1). https://doi.org/10.1186/s13073-019-0650-x [Link](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-019-0650-x)
