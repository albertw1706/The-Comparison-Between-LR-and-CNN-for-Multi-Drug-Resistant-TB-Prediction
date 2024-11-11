# The Comparison Between Logistic Regression and Convolutional Neural Network for Multi-Drug Resistant TB Prediction

## Project Overview
Multi-drug resistant tuberculosis (MDR-TB) is a disease caused by the Mycobacterium tuberculosis that could evade at least two different first-line anti tuberculosis drugs. MDR-TB poses a significant global health challenge, particularly in middle to lower income countries where affordable and rapid diagnostic tools are crucially needed. This situation has brought the idea of leveraging whole genome sequencing and machine learning for drug resistance predictions. Utilization of Mycobacterium tuberculosis genomic data from databases and data pre-processing allows the model to be able to train on the data. The goal of this project is to develop a Convolutional Neural Network (CNN) model with an architecture from Green et al. (2022) and also a Logistic Regression (LR) model to predict drug resistance in multi-drug resistant TB. 

## Performance Evaluation Metric
The performance evaluation metric that was used for the models are accuracy, sensitivity, and specificity. The main parameter was the accuracy, followed by sensitivity. The sensitivity parameter was chosen to be the more significant parameter over specificity for the assessment because the risk of further tests is insignificant and TB itself is curable in the preclinical phase (Gupta, 2013). The parameters were calculated from the True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN) that are obtained from the confusion matrix. 

- Accuracy = $\frac{TP + TN}{TP + TN + FP + FN}$
- Sensitivity = $\frac{TP}{TP + FN}$
- Specificity = $\frac{TN}{TN + FP}$

## Data Overview
- The input for the CNN was a 4D tensor of one-hot encoded selected genes extracted from MSA files obtained from Green et al. (2022), while the input for LR was the presence of mutations in certain positions obtained from VCF files that come from preprocessing FASTQ files (using TB-Profiler from Phelan et al. (2019)). The mutations were non-phylogenetic mutations obtained from Coll et al. (2015) that are significant to cause resistance to their respective drugs.
  
- The samples used were 12,179 TB samples selected from the fact that all of them had the Isoniazid, Pyrazinamide, Ethambutol, and Rifampicin (First-Line Drugs). The sample accessions can be found in the CSV file.

## Model Performance

- CNN's performance on the test set : 

| **Parameter** | **Rifampicin** | **Isoniazid** | **Ethambutol** | **Pyrazinamide** |
|-----------|----------------|---------------|----------------|------------------|
| **Accuracy**         | 97.04% | 96.80% | 90.48% | 90.31% |
| **Sensitivity**      | 93.75% | 94.09% | 94.12% | 84.85% |
| **Specificity**      | 98.06% | 97.99% | 89.89% | 91.17% |

- LR's performance on the test set :

| **Parameter** | **Rifampicin** | **Isoniazid** | **Ethambutol** | **Pyrazinamide** |
|-----------|----------------|---------------|----------------|------------------|
| **Accuracy**         | 91.22%  | 97.04% | 91.54% | 88.34% |
| **Sensitivity**      | 95.33%  | 94.32% | 91.00% | 88.04% |
| **Specificity**      | 89.87%  | 98.32% | 91.65% | 88.39% |

- Bar chart of model's comparisons :

<img width="900" src="images/Figure accuracy.png">
<img width="900" src="images/Figure sensitivity.png">
<img width="900" src="images/Figure specificity.png">

## References : 

- Coll, F., McNerney, R., Preston, M. D., Guerra-Assunção, J. A., Warry, A., Hill-Cawthorne, G., Mallard, K., Nair, M., Miranda, A., Alves, A., Perdigão, J., Viveiros, M., Portugal, I., Hasan, Z., Hasan, R., Glynn, J. R., Martin, N., Pain, A., & Clark, T. G. (2015). Rapid determination of anti-tuberculosis drug resistance from whole-genome sequences. Genome Medicine, 7(1). https://doi.org/10.1186/s13073-015-0164-0
- Green, A. G., Yoon, C. H., Chen, M. L., Ektefaie, Y., Fina, M., Freschi, L., Gröschel, M. I., Kohane, I., Beam, A., & Farhat, M. (2022). A convolutional neural network highlights mutations relevant to antimicrobial resistance in Mycobacterium tuberculosis. Nature Communications, 13(1). https://doi.org/10.1038/s41467-022-31236-0
- Gupta, N. G. N. (2013). Accuracy, sensitivity and specificity measurement of various classification techniques on healthcare data. IOSR Journal of Computer Engineering, 11(5), 70–73. https://doi.org/10.9790/0661-1157073 
- Phelan, J. E., O’Sullivan, D. M., Machado, D., Ramos, J., Oppong, Y. E. A., Campino, S., O’Grady, J., McNerney, R., Hibberd, M. L., Viveiros, M., Huggett, J. F., & Clark, T. G. (2019). Integrating informatics tools and portable sequencing technology for rapid detection of resistance to anti-tuberculous drugs. Genome Medicine, 11(1). https://doi.org/10.1186/s13073-019-0650-x


