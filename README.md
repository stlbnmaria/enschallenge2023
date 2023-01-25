# ENS Data Challenge 2023 - Detecting PIK3CA mutation in breast cancer by OWKIN 
Link: https://challengedata.ens.fr/participants/challenges/98/

Authors: Arianna Morè, João Melo, Maria Stoelben

## Challenge context
🔬 Histopathology
Histopathology is the study of the microscopic structure of diseased human tissue. Analysis of histopathology slides is a critical step for many diagnoses, specifically in oncology where it defines the gold standard. Tissue samples are usually collected during surgery or biopsy. After being preprocessed by expert technicians, pathologists review samples under a microscope in order to assess several biomarkers such as the nature of the tumor, cancer staging etc.

🧬 PIK3CA mutation in breast cancer
Recent studies have also shown that histopathology slides contain information that underlie tumor genotype, therefore they can be used to predict genomic alterations such as point mutations. One of the genomic alterations that is particularly interesting is PIK3CA mutation in breast cancer. They occur in around 30%-40% of breast cancer and are most commonly found in estrogen receptor-positive breast cancer. PIK3CA mutations have been associated with good outcomes. More importantly, patients who carry these mutations and are resistant to endocrine therapy may respond to a class of target therapy - the PI3Kα inhibitor.

🔍 Challenge's purpose
Current method for identifying PIK3CA mutations is DNA sequencing, which requires technical and bioinformatic expertise that is not accessible in all laboratories. An automated solution to detect PIK3CA mutation has high clinical relevance as it could provide a fast, reliable screening tool allowing more patients, especially in tertiary centers, to be eligible to personalized therapies associated to better outcomes.


## Challenge context
The challenge proposed by Owkin is a weakly-supervised binary classification problem. Weak supervision is crucial in digital pathology due to the extremely large dimensions of whole-slide images (WSIs), which cannot be processed as is. To use standard machine learning algorithms one needs, for each slide, to extract smaller images (called tiles) of size 224x224 pixels (approx 112 µm²). Since a slide is given a single binary annotation (presence or absence of mutation) and is mapped to a bag of tiles, one must learn a function that maps multiple items to a single global label. This framework is known as multiple-instance learning (MIL). More precisely, if one of the pooled tiles exhibits a mutation pattern, presence of mutation is predicted while if none of the tiles exhibit the pattern, absence of mutation is predicted. This approach alleviates the burden of obtaining locally annotated tiles, which can be costly or impractical for pathologists.

In this challenge, we aim to predict whether a patient has a mutation of the gene PIK3CA, directly from a slide. For computational purposes, we kept a total of 1,000 tiles per WSI. Each tile was selected such that there is tissue in it.

Here we display an example of whole slide image with 1,000 tiles highlighted in black.

<!---![plot](./directory_1/directory_2/.../directory_n/plot.png)-->

*Figure 1: Example of a whole slide image with the 1,000 tiles selected during preprocessing highlighted in black*

Some of those tiles are displayed below. The coordinates are indicated in parenthesis for each tile.

<!---![plot](./directory_1/directory_2/.../directory_n/plot.png)-->

*Figure 2: Example of 224x224 pixels tiles extracted at a 20x magnification with their (x, y)-coordinates*

