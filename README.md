# folder structure:
## data
1. Datasets Caviar/: Caviar covert network dataset
2. coauth-DBLP-full/: DBLP coauthorship dataset
3. Bali2_Relations_Public_Version2.csv: Bali bombings terrorism dataset
4. CE_Relations_Public_Version2.csv: Christmas Eve Bombings terrorism dataset
5. communication.csv: email data, to produce Fig.4
## code_repo/
augmented.py: file to predict links by intermediate network structure, and evaluate performance. 
Call function "intermediate_auc" to produce AUC plots
To change evaluation method using super adjacency matrix: change pair_set_future from lines 782-787.
