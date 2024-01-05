# folder structure:
## data
1. Datasets Caviar/: Caviar covert network dataset, to produce Fig.2
2. coauth-DBLP-full/: DBLP coauthorship dataset, to produce Fig.5
3. Bali2_Relations_Public_Version2.csv: Bali bombings terrorism dataset, to produce Fig.3b
4. CE_Relations_Public_Version2.csv: Christmas Eve Bombings terrorism dataset, to produce Fig. 3a
5. communication.csv: email data, to produce Fig.4
## code_repo
augmented.py: file to predict links by intermediate network structure, and evaluate performance. 

Call function "intermediate_auc" to produce AUC plots
To change evaluation method using super adjacency matrix: change pair_set_future from lines 782-787.
