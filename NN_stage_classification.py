import helper.rnaseq_dataframe as rnaseq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection, preprocessing

DE_LOG2_FOLD_CHANGE_MIN = 2
DE_PADJ_MAX = 0.001
PROTON_TRANSPORTERS = [
    "ATP6V0A1", "ATP6V1H", "CA11", "CA12", "AQP6", "ATP6V1D", "AQP8", "AQP9", "CA2",
    "ATP6V0A4", "CA9", "TCIRG1", "ATP6V0E1", "ATP6V1A", "ATP6V1B1", "ATP6V0B", "CA14",
    "ATP6V1F", "ATP6V1E1", "CA6", "CA1", "ATP6V1G1", "AQP10", "ATP6V1C2", "ATP6V1B2",
    "ATP6V0D2", "ATP6V1G3", "CA10", "ATP6V1C1", "ATP6V0D1", "AQP5", "CA3", "AQP7",
    "AQP7", "AQP3", "CA4", "AQP2", "CA7", "CA5B", "ATP6V0E2", "AQP4", "CA5A", "AQP11",
    "CA8", "AQP12A", "CA13", "AQP12B", "ATP6V0A2", "ATP6V0C", "ATP6V1G2", "AQP1",
    "ATP6V1E2", "AL845331.2", "SLC4A1", "SLC4A7", "SLC4A8", "SLC9A7", "SLC9A3", "SLC4A4",
    "SLC4A11", "SLC9A1", "SLC26A4", "SLC26A3", "SLC26A8", "SLC4A9", "SLC4A3", "SLC9A2",
    "SLC26A10", "SLC9A5", "SLC16A3", "SLC4A10", "SLC26A1", "SLC16A2", "SLC26A7",
    "SLC16A1", "SLC26A2", "SLC4A1AP", "SLC4A2", "SLC16A4", "SLC26A5", "SLC26A9",
    "SLC9A4", "SLC26A11", "SLC4A5", "SLC9A6", "SLC26A6"
]

stageAnnot = pd.read_csv("/home/jovyan/CSBL_shared/RNASeq/TCGA/annotation/fpkm_annot.csv")
geneIDMap = pd.read_csv("/home/jovyan/CSBL_shared/ID_mapping/Ensembl_symbol_entrez.csv",
                        dtype=str)

# Read FPKM data in and convert to TPM values
geneExp = pd.read_csv("/home/jovyan/CSBL_shared/RNASeq/TCGA/FPKM/TCGA-COAD.FPKM.csv")
geneExp.index = geneExp["Ensembl"]
geneExp = geneExp.drop("Ensembl", axis=1)
geneExp = rnaseq.FPKM_to_TPM(geneExp)
geneExp = rnaseq.convert_geneID(geneExp, geneIDMap, colName="index",
                                fromID="ensembl_gene_id", toID="external_gene_name",
                                protein_coding_only=True)

# Remove samples without cancer stage labels
projStage = rnaseq.get_barcode_stage(stageAnnot, "TCGA-COAD")
geneExp = geneExp[projStage["barcode"]]

# Get features that are differentially expressed
geneDE = pd.read_csv("/home/jovyan/storage/data/TCGA/DEA/csv/TCGA-COAD_I_vs_N.csv", index_col=0)
geneDE = rnaseq.convert_geneID(geneDE, geneIDMap, colName="index",
                               fromID="ensembl_gene_id", toID="external_gene_name",
                               protein_coding_only=True)
geneDE["absLog2FC"] = geneDE["log2FoldChange"].abs()
geneDE = geneDE[(geneDE["absLog2FC"] >= DE_LOG2_FOLD_CHANGE_MIN) &
                (geneDE["padj"] <= DE_PADJ_MAX)]
geneDE = geneDE.sort_values(["padj", "absLog2FC"], ascending=[True, False])
print(f"""
    {geneDE.shape[0]} differentially expressed genes are taken.
    {geneDE[geneDE.index.isin(PROTON_TRANSPORTERS)].shape[0]} genes in the trasnporter list are DEGs.
    """)

# Transform features into numpy arrays
geneFeatures = set(geneDE.index.tolist() + PROTON_TRANSPORTERS)
geneFeatures = geneExp[geneExp.index.isin(geneFeatures)]

geneFeatureNames = np.array(geneFeatures.index)
geneFeatures = geneFeatures.T.values

# Transform labels into numpy arrays
labelEnc = preprocessing.LabelEncoder()

geneLabels = projStage["cancer_stage"].values
labelEnc.fit(geneLabels)
geneLabels = labelEnc.transform(geneLabels)
geneLabelNames = labelEnc.inverse_transform(geneLabels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(geneFeatures, geneLabels,
                                                                    test_size=0.2, random_state=None)
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

print("X_test original shape", X_test.shape)
print("y_test original shape", y_test.shape)
