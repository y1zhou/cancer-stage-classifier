import keras
import pandas as pd
import matplotlib.pyplot as plt

from importlib import reload
import helper.rnaseq_dataframe as rnaseq

DE_LOG2_FOLD_CHANGE_MIN = 2
DE_PADJ_MAX = 0.001
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
geneDE.shape
