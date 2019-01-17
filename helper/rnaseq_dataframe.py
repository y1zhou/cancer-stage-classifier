import pandas as pd


def FPKM_to_TPM(df):
    """FPKM expression values to TPMs.
    The sum of all TPMs in each sample are the same.
    Refer to : https://www.rna-seqblog.com/rpkm-fpkm-and-tpm-clearly-explained/

    Parameters
    ----------
        df: pandas.DataFrame
            Each row is a gene, and each column is a sample.

    Returns
    -------
        a pandas.DataFrame with the same shape as df.
    """
    colSum = df.sum(axis=0)
    df = df.div(colSum, axis=1) * 1e+6
    return df


def get_barcode_stage(annotation, project):
    """Return the barcodes of the project with corresponding cancer stages.

    Parameters
    ----------
        annotation: pandas.DataFrame
            A data frame with matching uuids and tumor stages.
        project: str
            The desired cancer project.

    Returns
    -------
        A pandas.DataFrame of barcodes and their corresponding cancer stages.
    """
    assert project in annotation["project"].unique(), f"{project} not found in annotation."
    annot = annotation[annotation["project"] == project]
    normalUUID = annot[annot["sample_type"].str.lower().str.contains("normal")]["barcode"].tolist()
    # separate stages (and uncharacterized)
    stage1ID = annot[(~annot["barcode"].isin(normalUUID)) &
                     (annot["tumor_stage"].str.contains(r"^i$|\si[abc]?$|1", regex=True))]["barcode"].tolist()
    stage2ID = annot[(~annot["barcode"].isin(normalUUID)) &
                     (annot["tumor_stage"].str.contains(r"^ii$|\si{2}[abc]?$|2", regex=True))]["barcode"].tolist()
    stage3ID = annot[(~annot["barcode"].isin(normalUUID)) &
                     (annot["tumor_stage"].str.contains(r"^iii$|\si{3}[abc]?$|3", regex=True))]["barcode"].tolist()
    stage4ID = annot[(~annot["barcode"].isin(normalUUID)) &
                     (annot["tumor_stage"].str.contains(r"^iv$|\siv[abc]?$|4", regex=True))]["barcode"].tolist()
    stageID = pd.DataFrame(data={
        "barcode": normalUUID + stage1ID + stage2ID + stage3ID + stage4ID,
        "cancer_stage": ["normal"] * len(normalUUID) + ["i"] * len(stage1ID) + ["ii"] * len(stage2ID) + ["iii"] * len(stage3ID) + ["iv"] * len(stage4ID)
    })
    return stageID


def convert_geneID(df, geneIDMap, colName="index",
                   fromID="ensembl_gene_id", toID="external_gene_name", protein_coding_only=False):
    """Convert one type of gene ID in the dataframe to another.

    df.colName is converted from fromID to toID. If df.colName is versioned Ensembl gene IDs,
    the version number is disregarded.

    Parameters
    ----------
        df: pandas.DataFrame
            The dataframe with column `colName` to be converted.
        geneIDMap: pandas.DataFrame
            A dataframe of matching gene IDs. Columns should include strings `from` and `to`.
        colName: str, optional
            The name of the column in `df` whose IDs are to be converted. Default is the index.
        fromID: str, optional
            The current gene ID type.
        toID: str, optional
            The desired output gene ID type.
        protein_coding_only: bool, optional
            if True, filter out ones without entrez gene IDs.
    Returns
    -------
        A pandas.DataFrame with converted gene IDs.
    """
    assert colName in ["index"] + df.columns.tolist(), f"{colName} not found in df."
    assert fromID in geneIDMap.columns, f"{fromID} not found in geneIDMap."
    assert toID in geneIDMap.columns, f"{toID} not found in geneIDMap."

    if protein_coding_only:
        geneIDMap = geneIDMap[~geneIDMap["entrezgene"].isna()]
    geneIDMap = geneIDMap[[fromID, toID]]

    if colName == "index":
        df["index"] = df.index
    if fromID == "ensembl_gene_id":
        df[colName] = df[colName].str.replace(r"\.\d*$", "")

    df = pd.merge(geneIDMap, df, how="inner", left_on=fromID, right_on=colName)
    df.index = df[toID]
    df = df.drop([colName, fromID, toID], axis=1)
    return df
