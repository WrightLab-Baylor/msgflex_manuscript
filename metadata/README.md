Reanalysis of six public proteomics datasets with the MSGFLEX pipeline (MS-GF+ search, xpectra rescoring). The submission contains result files only; original raw data remain in the source repositories.

## 1. Source datasets

| PXD accession | MSV accession | Source repository | Runs | Pipeline(s) | Publication DOI |
|---|---|---|---|---|---|
| PXD006118 | NA | PRIDE | 144 | conventional | 10.1038/s41467-017-01544-x |
| PXD018408 | MSV000085239 | MassIVE | 16 | conventional | 10.1021/acsomega.0c04030 |
| PXD035981 | NA | PRIDE | 24 | binning/splitting | 10.1016/j.isci.2024.110093 |
| PXD052873 | NA | PRIDE | 34 | binning/splitting | 10.1038/s41467-025-59250-y |
| PXD057701 | NA | PRIDE | 8 (x2 pipelines) | conventional and binning/splitting | 10.1016/j.cell.2024.12.016 |
| PXD074461 | NA | PRIDE | 24 | conventional | 10.3390/ijms27093918 |

Datasets without an MSV accession are PRIDE-sourced and are identified here by their PXD accession. Each dataset was used for a different test within the study. Reanalysis parameters follow the original publications (see Section 3).

## 2. Software

| Tool | Version | Role |
|---|---|---|
| MSGFLEX | 1.1.0 | Pipeline developed by the authors; includes xpectra. https://github.com/WrightLab-Baylor/msgflex |
| MS-GF+ | 2024.03.26 | Database search engine used within MSGFLEX |
| xpectra | part of MSGFLEX 1.1.0 | Rescoring of PSMs |



## 3. Processing workflow

1. Raw files were converted to mzML.
2. Spectra were searched with MS-GF+ inside MSGFLEX. Search parameters, enzyme, tolerances, fixed/variable modifications and instrument settings follow each original study.
3. Protein databases (FASTA) are the ones supplied with each original study. No custom database was used.
4. MSGFLEX writes an `.mzid` file and a `_fht_SICStats.txt` file per run. The latter is produced at the end of the pipeline after PHRP processing and contains MASIC-extracted statistics.
5. PSMs were rescored with xpectra (`_rescored.tsv`).
6. For selected datasets, `.pin` and `.pepXML` files are provided (see Section 5).

### Pipelines compared

- **conventional**: search of the full database without splitting.
- **binning/splitting**: Large metagenome databases were randomly split into bins and searched separately, and results merged.

PXD057701 was processed with both pipelines. Files from the two pipelines are distinguished by the `conv_` and `bin_` file name prefixes and by the `Processing_pipeline` column of the mapping file.


## 3. Mapping file

`accession_map.tsv` has one row per original run (and per pipeline for PXD057701). Missing values are `NA`.

| Column | Description |
|---|---|
| PXD_Accession | ProteomeXchange accession of the original dataset |
| MSV_Accession | MassIVE accession of the original dataset, `NA` if none |
| source_repository | Repository hosting the original data |
| original_raw_file | Original raw file name in the source repository |
| mzml_file | mzML file name used for the reanalysis |
| Processing_pipeline | `conventional` or `binning/splitting` |
| result_mzid | MS-GF+ mzIdentML file |
| result_tsv | `_fht_SICStats.txt` file |
| rescored_psm_file | xpectra rescored PSM table |
| pin_file | `.pin` file, `NA` if not generated |
| pepxml_file | `.pepXML` file, `NA` if not generated |
| publication_doi | DOI of the original publication |

## 4. Contact

- Contact: tulasi_relangi@baylor.edu; a_wright@baylor.edu
