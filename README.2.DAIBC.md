# DAIBC
###### Data Analysis of *In situ* sequencing Base Calling
###### (R + shiny version)

---

## Introduction

This software is used to analyze the results of pyIRIS by importing barcode info file and the two result files of 
pyIRIS.

## The file format of imported files
### The format of barcode info file

This file should be prepared by manual with a format like following, of which, the 1st field means barcode sequence, 
and the 2nd one means gene info. **DON'T INSERT ANY SPACE CHARACTER INTO GENE INFO**:

    AACA    SOX2
    AGTC    BIRC5
    GTCA    SCUBE2
    AACT    KLF4
    AGCT    CCNB1
    GCAT    ACTB
    AACG    TP53
    ACTG    MYBL2
    GCTA    GAPDH
    TGAC    HER2
    CTGA    VIM

### The format of result files of pyIRIS

See 'README.1.pyIRIS.md' for detail.