# TriMod-DTI
TriMod-DTI: A Tri-Modal Contrastive Learning Framework for Drug-Target Interaction Prediction
<ul>
    <li><a href="#section1">Overview</a></li>
    <li><a href="#section2">Installation</a></li>
    <li><a href="#section3">Data</a></li>
    <li><a href="#section4">Using</li>
    <li><a href="#section5">Acknowledgements</li>
</ul>

<h2 id="section1">Overview</h2>
<p> TriMod-DTI is an innovative drug-target interaction (DTI) prediction framework that integrates three modalities of data, including 1D sequences, 2D molecular graphs, and 3D structures, for feature representation of drugs and proteins. The framework employs a tri-modal contrastive learning strategy, constructing cross-modal positive and negative sample pairs to align feature representations of different modalities in the latent space, thereby achieving feature enhancement.</p>
<img src="https://github.com/llleXu/TriMod-DTI/raw/main/img.png" alt="Image" />

<h2 id="section2">Installation</h2>
<h3> Create a virtual environment </h3>
<p></p>

```bash
conda create -n TriMod-DTI python=3.10
conda activate TriMod-DTI
```

<h3> Clone the repo and install requried python dependencies</h3>
<p></p>

```bash
git clone https://github.com/llleXu/TriMod-DTI.git
cd TriMod-DTI
pip install -r requirements.txt
```

<h2 id="section3">Data</h2>
<h3> Datasets </h3>
The data directory includes all the experimental datasets utilized in TriMod-DTI, covering GPCR, Human, and DrugBank.
<h3> Download SDF and PDB Files </h3>
The drug SDF files and protein PDB files are too large to be hosted directly on GitHub. These files are available for download via Google Drive at the following links:

- GPCR Dataset : [download](https://drive.google.com/drive/folders/17kwk8Nfdu3m0xShX-6AFLL0Y-_VRKqaV?usp=drive_link)  
- Human Dataset : [download](https://drive.google.com/drive/folders/1POucIaqWUIFS3o-Rq3-frwsCInbq4mWs?usp=drive_link)  
- DrugBank Dataset : [download](https://drive.google.com/drive/folders/13UxecbtQehpnpqKc72JJ82A4UE0bnjht?usp=sharing) 


<h2 id="section4">Using</h2>
<h3> Configuration </h3>
Update the file paths in main.py to match your local directory structure. Modify the Data_Encoder initialization as follows:

```bash
Data_set = Data_Encoder(
    txtpath="data/gpcr/train_gpcr.txt",                 # Path to the text file
    sdf_directory="data/sdf_files/",                    # Directory containing SDF files
    sdf_map_path="data/gpcr/sdf_train_id.txt",          # Path to the SDF mapping file
    pdb_directory="data/pdb_files/",                    # Directory containing PDB files
    pdb_map_path="data/pdb_train_id.txt"                # Path to the PDB mapping file
)
```
<h3> Training and evaluation </h3>

```bash
cd dataset
python main.py
```
dataset specifically refers to gpcr, human and drugbank.

<h2 id="section5">Acknowledgements</h2>
This implementation takes inspiration from prior work [1],[2],[3] and introduces several improvements and extensions to the original approach.

```bash
[1]Lin X, Zhang X, Yu Z G, et al. CSCL-DTI: predicting drug-target interaction through cross-view and self-supervised contrastive learning[C]//2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2024: 707-712.
[2]Yazdani-Jahromi M, Yousefi N, Tayebi A, et al. AttentionSiteDTI: an interpretable graph-based model for drug-target interaction prediction using NLP sentence-level relation classification[J]. Briefings in Bioinformatics, 2022, 23(4): bbac272.
[3]Mu S, Liang S, Li X. Medication Recommendation via Dual Molecular Modalities and Multi-Substructure Enhancement[J]. arXiv preprint arXiv:2405.20358, 2024.
```
