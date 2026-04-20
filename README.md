# DL-KG-project

**Music Recommendation with Knowledge Graph and GNNs**

A notebooks-first project that combines symbolic music analysis with heterogeneous graph learning to build a music recommendation system and perform link prediction on a Music Knowledge Graph.

---

## Project structure

```
DL-KG-project/
├── data/
│   ├── raw/            ← original datasets (not committed to git)
│   ├── interim/        ← intermediate artefacts
│   └── processed/      ← clean Parquet / KG files ready for modelling
├── docs/
├── notebooks/
│   ├── 00_dataset_extraction_exploration.ipynb   ← MIDI × MSD alignment
│   ├── 01_music_features_extraction.ipynb        ← musif symbolic features
│   ├── 02_kg_construction.ipynb                  ← rdflib
│   └── 03_DL_model_training.ipynb                ← GNN
└── src/
    ├── __init__.py
    └── data/
        ├── __init__.py
        ├── dataset_extraction.py       ← LakhMSDLinker, read_msd_metadata
        └── music_feature_extraction.py ← musif wrappers (batch_extract, …)
```

---

## Datasets required

| Dataset | Path (set in notebooks) |
|---------|------------------------|
| Lakh Matched MIDI | `lmd_matched/` |
| MSD HDF5 summaries | `lmd_matched_h5/` |
| DTW match scores | `match_scores.json` |

Download from [colinraffel.com/projects/lmd](https://colinraffel.com/projects/lmd/).

---

## Quickstart

### 1 — Create and activate the virtual environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Register a named Jupyter kernel

Install the venv as a kernel so it shows up with a recognisable name in any
Jupyter interface (VS Code, JupyterLab, classic Notebook):

```bash
pip install ipykernel                          # already in requirements, just in case
python -m ipykernel install \
    --user \
    --name        "dl-kg-project" \
    --display-name "DL-KG Project (.venv)"
```

| Flag | Purpose |
|------|---------|
| `--user` | Installs the kernel spec into `~/.local/share/jupyter/kernels/` (no sudo needed) |
| `--name` | Internal identifier used by Jupyter |
| `--display-name` | Human-readable label shown in the kernel picker |

After running the command you can verify the kernel is registered:

```bash
jupyter kernelspec list
# dl-kg-project   /home/<you>/.local/share/jupyter/kernels/dl-kg-project
```

### 3 — Open notebooks

```bash
jupyter lab          # or open the .ipynb files directly in VS Code
```

Select **"DL-KG Project (.venv)"** from the kernel picker when prompted.

Open notebooks in order: **00 → 00b → 01 → 02 → 03**.

> **Tip — removing a kernel**  
> `jupyter kernelspec remove dl-kg-project`

---

## Pipeline overview

```
lmd_matched/  +  lmd_matched_h5/  +  match_scores.json
        │
        ▼  notebook 00
  lakh_msd_dataset.parquet   (one row per track, MSD metadata + MIDI path)
        │
        ▼  notebook 01 
  music_features_with_meta.parquet   (900+ symbolic features)
        │
        ▼  notebook 02  (rdflib)
  music_kg.ttl   +   Data transformation
        │
        ▼  notebook 03  GNN training 
```

---

## Key dependencies

- **[musif](https://musif.didone.eu)** — MIDI-native symbolic feature extraction (DIDONE project)
- **h5py** — MSD HDF5 reading without pytables
- **rdflib** — RDF Knowledge Graph serialisation
- **PyTorch Geometric** — GraphSAGE heterogeneous graph learning

---

## Acknowledgements

- Lakh MIDI Dataset — Colin Raffel
- Million Song Dataset — Thierry Bertin-Mahieux et al.
- musif — DIDONE project (Universitat Pompeu Fabra)
