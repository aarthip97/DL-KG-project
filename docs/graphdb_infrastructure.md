# 🗄️ Music RecSys: GraphDB Infrastructure

This directory contains the automated setup scripts to deploy **Ontotext GraphDB (v11.3.3)** for the Music Recommendation System Knowledge Graph. 

By running the provided setup scripts, the database will automatically download, apply the enterprise/free license, and configure itself as a persistent, system-wide background daemon on your local machine or cluster. 

>*Note: This performs a system-wide installation (e.g., `~/graphdb-server` or `%USERPROFILE%\graphdb-server`) to keep this Git repository clean and allow you to use the database for other projects.*

---

## 📋 Prerequisites

* **Java 11 or 17** must be installed and available in your system's `PATH`.
* **License File (Optional but Recommended):** If you received a `.license` file from Ontotext, save it in the root of this repository and name it exactly `graphdb.license`. The setup scripts will automatically inject it into the database configuration.

---

## 🚀 Quick Start Installation

### For Linux (Arch / Ubuntu / RHEL Clusters)
Run the bash script from your terminal. It will download GraphDB, inject the license, and set it up as a `systemd` background service.

```bash
cd scripts/
chmod +x setup_graphdb.sh
./setup_graphdb.sh
```

*(Note: The script will prompt you for your `sudo` password to register and enable the `systemd` service).*

### For Windows

Run the batch file from PowerShell, CMD, or by double-clicking it in File Explorer. It will download the distribution, extract it to your user profile, inject the license, and launch it in the background.

```cmd
cd scripts\
setup_graphdb.bat
```

---

## 🧭 Exploring the GraphDB Workbench

Once the setup script finishes, the database will be running silently in the background.
Open your web browser and navigate to: **[http://localhost:7200](http://localhost:7200)**

### Key Features:

#### 📥 **Importing Data (GUI Method):**
* Go to **Import -> User data**.
* Upload your `.ttl` (Schema) and `.nt` (Interactions) files.
* Click the **Import** button to ingest them into the repository.


#### 🕸️ **Visual Graph Exploration:**
* Go to **Explore -> Visual graph**.
* Type in a specific URI (e.g., `track:TR12345`) to visually map its relationships (Genres, Artists, Users) in an interactive web interface. Highly recommended for generating screenshots for reports.


#### 🔍 **SPARQL Endpoint:**
* Go to **SPARQL**.
* Use this interface to write and test your graph queries before locking them into the Python extraction pipeline. You can export query results directly as CSV/TSV from this window.


#### 🔒 **Security & Team Access:**
* Go to **Setup -> Users**.
* Toggle Security to "On" to set an admin password. *(Remember to copy `.env.example` to `.env` and update it with your credentials for the Python pipeline).*



---

## 🐍 Connecting the Python Pipeline

Once the database is running, do **not** use `rdflib` to load massive files directly into memory.

Instead, use our Python REST API controller located at `src/data/kg/graphdb_client.py`. This script automatically pushes new `.ttl` updates to the database and extracts the lightweight `.tsv` artifacts required for the PyKEEN and PyTorch Geometric deep learning pipelines.

