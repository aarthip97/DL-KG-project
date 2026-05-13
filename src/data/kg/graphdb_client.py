import os
import requests
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()
GDB_URL = os.getenv("GRAPHDB_URL")
GDB_USER = os.getenv("GRAPHDB_USERNAME")
GDB_PWD = os.getenv("GRAPHDB_PASSWORD")
REPO_NAME = os.getenv("GRAPHDB_REPO")

# Setup authentication session
session = requests.Session()
if GDB_USER and GDB_PWD:
    session.auth = (GDB_USER, GDB_PWD)

def setup_admin_user():
    """Creates the user with Admin privileges if credentials are provided in .env."""
    if not GDB_USER or not GDB_PWD:
        print("[*] No credentials in .env. Skipping user creation.")
        return

    print(f"[*] Verifying user '{GDB_USER}'...")
    url = f"{GDB_URL}/rest/security/users/{GDB_USER}"
    
    # Payload to grant full admin rights
    user_payload = {
        "password": GDB_PWD,
        "grantedAuthorities": ["ROLE_ADMIN", "ROLE_USER"]
    }
    
    # Try to create the user
    headers = {"Content-Type": "application/json"}
    response = session.post(url, json=user_payload, headers=headers)
    
    if response.status_code == 201:
        print(f"[+] User '{GDB_USER}' successfully created and granted Admin roles.")
    elif response.status_code == 409 or response.status_code == 400:
        print(f"[*] User '{GDB_USER}' already exists. Proceeding...")
    else:
        print(f"[!] Warning during user creation: {response.text}")

def create_repository():
    """Creates the repository if it doesn't exist."""
    print(f"[*] Verifying repository '{REPO_NAME}'...")
    repo_config = {
        "id": REPO_NAME,
        "params": {
            "title": "Music RecSys Knowledge Graph",
            "ruleset": "rdfsplus-optimized", 
            "disableSameAs": "false"
        },
        "type": "graphdb"
    }
    
    response = session.post(f"{GDB_URL}/rest/repositories", json=repo_config)
    if response.status_code == 201:
        print(f"[+] Repository '{REPO_NAME}' created successfully.")
    elif response.status_code == 409:
        print(f"[*] Repository '{REPO_NAME}' already exists.")
    else:
        print(f"[!] Error creating repo: {response.text}")

def upload_rdf_file(file_path, content_type="text/turtle"):
    """Uploads a .ttl or .nt file directly to the database."""
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}. Skipping upload.")
        return

    print(f"[*] Uploading {file_path} to GraphDB... (This may take a minute)")
    url = f"{GDB_URL}/repositories/{REPO_NAME}/statements"
    headers = {"Content-Type": content_type}
    
    with open(file_path, 'rb') as f:
        response = session.post(url, headers=headers, data=f)
        
    if response.status_code == 204:
        print(f"[+] Successfully uploaded {file_path}")
    else:
        print(f"[!] Failed to upload: {response.text}")

def export_pykeen_tsv(output_path):
    """Executes a SPARQL query and downloads the result as a TSV."""
    print(f"[*] Extracting DL artifacts to {output_path}...")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sparql_query = """
    PREFIX mrc: <http://purl.org/ontology/mrc/>
    
    SELECT ?head ?relation ?tail WHERE {
        ?head ?relation ?tail .
        FILTER(?relation != <http://www.w3.org/2002/07/owl#sameAs>)
    }
    """
    
    url = f"{GDB_URL}/repositories/{REPO_NAME}"
    headers = {
        "Accept": "text/tab-separated-values", 
        "Content-Type": "application/sparql-query"
    }
    
    response = session.post(url, headers=headers, data=sparql_query)
    
    if response.status_code == 200:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("[+] TSV successfully extracted and saved!")
    else:
        print(f"[!] Query failed: {response.text}")

# ==========================================
# Execution Flow
# ==========================================
if __name__ == "__main__":
    print("=== Starting GraphDB Data Pipeline ===")
    
    # 1. Setup Infrastructure
    setup_admin_user()
    create_repository()
    
    # 2. Upload Data (Update these paths to point to your actual data files!)
    # upload_rdf_file("path/to/your/music_ontology.ttl", "text/turtle")
    # upload_rdf_file("path/to/your/user_interactions.nt", "text/plain") 
    
    # 3. Extract Deep Learning Artifacts
    export_pykeen_tsv("data/interim/kg_triples.tsv")
    
    print("=== Pipeline Complete ===")