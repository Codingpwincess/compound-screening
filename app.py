import streamlit as st
import pandas as pd
import requests
import time
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw, Descriptors
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ---------- Session State Initialization ----------
for key in ['filtered', 'aggregators', 'rejected_by_model']:
    if key not in st.session_state:
        st.session_state[key] = []

# ---------- Utility Functions ----------

def smiles_to_mols(smiles_list):
    return [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]

def get_chembl_smiles(limit=500):
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule?limit=1000"
    compounds = []
    total = 0
    retries = 0
    max_retries = 5

    while url and total < limit:
        try:
            res = requests.get(url, timeout=10)
            if res.status_code != 200:
                st.warning(f"ChEMBL API returned status code {res.status_code}")
                break

            try:
                data = res.json()
            except ValueError:
                st.error("ChEMBL response is not valid JSON. Aborting.")
                break

            for mol in data.get("molecules", []):
                if 'molecule_structures' in mol and mol['molecule_structures']:
                    s = mol['molecule_structures'].get('canonical_smiles')
                    if s:
                        compounds.append(s)
                        total += 1
                        if total >= limit:
                            break

            url = data['page_meta'].get('next')
            time.sleep(0.5)  # respect API rate limits

        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            retries += 1
            if retries >= max_retries:
                st.error("Too many failed attempts to fetch from ChEMBL.")
                break
            time.sleep(2)

    return compounds

def is_dissimilar(candidate_smiles, clinical_mols, threshold=0.4):
    mol = Chem.MolFromSmiles(candidate_smiles)
    if not mol:
        return False
    cand_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    for ref in clinical_mols:
        if ref:
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref, 2)
            if DataStructs.TanimotoSimilarity(cand_fp, ref_fp) >= threshold:
                return False
    return True

aggregator_smarts = [
    "c1ccccc1C(=O)O", "c1ccccc1CC(=O)O",
    "c1ccccc1CCCC(=O)O", "C1CCCCC1", "c1ccccc1O"
]
aggregator_mols = [Chem.MolFromSmarts(s) for s in aggregator_smarts]

def is_aggregator(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return any(mol.HasSubstructMatch(a) for a in aggregator_mols) if mol else False

def featurize(mol):
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ]

def train_druglikeness_model():
    X = np.array([
        [290, 2.7, 1, 4],
        [310, 3.1, 2, 5],
        [160, 5.4, 1, 1],
        [530, 6.0, 0, 0],
        [370, 2.3, 1, 3],
        [120, 7.2, 0, 1],
        [490, 3.0, 1, 4],
        [180, 6.3, 1, 0]
    ])
    y = [1, 1, 0, 0, 1, 0, 1, 0]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

drug_model = train_druglikeness_model()

def passes_ml_model(smiles, model):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        feats = np.array(featurize(mol)).reshape(1, -1)
        return model.predict(feats)[0] == 1
    return False

def get_mol_properties(mol):
    return {
        "MolWt": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "HDonors": Descriptors.NumHDonors(mol),
        "HAcceptors": Descriptors.NumHAcceptors(mol)
    }

# ---------- Streamlit UI ----------

st.title("Compound Filter: Structural Dissimilarity + Aggregator Check")

uploaded_file = st.file_uploader("Upload Clinical Compound SMILES (CSV with 'smiles' column)", type=["csv"])
max_compounds = st.slider("Number of ChEMBL compounds to process", 100, 5000, 500, step=100)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'smiles' not in df.columns:
        st.error("CSV must contain a 'smiles' column.")
    else:
        clinical_mols = smiles_to_mols(df['smiles'].tolist())
        st.success(f"Loaded {len(clinical_mols)} clinical compounds.")

        if st.button("Run Filtering"):
            with st.spinner("Downloading and filtering ChEMBL compounds..."):
                public_smiles = get_chembl_smiles(limit=max_compounds)

                st.session_state.filtered = []
                st.session_state.aggregators = []
                st.session_state.rejected_by_model = []

                for s in public_smiles:
                    if is_aggregator(s):
                        st.session_state.aggregators.append(s)
                    elif is_dissimilar(s, clinical_mols):
                        if passes_ml_model(s, drug_model):
                            st.session_state.filtered.append(s)
                        else:
                            st.session_state.rejected_by_model.append(s)

                st.subheader("Results")
                st.write(f"**{len(st.session_state.filtered)}** compounds passed filtering.")
                st.write(f"**{len(st.session_state.aggregators)}** potential aggregators detected.")
                st.write(f"**{len(st.session_state.rejected_by_model)}** compounds rejected by ML scoring.")

                st.download_button("Download Filtered Compounds",
                                   pd.DataFrame({'smiles': st.session_state.filtered}).to_csv(index=False),
                                   file_name="filtered_compounds.csv")

                st.download_button("Download Aggregators",
                                   pd.DataFrame({'aggregators': st.session_state.aggregators}).to_csv(index=False),
                                   file_name="flagged_aggregators.csv")

                st.download_button("Download ML-Rejected Compounds",
                                   pd.DataFrame({'ml_rejected': st.session_state.rejected_by_model}).to_csv(index=False),
                                   file_name="ml_rejected.csv")

# ---------- Molecule Preview ----------
if st.session_state.filtered:
    st.subheader("Preview Filtered Molecules (Top 10)")
    for smiles in st.session_state.filtered[:10]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            col1, col2 = st.columns([1, 2])
            with col1:
                img = Draw.MolToImage(mol, size=(250, 250))
                st.image(img, caption=smiles)
            with col2:
                props = get_mol_properties(mol)
                st.write("**Properties:**")
                st.write(f"- Molecular Weight: {props['MolWt']}")
                st.write(f"- LogP: {props['LogP']}")
                st.write(f"- H-Bond Donors: {props['HDonors']}")
                st.write(f"- H-Bond Acceptors: {props['HAcceptors']}")
