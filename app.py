import streamlit as st
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# ---------- Utility Functions ----------

def smiles_to_mols(smiles_list):
    return [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]

def get_chembl_smiles(limit=5000):
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule?limit=1000"
    compounds = []
    total = 0
    while url and total < limit:
        res = requests.get(url).json()
        for mol in res['molecules']:
            if 'molecule_structures' in mol and mol['molecule_structures']:
                s = mol['molecule_structures'].get('canonical_smiles')
                if s:
                    compounds.append(s)
                    total += 1
                    if total >= limit:
                        break
        url = res['page_meta'].get('next')
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

# ---------- Streamlit UI ----------

st.title("Compound Filter: Structural Dissimilarity + Aggregator Check")

uploaded_file = st.file_uploader("Upload Clinical Compound SMILES (CSV with 'smiles' column)", type=["csv"])
max_compounds = st.slider("Number of ChEMBL compounds to process", 100, 10000, 1000, step=500)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'smiles' not in df.columns:
        st.error("CSV must contain a 'smiles' column.")
    else:
        clinical_mols = smiles_to_mols(df['smiles'].tolist())
        st.success(f"Loaded {len(clinical_mols)} clinical compounds.")

        if st.button("Run Filtering"):
            with st.spinner("Downloading and filtering compounds..."):
                public_smiles = get_chembl_smiles(limit=max_compounds)

                filtered, aggregators = [], []
                for s in public_smiles:
                    if is_aggregator(s):
                        aggregators.append(s)
                    elif is_dissimilar(s, clinical_mols):
                        filtered.append(s)

                st.subheader("Results")
                st.write(f"**{len(filtered)}** compounds passed filtering.")
                st.write(f"**{len(aggregators)}** potential aggregators detected.")

                st.download_button("Download Filtered Compounds",
                                   pd.DataFrame({'smiles': filtered}).to_csv(index=False),
                                   file_name="filtered_compounds.csv")

                st.download_button("Download Aggregators",
                                   pd.DataFrame({'aggregators': aggregators}).to_csv(index=False),
                                   file_name="flagged_aggregators.csv")
