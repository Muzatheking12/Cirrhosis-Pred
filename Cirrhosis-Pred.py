import streamlit as st
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import pandas as pd
from streamlit_ketcher import st_ketcher
from rdkit.Chem import Draw
# Load pre-trained models
main = os.path.dirname(__file__)
model1path = os.path.join(main, r"TGF/model.joblib")
model2path = os.path.join(main, r"PDGF/model.joblib")
model3path = os.path.join(main, r"1KK/model.joblib")
model4path = os.path.join(main, r"TNF/model.joblib")
model1col = os.path.join(main, r"TGF/col.csv")
model2col = os.path.join(main, r"PDGF/col.csv")
model3col = os.path.join(main, r"1KK/col.csv")
model4col = os.path.join(main, r"TNF/col.csv")
smiledraw = os.path.join(main, 'mol.png')
tantgf = os.path.join(main, r'TGF/tanimoto_dist.png')
tanpdgf = os.path.join(main, r'PDGF/tanimoto_dist.png')
tanikk = os.path.join(main, r'1KK/tanimoto_dist.png')
tantnf = os.path.join(main, r'TNF/tanimoto_dist.png')
tsnetgf = os.path.join(main, r'TGF/tsne.png')
tsnepdgf = os.path.join(main, r'PDGF/tsne.png')
tsneikk = os.path.join(main, r'1KK/tsne.png')
tsnetnf = os.path.join(main, r'TNF/tsne.png')
comtgf = os.path.join(main, r'TGF/Figure_1.png')
compdgf = os.path.join(main, r'PDGF/Figure_1.png')
comikk = os.path.join(main, r'1KK/Figure_1.png')
comtnf = os.path.join(main, r'TNF/Figure_1.png')
mettgf = os.path.join(main, r'TGF/class_metrics.png')
metpdgf = os.path.join(main, r'PDGF/class_metrics.png')
metikk = os.path.join(main, r'1KK/class_metrics.png')
mettnf = os.path.join(main, r'TNF/class_metrics.png')
liver = os.path.join(main , 'livercirr.png')
TGF = joblib.load(model1path)
PDGF = joblib.load(model2path)
IKK = joblib.load(model3path)
TNF = joblib.load(model4path)

# Function to compute Morgan fingerprint
def tanimoto(fp1, fp2):
        intersection = np.sum(np.bitwise_and(fp1, fp2))
        union = np.sum(np.bitwise_or(fp1, fp2))

        if union == 0:
            return 0
        else:
            return intersection / union
def ext_tanimoto(threshold, fingerprinty, fingerprintx):
    external_molecule = np.array(fingerprinty)
    tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
    sorted_indices = np.argsort(tanimoto_similarities)[::-1]
    top_k_indices = sorted_indices[1:6]
    mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
    print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
    if mean_top_k_similarity >= threshold:
                                    AD = 'IN Applicability Domain'
    else:
                                    AD = 'OUT Applicability Domain'
    return AD

def compute_morgan_fingerprint(smiles, col, radius=2, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES string. Please check your input.")
            return None
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
         # Convert to numpy array and align with model features
        fp_arr = np.zeros((1, n_bits), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fingerprint, fp_arr[0])
        fp_df = pd.DataFrame(fp_arr, columns=[f"FP_{i+1}" for i in range(n_bits)])
        df = pd.read_csv(col)
        R = df.drop(columns=['Bioactivity'], axis=1)  # Assuming these are your feature columns
        common_columns = [col for col in fp_df.columns if col in R.columns]
        
        # Filter the fp_df to only include the common columns that exist in both
        X = fp_df[common_columns]

        # Convert DataFrame to numpy array for model input
        X = X.values # Ensure column order m
        y = R.values
        return X, y
    except Exception as e:
        st.error(f"Error computing fingerprint: {e}")
        return None

st.image(liver, width=200)
st.title("Machine Learning-Based Multi-Model System for Predictive Inhibition in Cirrhosis Pathways")

st.write("""
This application integrates multiple binary classification models to predict inhibitory activity in cirrhosis pathways. 
The system includes:
- **Two Anti-Fibrotic Models**: TGF-Beta and PDGF
- **Two Anti-Inflammatory Models**: IKKB and TNF-Alpha

### Key Features:
- Models are validated using a 10x10 iteration of K-Fold Cross Validation.
- Applicability Domain (AD) is defined for reliable predictions (Based on Tanimoto Similarity).
- All models achieve **MCC_ext ≥ 0.54** and **ACC_ext ≥ 0.79**.
""")

st.write("## Instructions")
st.write("""
- **Input SMILES**: Enter a SMILES string and submit to predict active\inactive based on IC50 threshold 500nM.
- **Draw Compounds**: Use the compound sketcher and click 'Apply' to copy the SMILES into the input field.
- **Adjust Thresholds**: Modify the AD threshold to filter out compounds with low similarity.
- **Batch Input**: Upload an Excel file (.XLSX) containing a column labeled 'SMILES' for batch predictions.
""")



# Sidebar: SMILES input
sketched_smiles = st_ketcher()
smiles_input = st.sidebar.text_input("Enter a SMILES string:", sketched_smiles, placeholder="C1=CC=CC=C1")
button1 = st.sidebar.button("Submit")

# Sidebar: Upload Excel file
uploaded_file = st.sidebar.file_uploader("Upload an Excel file with SMILES", type=["xlsx"])

adjust_tgf_threshold = st.sidebar.checkbox("Adjust TGF-Beta Threshold")
tgf_threshold = st.sidebar.slider("TGF-Beta Threshold", 0.0, 1.0, 0.15, 0.01) if adjust_tgf_threshold else 0.15

adjust_pdgf_threshold = st.sidebar.checkbox("Adjust PDGF-Beta Threshold")
pdgf_threshold = st.sidebar.slider("PDGF-Beta Threshold", 0.0, 1.0, 0.15, 0.01) if adjust_pdgf_threshold else 0.15

adjust_ikk_threshold = st.sidebar.checkbox("Adjust IKKB Threshold")
ikk_threshold = st.sidebar.slider("IKKB Threshold", 0.0, 1.0, 0.14, 0.01) if adjust_ikk_threshold else 0.14

adjust_tnf_threshold = st.sidebar.checkbox("Adjust TNF-Alpha Threshold")
tnf_threshold = st.sidebar.slider("TNF-Alpha Threshold", 0.0, 1.0, 0.14, 0.01) if adjust_tnf_threshold else 0.14

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prediction Results", "TGF", "PDGF", "IKK", "TNF"])


    # Add SMILES from text input
if button1:
        with tab1:
            m = Chem.MolFromSmiles(smiles_input, sanitize=False)
            AllChem.Compute2DCoords(m)

            # Save the 2D structure as an image file
        
            img = Draw.MolToImage(m)
            img.save(smiledraw)
            st.image(smiledraw)
            fingerprint1, y1 = compute_morgan_fingerprint(smiles_input, model1col)
            AD_TGF = ext_tanimoto(tgf_threshold, fingerprint1, y1)
            prediction1 = TGF.predict(fingerprint1)[0]
            fingerprint2, y2 = compute_morgan_fingerprint(smiles_input, model2col)
            AD_PDG = ext_tanimoto(pdgf_threshold, fingerprint2, y2)
            prediction2 = PDGF.predict(fingerprint2)[0]
            fingerprint3, y3 = compute_morgan_fingerprint(smiles_input, model3col)
            AD_IKK = ext_tanimoto(ikk_threshold, fingerprint3, y3)
            prediction3 = IKK.predict(fingerprint3)[0]
            fingerprint4, y4 = compute_morgan_fingerprint(smiles_input, model4col)
            AD_TNF = ext_tanimoto(tnf_threshold, fingerprint4, y4)
            prediction4 = TNF.predict(fingerprint4)[0]
                
            # Display predictions
            st.subheader("Predictions")
            st.write(f"Anti-Fibrotic Pathway TGF-Beta 1  : **{'Active' if prediction1 == 1 else 'Inactive'}** (**{AD_TGF}**)")
            st.write(f"Anti-Fibrotic Pathway PDGF-Beta : **{'Active' if prediction2 == 1 else 'Inactive'}** (**{AD_PDG}**)")
            st.write(f"Anit-Inflammatory Pathway IKKB : **{'Active' if prediction3 == 1 else 'Inactive'}** (**{AD_IKK}**)")
            st.write(f"Anit-Inflammatory Pathway TNF-Alpha : **{'Active' if prediction4 == 1 else 'Inactive'}** (**{AD_TNF}**)")

    # Add SMILES from sketcher
smiles_list = []

# Add SMILES from uploaded file
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        if "SMILES" in df.columns:
            smiles_list.extend(df["SMILES"].dropna().tolist())
        else:
            st.sidebar.error("The uploaded file must contain a 'SMILES' column.")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

# Create a table for SMILES, predictions, and AD

if smiles_list:
  with tab1:
    st.subheader("Predictions for Uploaded SMILES")
    results = []  # To store results for the table
    for smiles in smiles_list:
        try:
            fingerprint1, y1 = compute_morgan_fingerprint(smiles, model1col)
            AD_TGF = ext_tanimoto(tgf_threshold, fingerprint1, y1) if fingerprint1 is not None else "N/A"
            prediction1 = TGF.predict(fingerprint1)[0] if fingerprint1 is not None else "N/A"

            fingerprint2, y2 = compute_morgan_fingerprint(smiles, model2col)
            AD_PDG = ext_tanimoto(pdgf_threshold, fingerprint2, y2) if fingerprint2 is not None else "N/A"
            prediction2 = PDGF.predict(fingerprint2)[0] if fingerprint2 is not None else "N/A"

            fingerprint3, y3 = compute_morgan_fingerprint(smiles, model3col)
            AD_IKK = ext_tanimoto(ikk_threshold, fingerprint3, y3) if fingerprint3 is not None else "N/A"
            prediction3 = IKK.predict(fingerprint3)[0] if fingerprint3 is not None else "N/A"

            fingerprint4, y4 = compute_morgan_fingerprint(smiles, model4col)
            AD_TNF = ext_tanimoto(tnf_threshold, fingerprint4, y4) if fingerprint4 is not None else "N/A"
            prediction4 = TNF.predict(fingerprint4)[0] if fingerprint4 is not None else "N/A"

            # Append results for each SMILES
            results.append({
                "SMILES": smiles,
                "TGF Prediction": "Active" if prediction1 == 1 else "Inactive",
                "TGF AD": AD_TGF,
                "PDGF Prediction": "Active" if prediction2 == 1 else "Inactive",
                "PDGF AD": AD_PDG,
                "IKK Prediction": "Active" if prediction3 == 1 else "Inactive",
                "IKK AD": AD_IKK,
                "TNF Prediction": "Active" if prediction4 == 1 else "Inactive",
                "TNF AD": AD_TNF,
            })
        except Exception as e:
            st.error(f"Error processing SMILES {smiles}: {e}")

    # Convert results to a DataFrame and display as a table
    results_df = pd.DataFrame(results)
    st.subheader("Predictions Table")
    st.dataframe(results_df)  # Use st.table(results_df) for a static table

with tab2:
    st.header("TGF-Beta 1 Model")
    st.write("### Model Overview")
    st.write("### Pairwise Tanimoto Similarity Distribution")
    st.image(tantgf)
    st.write("### t-SNE Visualization")
    st.image(tsnetgf)
    st.write("### Model Performance Metrics")
    st.write("# Model: Random Forest Classifier")
    st.write("# FP: Morgan")
    st.image(mettgf)
    st.write("### Comparative Analysis")
    st.image(comtgf)

with tab3:
       st.header("PDGF-Beta Model")
       st.write("### Model Overview")
       st.write("### Pairwise Tanimoto Similarity Distribution")
       st.image(tanpdgf)
       st.write("### t-SNE Visualization")
       st.image(tsnepdgf)
       st.write("### Model Performance Metrics")
       st.write("# Model: HistGradientBoosting Classifier")
       st.write("# FP: Morgan")
       st.image(metpdgf)
       st.write("### Comparative Analysis")
       st.image(compdgf)

with tab4:
       st.header("IKKB Model")
       st.write("### Model Overview")
       st.write("### Pairwise Tanimoto Similarity Distribution")
       st.image(tanikk)
       st.write("### t-SNE Visualization")
       st.image(tsneikk)
       st.write("### Model Performance Metrics")
       st.write("# Model: Random Forest Classifier")
       st.write("# FP: Morgan")
       st.image(metikk)
       st.write("### Comparative Analysis")
       st.image(comikk)

with tab5:
       st.header("TNF-Alpha Model")
       st.write("### Model Overview")
       st.write("### Pairwise Tanimoto Similarity Distribution")
       st.image(tantnf)
       st.write("### t-SNE Visualization")
       st.image(tsnetnf)
       st.write("### Model Performance Metrics")
       st.write("# Model: SVM Classifier")
       st.write("# FP: Morgan")
       st.image(mettnf)
       st.write("### Comparative Analysis")
       st.image(comtnf)
