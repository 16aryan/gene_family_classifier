import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_kmers(seq, k=3):
    return " ".join([seq[i:i+k] for i in range(len(seq) - k + 1)])

def preprocess_data(csv_path, k=3):
    # Read and clean data
    df = pd.read_csv(csv_path).dropna()

    # Rename columns for consistency (optional)
    df.rename(columns={'Sequence': 'sequence', 'Label': 'gene_family'}, inplace=True)

    # Encode DNA sequences into k-mers
    df['encoded_sequence'] = df['sequence'].apply(lambda x: encode_kmers(x, k))

    # Encode gene family labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['gene_family'])

    return df, label_encoder