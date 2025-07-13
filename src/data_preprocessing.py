import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_kmers(sequence, k=3):
    """Converts a DNA sequence into k-mers."""
    return ' '.join(sequence[i:i+k] for i in range(len(sequence) - k + 1))

def preprocess_data(csv_path, k=3):
    """Loads the CSV, encodes sequences and labels."""
    df = pd.read_csv(csv_path).dropna()

    # Encode gene family labels to numeric
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['gene_family'])

    # Encode DNA sequences into k-mers
    df['encoded_sequence'] = df['sequence'].apply(lambda seq: encode_kmers(seq, k))

    return df, label_encoder
