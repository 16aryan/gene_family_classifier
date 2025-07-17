import pandas as pd
import numpy as np

chimp = pd.read_csv('/Users/aryan/gene_family_classifier-main/data/chimpanzee_sequence.csv')
mouse = pd.read_csv('/Users/aryan/gene_family_classifier-main/data/mouse_sequence.csv')
human = pd.read_csv('/Users/aryan/gene_family_classifier-main/data/human_sequence.csv')

df = pd.concat([chimp, human, mouse], ignore_index=True)
df.to_csv("/Users/aryan/gene_family_classifier-main/data/gene_families.csv", index=False)






