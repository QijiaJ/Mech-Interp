# Attention-feature experiments

Code accompanying *Understanding Attention Heads*. The repository studies a
joint relation-message decomposition of Gemma-3-1B-IT attention heads. Its main
experiment assigns one latent unit to each active query event, fits a Q/K
relation subspace `U_a` and message-innovation subspace `V_a`, and compares
ordinary PCA/K-means initialization with a label-blind product-kernel spectral
initialization.

