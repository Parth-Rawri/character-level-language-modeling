Character-level language modeling using a Transformer (decoder only). 
So far the repo only supports the pre-training phase. 

Code summary: Read data, determine vocabulary, define functions to encode/decode text to integers and vice versa, encode entire dataset, split dataset into train/val, build a dataloader, build and train a baseline bigram model. Add positional encoding, construct single head attention, construct multiple head attention, add feed-forwad network, construct a block comprising of MHA and MLP, add residual connections, add layer normalization. 

Credits: Andrej Karpathy, Stanford CS229S