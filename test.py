from embeddings import KazumaCharEmbedding
k = KazumaCharEmbedding()
for w in ['canada', 'vancouver', 'toronto']:
    print('embedding {}'.format(w))
    print(k.emb(w))
