from embeddings import KazumaCharEmbedding
k = KazumaCharEmbedding()

k = KazumaCharEmbedding()
for w in ['canada', 'vancouver', 'toronto', 'BMT3325']:
    print('embedding {}'.format(w))
    print(k.emb(w))
