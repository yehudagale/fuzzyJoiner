from embeddings import KazumaCharEmbedding
k = KazumaCharEmbedding()
<<<<<<< HEAD
for w in ['canada', 'vancouver', 'toronto']:
=======

k = KazumaCharEmbedding()
for w in ['canada', 'vancouver', 'toronto', 'BMT3325']:
>>>>>>> 19322bd526f4aa868e980086fd4449c9da3b1b7e
    print('embedding {}'.format(w))
    print(k.emb(w))
