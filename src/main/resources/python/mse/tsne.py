import numpy as np
from sklearn.manifold import MDS
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection

def reduce_dimensions(space, n, metric='cosine'):
    # model = MDS(n_components=n, n_jobs=20)
    # rp = SparseRandomProjection(20)
    # rp = GaussianRandomProjection(20)
    model = TSNE(n_jobs=8, n_components=n) # , learning_rate=1000, n_iter=250)#, metric=metric)
    return model.fit_transform(space)
    # return model.fit_transform(rp.fit_transform(space))


if __name__ == "__main__":
    import sys
    import loaders
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    space, idx, ridx = loaders.load_space(input_file, None)
    # if space.shape[1] > 5:
    #     print("pca")
    #     pca = PCA(n_components=5)
    #     space = pca.fit_transform(space)
    print("tsne")
    space2d = reduce_dimensions(space, 2)
    f = open(output_file, 'w')
    f.write('{} {}\n'.format(len(ridx), 2))
    for i, word in enumerate(ridx):
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(space2d[i, :])))))
    f.close()
