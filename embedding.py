from annoy import AnnoyIndex
import numpy as np
import pickle
import math
import pymorphy2 as pm2


def add_pos(morph, word):
    M = {
        "ADVB": "ADV",
        "COMP": "ADV",
        "NUMR": "ADV",
        "INFN": "VERB",
        "ADJS": "NOUN",
        "ADJF":"ADJ"
    }
    res = morph.parse(word)[0]
    suff = res.tag.POS
    if suff == None:
        suff = ""

    if suff in M:
        suff = M[suff]
    return word + "_" + suff


class PreTrainedEmbeddings(object):
    """ A wrapper around pre-trained word vectors and their use """

    def __init__(self, word_to_index, word_vectors, build_index=True, index_file="index.bin"):
        """
        Args:
            word_to_index (dict): mapping from word to integers
            word_vectors (list of numpy arrays)
        """
        self.word_to_index = word_to_index
        self.word_vectors = word_vectors
        self.dim = len(word_vectors[0])
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}

        self.index = AnnoyIndex(len(word_vectors[0]), metric='euclidean')
        self.fname = index_file

        if build_index:
            print("Building Index!")
            for _, i in self.word_to_index.items():
                self.index.add_item(i, self.word_vectors[i])
            self.index.build(50)
            self.index.save(self.fname)
            print("Finished!")
        else:
            print("Load Index")
            self.load_index()

    def load_index(self):

        self.index.load(self.fname)

    @classmethod
    def from_embeddings_file(cls, embedding_file, build_index, index_file):
        """Instantiate from pre-trained vector file.

        Vector file should be of the format:
            word0 x0_0 x0_1 x0_2 x0_3 ... x0_N
            word1 x1_0 x1_1 x1_2 x1_3 ... x1_N

        Args:
            embedding_file (str): location of the file
        Returns:
            instance of PretrainedEmbeddigns
        """
        word_to_index = {}
        word_vectors = []
        ind = 0

        with open(embedding_file) as fp:
            for line in fp.readlines()[1:]:
                line = line.split(" ")
                word = line[0]  # .split("_")[0]
                # print(word)
                # print(word.split("_")[0])
                vec = np.array([float(x) for x in line[1:]])

                word_to_index[word] = ind  # len(word_to_index)
                ind += 1
                word_vectors.append(vec)

        return cls(word_to_index, word_vectors, build_index, index_file)

    def get_word_vector(self,word):
        morph = pm2.MorphAnalyzer()

        res = morph.parse(word)[0]
        # res = res.normalized
        # print(word,res.word)
        word = add_pos(morph, res.word)
        vec = self.get_embedding(word)
        return res.word, vec
    def get_embedding(self, word):
        """
        Args:
            word (str)
        Returns
            an embedding (numpy.ndarray)
        """
        # print(word)
        try:
            vec = self.word_vectors[self.word_to_index[word]]
            # print(vec)
            return vec
        except:

            return np.random.rand(self.dim) #np.zeros((self.dim),dtype=float)#None

    def get_closest_to_vector(self, vector, n=1):
        """Given a vector, return its n nearest neighbors

        Args:
            vector (np.ndarray): should match the size of the vectors
                in the Annoy index
            n (int): the number of neighbors to return
        Returns:
            [str, str, ...]: words that are nearest to the given vector.
                The words are not ordered by distance
        """
        nn_indices = self.index.get_nns_by_vector(vector, n)
        return [self.index_to_word[neighbor] for neighbor in nn_indices]

    def closest(self, word):
        morph = pm2.MorphAnalyzer()
        word = add_pos(morph, word)
        vec = self.get_embedding(word)

        closest_words = self.get_closest_to_vector(vec, n=4)
        closest_words = [w.split("_")[0] for w in closest_words]
        return closest_words

    def compute_place(self, word1, word2, word3, v1, v2):
        morph = pm2.MorphAnalyzer()
        word1 = add_pos(morph, word1)
        word2 = add_pos(morph, word2)
        word3 = add_pos(morph, word3)
        v1 = add_pos(morph, v1)
        v2 = add_pos(morph, v2)
        # a = 1
        # b = -4
        # c = 1

        a = 5
        b = -1
        c = 8

        print(word1, word2, word3, v1, v2)
        vec1 = self.get_embedding(word1)
        vec2 = self.get_embedding(word2)
        vec3 = self.get_embedding(word3)
        vec4 = self.get_embedding(v1)
        vec5 = self.get_embedding(v2)
        # print(vec1)
        # print(vec2)
        # print(vec3)
        # print(vec4)
        # print(vec5)
        # now compute the fourth word's embedding!
        spatial_relationship1 = vec1 - vec4
        spatial_relationship2 = vec2 - vec5
        vec6 = (vec3 + a * spatial_relationship1 + b * spatial_relationship2) / c

        closest_words = self.get_closest_to_vector(vec6, n=4)
        existing_words = set([word1, word2, word3, v1, v2])
        closest_words = [word for word in closest_words
                         if word not in existing_words]

        if len(closest_words) == 0:
            print("Could not find nearest neighbors for the computed vector!")
            return
        word1 = word1.split("_")[0]
        word2 = word2.split("_")[0]
        word3 = word3.split("_")[0]
        v1 = v1.split("_")[0]
        v2 = v2.split("_")[0]

        for word6 in closest_words:
            word6 = word6.split("_")[0]
            print(f"{word1} & {word2}& {word3} & {v1} & {v2}& {word6}")  # .format(word1, word2, word3, v1, v2, word6))

    def train_pairs(self, fname, dim):
        f = open(fname)
        morph = pm2.MorphAnalyzer()
        lines = f.readlines()
        N = len(lines)
        V = np.zeros((N, 2, dim))
        i = 0
        for line in lines:
            word1, word2 = line.split()
            word1 = add_pos(morph, word1)
            word2 = add_pos(morph, word2)
            vec1 = self.get_embedding(word1)
            vec2 = self.get_embedding(word2)
            V[i, 0, :] = vec1
            V[i, 1, :] = vec2
        U = V[1:, :, :] - V[:-1, :, :]
        X = U[:, 0, :]
        Y = U[:, 1, :]
        a = np.trace(X.dot(X.T))
        b = np.trace(X.dot(Y.T))
        return b / a

    def compute_and_print_analogy(self, word1, word2, word3):
        """Prints the solutions to analogies using word embeddings

        Analogies are word1 is to word2 as word3 is to __
        This method will print: word1 : word2 :: word3 : word4

        Args:
            word1 (str)
            word2 (str)
            word3 (str)
        """
        morph = pm2.MorphAnalyzer()
        word1 = add_pos(morph, word1)
        word2 = add_pos(morph, word2)
        word3 = add_pos(morph, word3)
        print(word1, word2, word3)
        vec1 = self.get_embedding(word1)
        vec2 = self.get_embedding(word2)
        vec3 = self.get_embedding(word3)

        # now compute the fourth word's embedding!
        # a = 0.07
        a = 1.7
        spatial_relationship = vec2 - a * vec1
        vec4 = a * vec3 + spatial_relationship

        closest_words = self.get_closest_to_vector(vec4, n=7)
        existing_words = set([word1, word2, word3])
        closest_words = [word for word in closest_words
                         if word not in existing_words]

        if len(closest_words) == 0:
            print("Could not find nearest neighbors for the computed vector!")
            return
        word1 = word1.split("_")[0]
        word2 = word2.split("_")[0]
        word3 = word3.split("_")[0]

        for word4 in closest_words:
            word4 = word4.split("_")[0]
            print("{} & {} & {} & {}".format(word1, word2, word3, word4))

    def dist(self,word1, word2):
        morph = pm2.MorphAnalyzer()
        word1 = add_pos(morph,word1)
        word2 = add_pos(morph,word2)
        vec1 = self.get_embedding(word1)
        vec2 = self.get_embedding(word2)
        dv = vec1 - vec2
        return math.sqrt(dv.dot(dv))

    def save(self, fname):
        pickle.dump(self, open(fname, "wb"))

    @staticmethod
    def load(fname):
        obj = pickle.load(open(fname), "rb")
        return obj


