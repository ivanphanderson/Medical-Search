import lightgbm as lgb
import numpy as np
import random

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine


class Letor:
  def __init__(self):
    self.NUM_NEGATIVES = 1
    self.NUM_LATENT_TOPICS = 200

    self.documents = self.getDocuments()
    self.queries = self.getQueries()

    self.group_qid_count, self.dataset = self.createDataset(self.documents, self.queries)
    self.dictionary, self.model = self.createBoW()

  def getDocuments(self):
    documents = {}
    with open("nfcorpus/train.docs") as file:
      for line in file:
        doc_id, content = line.split("\t")
        documents[doc_id] = content.split()
    return documents

  def getQueries(self):
    queries = {}
    with open("nfcorpus/train.vid-desc.queries", encoding='utf-8') as file:
      for line in file:
        q_id, content = line.split("\t")
        queries[q_id] = content.split()
    return queries

  def createDataset(self, documents, queries):
    q_docs_rel = {} # grouping by q_id terlebih dahulu
    with open("nfcorpus/train.3-2-1.qrel") as file:
      for line in file:
        q_id, _, doc_id, rel = line.split("\t")
        if (q_id in queries) and (doc_id in documents):
          if q_id not in q_docs_rel:
            q_docs_rel[q_id] = []
          q_docs_rel[q_id].append((doc_id, int(rel)))

    # group_qid_count untuk model LGBMRanker
    group_qid_count = []
    dataset = []
    for q_id in q_docs_rel:
      docs_rels = q_docs_rel[q_id]
      group_qid_count.append(len(docs_rels) + self.NUM_NEGATIVES)
      for doc_id, rel in docs_rels:
        dataset.append((queries[q_id], documents[doc_id], rel))
      # tambahkan satu negative (random sampling saja dari documents)
      dataset.append((queries[q_id], random.choice(list(documents.values())), 0))
    return group_qid_count, dataset

  def createBoW(self):
    dictionary = Dictionary()
    bow_corpus = [dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]
    model = LsiModel(bow_corpus, num_topics = self.NUM_LATENT_TOPICS) # 200 latent topics
    return (dictionary, model)

  # test melihat representasi vector dari sebuah dokumen & query
  def vector_rep(self, text):
    rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
    return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS

  def features(self, query, doc):
    v_q = self.vector_rep(query)
    v_d = self.vector_rep(doc)
    q = set(query)
    d = set(doc)
    cosine_dist = cosine(v_q, v_d)
    jaccard = len(q & d) / len(q | d)
    return v_q + v_d + [jaccard] + [cosine_dist]

  def getPointWiseRep(self, dataset):
    X = []
    Y = []
    for (query, doc, rel) in dataset:
      X.append(self.features(query, doc))
      Y.append(rel)

    # ubah X dan Y ke format numpy array
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

  def trainRanker(self):
    X, Y = self.getPointWiseRep(self.dataset) 

    ranker = lgb.LGBMRanker(
                        objective="lambdarank",
                        boosting_type = "gbdt",
                        n_estimators = 100,
                        importance_type = "gain",
                        metric = "ndcg",
                        num_leaves = 40,
                        learning_rate = 0.02,
                        max_depth = -1)

    # di contoh kali ini, kita tidak menggunakan validation set
    # jika ada yang ingin menggunakan validation set, silakan saja
    ranker.fit(X, Y,
              group = self.group_qid_count,
              verbose = 10)
    ranker.booster_.save_model('trained_letor.txt')
    return ranker
  

  def reranker(self, X_unseen, ranker):
    return ranker.predict(X_unseen)