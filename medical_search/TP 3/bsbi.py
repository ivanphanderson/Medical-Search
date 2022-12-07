import os
import pickle
import contextlib
import heapq
import time
import math
import nltk
import lightgbm as lgb

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from letor import Letor

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.avg_doc_length = -1
        # self.letor = Letor()

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        # TODO
        ps = PorterStemmer()

        path = os.path.join(self.data_dir, block_dir_relative)
        list = []
        removed_stop_words = None
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as f:
                isi_file = f.read()
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(isi_file)
                stemmed = [ps.stem(word) for word in tokens]
                removed_stop_words = [word for word in stemmed if word not in nltk.corpus.stopwords.words('english')]
                
                for term in removed_stop_words:
                    term_id = self.term_id_map[term]
                    doc_id = self.doc_id_map[os.path.join(self.data_dir, block_dir_relative, file)]
                    list.append((term_id, doc_id))
        return list

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # TODO
        # { termid : { docid : tflist } }
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = {}
                
            if doc_id in term_dict[term_id]:
                term_dict[term_id][doc_id] += 1
            else:
                term_dict[term_id][doc_id] = 1

        for term_id in sorted(term_dict.keys()):
            postings_list = []
            tf_list = []
            for key in sorted(term_dict[term_id].keys()):
                postings_list.append(key)
                tf_list.append(term_dict[term_id][key])
            index.append(term_id, postings_list, tf_list)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        ps = PorterStemmer()
        dict1 = {}

        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(query)
        stemmed = [ps.stem(word) for word in tokens]
        query_list = [word for word in stemmed if word not in nltk.corpus.stopwords.words('english')]

        with InvertedIndexReader(self.index_name, directory=self.output_dir, postings_encoding=self.postings_encoding) as reader:
            for term in query_list:
                
                try:
                    postings_list, tf_list = reader.get_postings_list(self.term_id_map[term])
                    N = len(reader.doc_length)
                    wtq = math.log(N/len(postings_list), 10) # IDF
                    for i in range(len(postings_list)):
                        wtd = 0
                        if (tf_list[i] <= 0):
                            wtd = 0
                        else:
                            wtd = 1 + math.log(tf_list[i], 10)
                        if postings_list[i] in dict1:
                            dict1[postings_list[i]] += wtd*wtq
                        else:
                            dict1[postings_list[i]] = wtd*wtq
                except:
                    continue
 
        list_of_did_and_tfidf = list(dict1.items())
        list_of_did_and_tfidf.sort(key=lambda x: x[1],reverse=True)
        
        final_result = []
        for i in range(min(k, len(list_of_did_and_tfidf))):
            final_result.append((list_of_did_and_tfidf[i][1],self.doc_id_map[list_of_did_and_tfidf[i][0]]))
        return final_result

    def calculate_average_doc_length(self, doc_length_dict: dict):
        sum = 0
        for val in doc_length_dict.values():
            sum += val
        return sum/len(doc_length_dict)


    def retrieve_bm25(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        ps = PorterStemmer()

        dict1 = {}
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(query)
        stemmed = [ps.stem(word) for word in tokens]
        query_list = [word for word in stemmed if word not in nltk.corpus.stopwords.words('english')]
        k1 = 1.6
        b = 0.75
        
        with InvertedIndexReader(self.index_name, directory=self.output_dir, postings_encoding=self.postings_encoding) as reader:
            for term in query_list:
                try:
                    postings_list, tf_list = reader.get_postings_list(self.term_id_map[term])
                    N = len(reader.doc_length)
                    wtq = math.log(N/len(postings_list), 10)
                    if self.avg_doc_length == -1:
                        self.avg_doc_length = self.calculate_average_doc_length(reader.doc_length)
                    
                    for i in range(len(postings_list)):
                        normalization = (1-b)+b*((reader.doc_length[postings_list[i]])/self.avg_doc_length)
                        okapibm25 = wtq*(k1+1)*tf_list[i]/((k1*normalization)+tf_list[i])
                        if postings_list[i] in dict1:
                            dict1[postings_list[i]] += okapibm25
                        else:
                            dict1[postings_list[i]] = okapibm25
                except:
                    continue
 
        list_of_did_and_tfidf = list(dict1.items())
        list_of_did_and_tfidf.sort(key=lambda x: x[1],reverse=True)
        
        final_result = []
        for i in range(min(k, len(list_of_did_and_tfidf))):
            final_result.append((list_of_did_and_tfidf[i][1],self.doc_id_map[list_of_did_and_tfidf[i][0]]))
        return final_result

    # def retrieve_bm25_then_letor(self, query, k=10):
    #     # Membaca model lgb yang sudah ditrain untuk menghemat waktu (tidak perlu train ulang tiap query dijalankan)
    #     if os.path.exists('trained_letor.txt'):
    #         ranker = lgb.Booster(model_file='trained_letor.txt')
    #     else:   
    #         ranker = self.letor.trainRanker()

    #     documents = []
    #     doc_name = []
    #     for (score, doc) in self.retrieve_bm25(query, k=k):

    #         f = open(doc, "r")
    #         doc_content = f.read()
    #         documents.append(self.letor.features(query.split(), doc_content.split()))
    #         doc_name.append(doc)
    #         f.close()

    #     scores = self.letor.reranker(documents, ranker)

    #     scores_did = [x for x in zip(scores, doc_name)]
    #     sorted_scores_did = sorted(scores_did, reverse = True)

    #     return sorted_scores_did

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":
    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
