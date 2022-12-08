import os

from .bsbi import BSBIIndex
from .compression import VBEPostings

def search_bm25(query):
    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                            postings_encoding = VBEPostings, \
                            output_dir = 'index')
    
    documents = []
    doc_name = []
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=100):
        with open(os.path.join(doc), 'r') as f:
            # f = open(doc, "r")
            doc_content = f.read()
            # documents.append(self.letor.features(query.split(), doc_content.split()))
            documents.append(doc_content)
            doc_name.append(doc)
            # f.close()
    # for (score, doc) in BSBI_instance.retrieve_bm25(query, k=100):
    #     print(f"{doc:30} {score:>.3f}")
    print(documents)
    return documents
        


if __name__ == '__main__':
    # sebelumnya sudah dilakukan indexing
    # BSBIIndex hanya sebagai abstraksi untuk index tersebut
    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                            postings_encoding = VBEPostings, \
                            output_dir = 'index')

    queries = ["alkylated with radioactive iodoacetate", \
            "psychodrama for disturbed children", \
            "lipid metabolism in toxemia and normal pregnancy",]

    for query in queries:
        print("Query  : ", query)
        print("BM25 Results:")
        documents = []
        doc_name = []

        for (score, doc) in BSBI_instance.retrieve_bm25(query, k=100):
            print(f"{doc:30} {score:>.3f}")
        print()

        print("LETOR Result:")
        for (score, did) in BSBI_instance.retrieve_bm25_then_letor(query, k=100):
            print(f"{did:30} {score:>.3f}")
        print()