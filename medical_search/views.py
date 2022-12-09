import os
import time
from django.shortcuts import render
from .TP3.search import search_bm25

# Create your views here.
def search(request):
    start = time.time()
    page = 1
    if request.GET.get('page') != None:
        page = request.GET.get('page')
    
    query = ''
    if request.GET.get('query') != None:
        query = request.GET.get('query')
    else:
        response = {'message': 'Welcome to Medical Search!\nType something in the search box and get the result!'}
        return render(request, 'index.html', response)

    document_path_and_content = search_bm25(query)

    if len(document_path_and_content) == 0:
        response = {'message': 'Your search did not match any documents'}
        return render(request, 'index.html', response)
    
    total_page = len(document_path_and_content)//10 + 1

    if not page.isnumeric():
        page = 1          
    elif (int(page) > total_page):
        page = total_page
    elif (int(page) < 1):
        page = 1

    first_index = (int(page)-1)*10
    last_index = (int(page)-1)*10 + 10 if ((int(page)-1)*10 + 10 < len(document_path_and_content)) else len(document_path_and_content)
    end = time.time()
    response = {
        'document_path_and_content': document_path_and_content[first_index:last_index],
        'curr_page': int(page),
        'prev_page': int(page)-1,
        'next_page': int(page)+1,
        'total_page': total_page,
        'total_docs': len(document_path_and_content),
        'query': query,
        'time': "{:.2f}".format(end-start),
    }
    return render(request, 'index.html', response)

def view_content(request, path1, path2, path3):
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
    doc_path = os.path.join(PARENT_DIR, path1, path2, path3)

    doc_content = ""
    with open(os.path.join(doc_path), 'r') as f:
        doc_content = f.read()
    response = {'doc_content': doc_content, 'title': path3}
    return render(request, 'content.html', response)

