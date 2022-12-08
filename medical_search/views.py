from django.shortcuts import render
from .TP3.search import search_bm25

# Create your views here.
def search(request):
    # user_stock = Stock.objects.filter(holder=request.user)
    if (request.method == "POST"):
        document_content = search_bm25(request.POST.get('query'))
        response = {'document_content': document_content}
        return render(request, 'index.html', response)
    document_content = search_bm25('alkylated with radioactive iodoacetate')
    print(document_content)
    # response = {}
    response = {'document_content': document_content}
    return render(request, 'index.html', response)
