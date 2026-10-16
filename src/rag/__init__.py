"""Retrieval-augmented question answering on top of the forecasting outputs.

    documents.py         load .md/.txt/.pdf files and split them into chunks
    index.py             TF-IDF search index over the chunks (saved in artifacts/rag)
    forecast_context.py  plain-text summary of outputs/ (metrics + latest forecast)
    ask.py               CLI: retrieve context, ask Claude, print the answer with sources
    ingest.py            CLI: (re)build the index

The forecasting code never imports this package; it only reads the files that code writes.
"""
