# Codebase Chatbot

set up:
```
python install -r requirements.txt
```

to run the application, simply run
``` bash
python -m chainlit run model.py
```

### Your custom data:

1. Place your data in `data/` folder
1. if your data is pdf file, run `python ingest_pdf.py`
2. if your data is an entire codebase, run `python ingest_codebase.py`

This will store your ingested data locally in your folder.

### ingesting:
#### ingest_codebase.py
add the file types you want to ingest in `extensions = ['.py', '.java', '.js', '.ts' , '.md']`
