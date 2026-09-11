"""Recover front-list page numbers from a rendered thesis PDF."""
from pathlib import Path
import argparse
import json
import re
import importlib.util
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]

def normalize(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf',type=Path)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('builder',ROOT/'scripts/build_dmu_thesis_docx.py')
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    pages=[p.extract_text() for p in PdfReader(args.pdf).pages]
    first=next(i for i,t in enumerate(pages) if 'Fleet dispatch is relational.' in t)
    maps={'figures':{},'tables':{},'headings':{}}
    labels={'figures':[x[0]+'.' for x in b.FIGURES], 'tables':[x[0]+':' for x in b.TABLES], 'headings':[]}
    for p in sorted((ROOT/'docs/dissertation').glob('0[1-6]_*.md')):
        for kind,payload in b.collect_blocks(p):
            if kind=='heading':
                level,text=payload
                if level==1:
                    m=re.match(r'(\d+)\.\s+(.+)',text);text=f'Chapter {m[1]}: {m[2].title()}'
                labels['headings'].append(text)
    labels['headings'].extend(['List of Publications', 'References', 'Appendices'])
    missing=[]
    for category,items in labels.items():
        for label in items:
            key=label[:-1] if category!='headings' else label
            matches=[]
            for i,t in enumerate(pages[first:],start=1):
                if category=='figures':
                    # A caption has a period after the number; in-text mentions do not.
                    found=bool(re.search(re.escape(label),t))
                elif category=='tables':
                    found=bool(re.search(re.escape(label),t))
                elif category=='headings' and label.startswith('Chapter '):
                    found=any(normalize(label)==normalize(line) for line in t.splitlines())
                else:
                    found=normalize(label) in normalize(t)
                if found:matches.append(i)
            if not matches:missing.append(label)
            else:maps[category][key]=str(matches[0])
    if missing:raise RuntimeError(f'Unresolved page labels: {missing}')
    out=ROOT/'docs/dissertation/rendered_page_map.json'
    previous=json.loads(out.read_text()) if out.exists() else None
    out.write_text(json.dumps(maps,indent=2)+'\n')
    print(json.dumps({'pdf_pages':len(pages),'front_pages':first,'labels':{k:len(v) for k,v in maps.items()},'changed':previous!=maps}))

if __name__=='__main__':main()
