import fitz  # PyMuPDF
from markdownify import markdownify as md

def pdf_to_markdown(pdf_path, md_path):
    # Öffne das PDF-Dokument
    document = fitz.open(pdf_path)
    
    # Erstelle eine leere Liste für den Text
    markdown_text = ""
    
    # Iteriere durch die Seiten des PDF-Dokuments
    for page in document:
        # Extrahiere den Text der Seite
        text = page.get_text()
        
        # Füge den Text zur Markdown-Liste hinzu
        markdown_text += text + "\n\n"  # Füge zwei neue Zeilen hinzu, um Absätze zu trennen
    
    # Schließe das Dokument
    document.close()
    
    # Optional: Wandle den extrahierten Text in Markdown um (falls gewünscht)
    markdown_text = md(markdown_text)
    
    # Speichere den Markdown-Text in einer .md-Datei
    with open(md_path, 'w', encoding='utf-8') as md_file:
        md_file.write(markdown_text)

# Verwende die Funktion
pdf_to_markdown('o365hb.pdf', 'o365.md')