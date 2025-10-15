# File: exporter.py
from fpdf import FPDF
import tempfile
import os

def export_to_pdf(markdown_content: str, output_path: str = None) -> str:
    """Convert Markdown to simple PDF using FPDF (basic text rendering)."""
    if output_path is None:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, 'report.pdf')
    
    # Basic Markdown to plain text conversion (remove headers, lists to bullets)
    lines = markdown_content.split('\n')
    plain_text = []
    for line in lines:
        if line.startswith('# '):
            plain_text.append(line[2:] + ' (Heading)')
        elif line.startswith('## '):
            plain_text.append(line[3:] + ' (Subheading)')
        elif line.startswith('- '):
            plain_text.append('• ' + line[2:])
        elif line.startswith('* '):
            plain_text.append('• ' + line[2:])
        else:
            plain_text.append(line)
    
    full_text = '\n'.join(plain_text)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add text (wrap manually if needed)
    for line in plain_text:
        pdf.cell(200, 10, txt=line, ln=True)
    
    pdf.output(output_path)
    return output_path