import fitz  # PyMuPDF
import re

def extract_clean_sections(pdf_files):
    documents = {}
    for pdf in pdf_files:
        doc = fitz.open(input_dir+pdf)
        text = ""
        for page in doc:
            text += page.get_text()

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Regex pattern to extract sections like 4.7.2, 4.7.3 etc. and their content
        pattern = r'(?m)^\s*(\d+(?:\.\d+)+)\s+(.*?)(?=^\s*\d+(?:\.\d+)+\s+|\Z)'  # matches 4.7.2 with content

        matches = re.finditer(pattern, text, re.DOTALL)

        processed_text = ""
        for match in matches:
            section_num = match.group(1)
            content = match.group(2)

            # Clean: join all lines in the section to make a single line
            single_line_content = ' '.join(line.strip() for line in content.splitlines() if line.strip())
            processed_text += f"{section_num} {single_line_content}\n"

        documents[pdf] = processed_text
    return documents

# Example usage
pdf_files = ['Janani suraksha yojana.pdf', 'SukanyaSamriddhiAccountSchemeRule.pdf', 'PMMVY.pdf']
input_dir = 'pdfs/'
output_dir = 'data/'
documents = extract_clean_sections(pdf_files)

# Optional: Save to a .txt file
for filename, content in documents.items():
    output_path = f"{output_dir}{filename}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Saved: {output_path}")
