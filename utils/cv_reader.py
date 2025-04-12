import re
import logging
from docx import Document
from pdfminer.high_level import extract_text

logger = logging.getLogger(__name__)

def read_resume_from_file(file_path):
    try:
        if file_path.lower().endswith('.txt'):
            with open(file_path, encoding='utf-8') as f:
                return f.read()
        elif file_path.lower().endswith('.docx'):
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_path.lower().endswith('.pdf'):
            return extract_text(file_path)
        else:
            logger.warning(f"Неподдерживаемый формат файла: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Ошибка при чтении файла {file_path}: {e}")
        return None

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    return text.lower()