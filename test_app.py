from app import chunk_text

def test_chunk_text_basic():
    text = "one two three four five six"
    chunks = chunk_text(text, chunk_size=2)
    assert len(chunks) == 3
    assert chunks[0] == "one two"
    assert chunks[1] == "three four"
    assert chunks[2] == "five six"


from app import Document

def test_document_missing_file():
    doc = Document("this_file_does_not_exist.txt")
    assert doc.content is None
    assert doc.word_count() == 0
    