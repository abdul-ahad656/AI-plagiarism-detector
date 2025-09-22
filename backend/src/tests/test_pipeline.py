from src.pipeline import process_document_pair

def test_similarity():
    doc1 = "Machine learning is a method of data analysis."
    doc2 = "Data analysis can be achieved using machine learning techniques."
    score = process_document_pair(doc1, doc2)
    assert score > 0.7, f"Expected high similarity, got {score}"
