from app.chunking import chunk_text
from app.config import CHUNK_SIZE


def test_empty_text_gives_no_chunks():
    assert chunk_text("", "a.txt") == []
    assert chunk_text("   \n\n  ", "a.txt") == []


def test_short_paragraphs_are_merged_into_one_chunk():
    chunks = chunk_text("First paragraph.\n\nSecond paragraph.", "a.txt")
    assert len(chunks) == 1
    assert chunks[0].text == "First paragraph.\n\nSecond paragraph."
    assert chunks[0].metadata.file_name == "a.txt"
    assert chunks[0].metadata.chunk_index == 0


def test_long_text_is_split_within_chunk_size_with_sequential_indices():
    sentence = "This sentence is about gradient descent and learning rates. "
    text = "\n\n".join(sentence * 6 for _ in range(5))
    chunks = chunk_text(text, "long.txt")
    assert len(chunks) > 1
    assert all(len(c.text) <= CHUNK_SIZE for c in chunks)
    assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))


def test_oversized_paragraph_breaks_on_sentence_boundary():
    paragraph = " ".join(f"Sentence number {i} ends here." for i in range(60))
    chunks = chunk_text(paragraph, "p.txt")
    assert len(chunks) > 1
    assert all(c.text.rstrip().endswith(".") for c in chunks[:-1])
