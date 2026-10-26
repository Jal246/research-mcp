import pytest

from research_mcp.extract import Page
from research_mcp.library import Library, LibraryError, chunk, clean_tags

RAFT = Page("https://raft.example/paper", "web page",
            "Raft is a consensus algorithm.\n\nServers elect a leader by voting in terms; a "
            "candidate needs votes from a majority of servers to become leader.",
            title="In Search of an Understandable Consensus Algorithm")
COFFEE = Page("https://coffee.example/brew", "web page",
              "Grind the beans finely and bloom the coffee for thirty seconds before pouring.",
              title="Brewing Coffee")


@pytest.fixture
def library(tmp_path, embedder):
    return Library(tmp_path / "library.db", embedder)


def test_chunk_keeps_short_text_whole():
    assert chunk("One paragraph.\n\nAnother.") == [(0, "One paragraph.\n\nAnother.")]


def test_chunk_joins_paragraphs_while_they_fit():
    text = "\n\n".join(["a" * 40, "b" * 40, "c" * 40])
    assert chunk(text, max_chars=90) == [(0, "a" * 40 + "\n\n" + "b" * 40), (84, "c" * 40)]


def test_chunk_cuts_huge_paragraphs_at_spaces():
    text = " ".join(["word"] * 50)  # 249 characters.
    pieces = chunk(text, max_chars=100)
    assert all(len(piece) <= 100 for _, piece in pieces)
    assert all(not piece.startswith(" ") and text[start:].startswith(piece) for start, piece in pieces)
    assert " ".join(piece for _, piece in pieces) == text


def test_chunk_cuts_text_without_spaces():
    assert chunk("x" * 250, max_chars=100) == [(0, "x" * 100), (100, "x" * 100), (200, "x" * 50)]


def test_clean_tags():
    assert clean_tags(["ML Papers", " thesis ", "ml-papers", "", "  "]) == ["ml-papers", "thesis"]


def test_save_and_search(library):
    document, existed = library.save(RAFT, note="  Best intro to Raft ", tags=["Consensus"], now=100)
    assert not existed
    assert (document.url, document.note, document.tags, document.chars) == (
        RAFT.url, "Best intro to Raft", ("consensus",), len(RAFT.text))
    library.save(COFFEE)

    hits = library.search("how do servers elect a leader", limit=5)
    assert [hit.document.url for hit in hits] == [RAFT.url]  # Coffee is unrelated, so dropped.
    assert hits[0].passage == RAFT.text and hits[0].start == 0
    assert library.search("coffee grind", limit=5)[0].document.title == "Brewing Coffee"


def test_search_shows_the_best_passage(library):
    page = Page("https://long.example/", "web page", "\n\n".join(
        ["Bananas are yellow fruit. " * 30, "Leader election needs a majority of votes. " * 20]))
    library.save(page)
    hit, = library.search("leader election votes", limit=5)
    assert hit.passage.startswith("Leader election") and page.text[hit.start:].startswith(hit.passage)


def test_search_by_tag(library):
    library.save(RAFT, tags=["consensus"])
    library.save(COFFEE, tags=["hobby"])
    assert library.search("leader", limit=5, tag="hobby") == []
    assert len(library.search("leader", limit=5, tag="Consensus")) == 1


def test_saving_again_updates(library):
    library.save(RAFT, note="first note", tags=["a"], now=100)
    newer = Page("http://www.raft.example/paper/", "web page", "Raft, updated.", title="Raft v2")
    document, existed = library.save(newer, tags=["b"], now=200)
    assert existed
    assert (document.title, document.note, document.tags, document.saved_at) == (
        "Raft v2", "first note", ("a", "b"), 200)
    assert len(library.documents()) == 1
    assert library.search("raft updated", limit=5)[0].passage == "Raft, updated."


def test_documents_and_tags(library):
    library.save(RAFT, tags=["consensus", "thesis"], now=100)
    library.save(COFFEE, tags=["thesis"], now=200)
    assert [d.title for d in library.documents()] == ["Brewing Coffee", RAFT.title]  # Newest first.
    assert [d.title for d in library.documents("consensus")] == [RAFT.title]
    assert library.tags() == {"thesis": 2, "consensus": 1}


def test_remove(library):
    library.save(RAFT, tags=["consensus"])
    assert library.remove("http://raft.example/paper/").url == RAFT.url
    assert library.documents() == [] and library.tags() == {}
    assert library.search("leader", limit=5) == []
    with pytest.raises(LibraryError, match="isn't in the library"):
        library.remove(RAFT.url)


def test_closest_can_leave_out_a_page(library, embedder):
    library.save(RAFT)
    vector = embedder.embed([RAFT.text])[0]
    assert [hit.document.url for hit in library.closest(vector, 5)] == [RAFT.url]
    assert library.closest(vector, 5, exclude_url=RAFT.url) == []


def test_pages_from_another_model_are_reembedded(library, embedder, tmp_path):
    library.save(RAFT)

    class NewModel(type(embedder)):
        name = "newer"

    newer = NewModel()
    assert Library(tmp_path / "library.db", newer).search("leader", limit=5)
    assert newer.embedded == [f"{RAFT.title}\n\n{RAFT.text}"]
    Library(tmp_path / "library.db", newer).search("leader", limit=5)
    assert len(newer.embedded) == 1  # Only once.
