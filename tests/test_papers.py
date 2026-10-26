import pytest

from research_mcp import papers
from research_mcp.papers import Paper, SearchError, arxiv_query, merge

from conftest import FIXTURES

pytestmark = pytest.mark.anyio


def paper(title: str, source: str, **fields) -> Paper:
    return Paper(title, (), None, "", f"https://{source}.example/{title}", source, **fields)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("transformers", 'all:"transformers"'),
        ("attention is all you need",
         'all:"attention is all you need" OR (all:attention AND all:all AND all:you AND all:need)'),
        ('BERT: "pre-training" (2018)', 'all:"bert pre-training 2018" OR (all:bert AND all:pre-training AND all:2018)'),
        ("the", 'all:"the"'),  # Only stop words: search for them anyway.
    ],
)
def test_arxiv_query(query, expected):
    assert arxiv_query(query) == expected


def test_arxiv_query_with_year():
    assert arxiv_query("rag", 2024) == '(all:"rag") AND submittedDate:[202401010000 TO 999912312359]'


def test_arxiv_query_needs_words():
    with pytest.raises(SearchError):
        arxiv_query("?!")


def test_parse_arxiv():
    first, second = papers.parse_arxiv((FIXTURES / "arxiv.xml").read_bytes())
    assert first == Paper(
        title="Attention Is All You Need",  # Line break in the feed tidied away.
        authors=("Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones",
                 "Aidan N. Gomez"),
        year=2017,
        abstract="The dominant sequence transduction models are based on complex recurrent or "
                 "convolutional neural networks. We propose a new simple network architecture, the "
                 "Transformer, based solely on attention mechanisms.",
        url="https://arxiv.org/abs/1706.03762",  # Without the version, so it's the latest one.
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/1706.03762",
        arxiv_id="1706.03762",
    )
    assert second.venue == "Journal of Examples 12 (2025)" and second.doi == "10.1234/example.2025"


def test_parse_arxiv_rejects_nonsense():
    with pytest.raises(SearchError, match="valid feed"):
        papers.parse_arxiv(b"<html>Oops")


async def test_semantic_scholar(fake_internet):
    first, second = await papers.search_semantic_scholar("attention", 2, year_from=2020)
    assert first.title == "Attention is All you Need"
    assert first.citations == 150_123 and first.arxiv_id == "1706.03762"
    assert first.abstract == "" and first.pdf_url is None  # null abstract, "" PDF link.
    assert second.venue is None and second.doi == "10.5555/Medicine.2021"
    assert second.pdf_url == "https://papers.example/medicine.pdf"
    assert "year=2020-" in fake_internet[0] and "limit=2" in fake_internet[0]


async def test_semantic_scholar_key_is_sent(monkeypatch):
    sent = []
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-key")

    async def spy(url, interval=0.0, headers=None):
        sent.append(headers)
        raise SearchError("stop here")

    monkeypatch.setattr(papers.polite, "get", spy)
    with pytest.raises(SearchError):
        await papers.search_semantic_scholar("attention", 2)
    assert sent[0]["x-api-key"] == "s2-key"


async def test_semantic_scholar_busy_explains_api_key():
    with pytest.raises(SearchError, match="SEMANTIC_SCHOLAR_API_KEY"):
        await papers.search_semantic_scholar("always busy", 2)


def test_merge_takes_turns_between_sources():
    a1, a2 = paper("A1", "arxiv"), paper("A2", "arxiv")
    s1 = paper("S1", "semantic_scholar")
    assert merge([[a1, a2], [s1]]) == [a1, s1, a2]


def test_merge_combines_the_same_paper():
    from_arxiv = paper("Attention Is All You Need", "arxiv", arxiv_id="1706.03762",
                       pdf_url="https://arxiv.org/pdf/1706.03762")
    from_s2 = paper("Attention is All you Need", "semantic_scholar", arxiv_id="1706.03762",
                    venue="NeurIPS", citations=150_123, doi="10.1/x")
    [combined] = merge([[from_arxiv], [from_s2]])
    assert combined.url == from_arxiv.url and combined.pdf_url == from_arxiv.pdf_url
    assert combined.venue == "NeurIPS" and combined.citations == 150_123 and combined.doi == "10.1/x"
    assert combined.source == "arxiv+semantic_scholar"


def test_merge_recognizes_papers_by_doi_or_title():
    assert len(merge([[paper("X", "arxiv", doi="10.1/ABC")], [paper("Y", "s2", doi="10.1/abc")]])) == 1
    assert len(merge([[paper("Deep Learning!", "arxiv")], [paper("deep learning", "s2")]])) == 1
    assert len(merge([[paper("", "arxiv")], [paper("", "s2")]])) == 2  # No title: can't tell.


async def test_search_papers_merges_both_sources():
    found, errors = await papers.search_papers("attention", ["arxiv", "semantic_scholar"], 5)
    assert errors == {}
    assert [p.title for p in found] == [
        "Attention Is All You Need", "Element-wise Attention Is All You Need",
        "Attention Mechanisms in Medicine",
    ]
    assert found[0].citations == 150_123 and found[0].source == "arxiv+semantic_scholar"


@pytest.mark.parametrize(
    "sort, expected",
    [
        ("newest", ["Element-wise Attention Is All You Need", "Attention Mechanisms in Medicine",
                    "Attention Is All You Need"]),
        ("most_cited", ["Attention Is All You Need", "Attention Mechanisms in Medicine",
                        "Element-wise Attention Is All You Need"]),
    ],
)
async def test_search_papers_sorting(sort, expected):
    found, _ = await papers.search_papers("attention", ["arxiv", "semantic_scholar"], 5, sort=sort)
    assert [p.title for p in found] == expected


async def test_arxiv_requests_are_spaced_out(sleeps):
    await papers.search_arxiv("attention", 5)
    await papers.search_arxiv("transformers", 5)
    assert len(sleeps) == 1 and 2.9 < sleeps[0] <= 3.0
