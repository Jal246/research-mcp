from research_mcp import similar
from research_mcp.extract import Page
from research_mcp.search import Result

PAGE = Page("https://www.blog.example/rsa", "web page",
            "RSA picks two primes. The primes are secret; their product is the modulus. "
            "See [primes](https://example.org/primes) and https://example.org/rsa for more. "
            "RSA encryption raises the message to the public exponent modulo the modulus.",
            title="How RSA Works | Crypto Blog")


def test_site():
    assert similar.site("https://www.Example.org/a?b") == "example.org"


def test_title_query():
    assert similar.title_query("How RSA Works | Crypto Blog") == "How RSA Works"
    assert similar.title_query("RSA cryptosystem - Wikipedia") == "RSA cryptosystem"
    assert similar.title_query("Docs » Getting started with Example") == "Getting started with Example"
    assert similar.title_query("Self-hosting: a guide") == "Self-hosting: a guide"  # No separator.


def test_keywords():
    # "primes" 3 times (once as link text); "rsa" twice, not counting the one in a URL.
    assert similar.keywords(PAGE.text, 4) == ["primes", "rsa", "modulus", "picks"]


def test_keywords_skip_urls_and_stop_words():
    words = similar.keywords("The the THE [link](https://x.example/zebra) https://y.example/zebra lion", 5)
    assert words == ["link", "lion"]


def test_queries():
    assert similar.queries(PAGE) == ["How RSA Works", "primes rsa modulus picks secret product"]
    assert similar.queries(Page("https://x.example", "text", "the of and")) == []


def test_candidates():
    results = [
        Result("Itself", "https://blog.example/rsa/", "", "duckduckgo"),
        Result("Same site", "https://blog.example/primes", "", "duckduckgo"),
        Result("RSA", "https://en.wikipedia.org/wiki/RSA", "", "duckduckgo"),
        Result("RSA again", "https://en.wikipedia.org/wiki/RSA", "", "wikipedia"),
        Result("Other", "https://example.org/rsa", "", "duckduckgo"),
    ]
    kept = similar.candidates(PAGE, results, "https://blog.example/old-rsa", include_same_site=False)
    assert [r.title for r in kept] == ["RSA", "Other"]
    kept = similar.candidates(PAGE, results, "https://blog.example/old-rsa", include_same_site=True)
    assert [r.title for r in kept] == ["Same site", "RSA", "Other"]


def test_rank(embedder):
    results = [
        Result("Baking bread", "https://a.example", "Knead the dough.", "duckduckgo"),
        Result("RSA primes", "https://b.example", "The modulus is a product of primes.", "duckduckgo"),
        Result("RSA", "https://c.example", "Public-key encryption.", "duckduckgo"),
    ]
    vector = embedder.embed([similar.page_text(PAGE)])[0]
    ranked = similar.rank(vector, results, embedder, limit=5)
    assert [r.title for _, r in ranked] == ["RSA primes", "RSA"]  # Bread shares no words: dropped.
    assert ranked[0][0] > ranked[1][0]
    assert len(similar.rank(vector, results, embedder, limit=1)) == 1
    assert similar.rank(vector, [], embedder, limit=5) == []
