import pytest

from research_mcp import structured

from conftest import FIXTURES

URL = "https://shop.example/kettles/example-kettle-2?ref=home"


@pytest.fixture(scope="module")
def tree():
    return structured.parse((FIXTURES / "product.html").read_bytes())


def test_metadata(tree):
    assert structured.metadata(tree, URL) == {
        "title": "Example Kettle 2 | Kettle Shop",
        "language": "en",
        "description": "A Bluetooth kettle that holds 1.7 litres.",
        "og:type": "product",
        "og:image": "https://shop.example/kettle.jpg",  # The first of two.
        "twitter:card": "summary",
        "canonical": "https://shop.example/kettles/example-kettle-2",
        "feeds": [{"title": "New kettles", "url": "https://shop.example/feed.xml"}],
    }


def test_json_ld(tree):
    items = structured.json_ld(tree)  # The broken one and the string are skipped.
    assert [item["@type"] for item in items] == ["Product", "BreadcrumbList", "Organization"]
    assert items[0]["offers"]["price"] == "49.99"


def test_tables(tree):
    assert structured.tables(tree, max_rows=10) == [  # The one-column layout table is skipped.
        {"caption": "Specs", "columns": ["Part", "Value", "Value"],
         "rows": [["Capacity", "1.7", "litres"], ["Power", "2200", "watts"],
                  ["Bluetooth", "5.0 (LE)", "5.0 (LE)"]]},
        {"caption": None, "columns": ["Kettle", "Price"],
         "rows": [["Example Kettle 1", "$39"], ["Example Kettle 2", "$49.99"]]},
    ]


def test_tables_are_cut_to_max_rows(tree):
    specs = structured.tables(tree, max_rows=1)[0]
    assert specs["rows"] == [["Capacity", "1.7", "litres"]] and specs["more_rows"] == 2


def test_headings(tree):
    assert structured.headings(tree) == [
        {"level": 1, "text": "Example Kettle 2"}, {"level": 2, "text": "Specifications"},
        {"level": 3, "text": "Compared"}, {"level": 2, "text": "Control it from code"},
    ]


def test_code(tree):
    assert structured.code(tree) == [
        {"language": "python", "code": "import kettle\nkettle.boil(degrees=92)"},
        {"language": None, "code": "plain block"},
    ]


def test_links(tree):
    assert structured.links(tree, URL) == [
        {"text": "All kettles", "url": "https://shop.example/kettles/", "internal": True},
        {"text": "Maker", "url": "https://maker.example/", "internal": False},
    ]


def test_wikipedia_noise_is_dropped():
    tree = structured.parse(b"""<html><body>
        <h2>History<span class="mw-editsection">[edit]</span></h2>
        <table class="wikitable"><tr><th>City<sup class="reference">[a]</sup></th>
            <th>Area<style>.nobold{font-weight:normal}</style> (km2)</th></tr>
            <tr><td>Jakarta</td><td>664<sup id="cite_ref-1" class="reference">[14]</sup></td></tr></table>
        <table class="navbox"><tr><th>vte</th><td>Cities</td></tr></table>
        <script>var x = 1;</script></body></html>""")
    assert structured.headings(tree) == [{"level": 2, "text": "History"}]
    assert structured.tables(tree, 10) == [
        {"caption": None, "columns": ["City", "Area (km2)"], "rows": [["Jakarta", "664"]]}]


def test_long_text_is_cut(monkeypatch):
    monkeypatch.setattr(structured, "MAX_TEXT", 5)
    tree = structured.parse(b"<html><body><h1>Hello world</h1></body></html>")
    assert structured.headings(tree) == [{"level": 1, "text": "Hello …"}]


def test_extract():
    body = (FIXTURES / "product.html").read_bytes()
    data = structured.extract(URL, body, ["links", "headings", "json_ld"], max_items=2)
    assert list(data) == ["json_ld", "headings", "links", "more"]  # In KINDS order.
    assert len(data["json_ld"]) == len(data["headings"]) == 2
    assert data["more"] == {"json_ld": 1, "headings": 2}


def test_not_html():
    with pytest.raises(structured.StructuredError):
        structured.parse(b"")
