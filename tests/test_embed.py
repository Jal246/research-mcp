import numpy as np

from research_mcp.embed import readable, unit


def test_readable():
    text = ("Espresso [machines](https://en.wikipedia.org/wiki/Espresso_machine) use pressure."
            "<sup>[\\[2\\]](https://en.wikipedia.org#cite_note-2)</sup> ![A cup](cup.jpg)\n\n  Next.")
    assert readable(text) == "Espresso [machines] use pressure.[\\[2\\]] Next."


def test_unit():
    assert np.allclose(unit(np.array([[3.0, 4.0], [0.0, 0.0]])), [[0.6, 0.8], [0.0, 0.0]])
