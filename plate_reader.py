"""Plate reader.

The easyocr.Reader initialization can be slow; we initialize it lazily.
"""

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'])
    return _reader


def read_plate(image):

    reader = _get_reader()
    results = reader.readtext(image)

    if len(results) > 0:
        return results[0][1]

    return "Unknown"