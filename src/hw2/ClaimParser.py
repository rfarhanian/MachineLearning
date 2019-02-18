import functools
import io
import sys
from typing import List

import numpy as np

genfromtxt_old = np.genfromtxt


# Numpy cheetsheet
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf
class ClaimParser:

    @functools.wraps(genfromtxt_old)
    def genfromtxt_py3_fixed(f, encoding="utf-8", *args, **kwargs):
        if isinstance(f, io.TextIOBase):
            if hasattr(f, "buffer") and hasattr(f.buffer, "raw") and \
                    isinstance(f.buffer.raw, io.FileIO):
                fb = f.buffer.raw
                fb.seek(f.tell())
                result = genfromtxt_old(fb, *args, **kwargs)
                f.seek(fb.tell())
            else:
                old_cursor_pos = f.tell()
                fb = io.BytesIO(bytes(f.read(), encoding=encoding))
                result = genfromtxt_old(fb, *args, **kwargs)
                f.seek(old_cursor_pos + fb.tell())
        else:
            result = genfromtxt_old(f, *args, **kwargs)
        return result

    if sys.version_info >= (3,):
        np.genfromtxt = genfromtxt_py3_fixed

    columns: List[str]

    def __init__(self, location):
        self.location = location
        self.__parse__()

    def __parse__(self):
        types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8', 'S5', 'S3', 'S3', 'S3', 'S3',
                 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3', 'S3', 'S3', 'S4', 'S14', 'S14']

        self.rows = np.genfromtxt(r'./input/claim.sample.csv', dtype=types, delimiter=',', names=True,
                                  usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                           22, 23, 24, 25, 26, 27, 28])
        self.columns = self.rows.dtype.names
        print(self.columns)
        print(self.rows.dtype)

    def get_rows(self):
        return self.rows

    def get_row_line_count(self):
        return len(self.rows)
