#!/usr/bin/env python3
"""
Embed queries for search protocol
"""

import os
import sys

from arc_tiptoe.search.query_processor import main

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


if __name__ == "__main__":
    print("starting embed_text_script", file=sys.stderr)
    main()
