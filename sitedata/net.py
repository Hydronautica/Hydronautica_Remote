"""Small HTTP helpers with retries (standard library only)."""

import json
import time
import urllib.request

UA = {"User-Agent": "Hydronautica-sitedata/0.1 (offshore wind site screening)"}


def get_bytes(url, timeout=120, tries=4):
    err = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as ex:  # transient service errors are common (ERDDAP 503s, THREDDS timeouts)
            err = ex
            time.sleep(2 * 2**i)
    raise err


def get_json(url, **kw):
    return json.loads(get_bytes(url, **kw).decode())
