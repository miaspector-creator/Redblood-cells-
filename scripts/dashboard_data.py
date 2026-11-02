"""Read/merge dashboard/data.json, the single file the HTML dashboard polls."""
import json
import os
from datetime import datetime, timezone

from common import DASHBOARD

PATH = DASHBOARD / "data.json"


def load():
    return json.loads(PATH.read_text()) if PATH.exists() else {}


def update(section, value):
    data = load()
    data[section] = value
    data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, PATH)
