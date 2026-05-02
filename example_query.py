"""
Exemplo: total de pessoas que passaram entre dois timestamps.

Uso:
    python example_query.py counts.json 2026-05-02T10:00:00+00:00 2026-05-02T11:00:00+00:00
"""

import sys
from datetime import datetime

from analytics import count_in_interval
from storage import SnapshotStore


def main():
    if len(sys.argv) != 4:
        print("uso: python example_query.py <storage.json> <start_iso> <end_iso>")
        sys.exit(1)

    path, start_s, end_s = sys.argv[1], sys.argv[2], sys.argv[3]
    store = SnapshotStore(path)
    result = count_in_interval(store, datetime.fromisoformat(start_s), datetime.fromisoformat(end_s))

    print(f"Intervalo: {start_s}  ->  {end_s}")
    print(f"  Entradas: {result['in']}")
    print(f"  Saídas:   {result['out']}")
    print(f"  Total:    {result['total']}")


if __name__ == "__main__":
    main()
