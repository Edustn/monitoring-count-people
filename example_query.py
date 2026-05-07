"""
Exemplo: total de pessoas que passaram entre dois timestamps.

Uso:
    # todas as câmeras combinadas
    python example_query.py counts.json 2026-05-02T10:00:00+00:00 2026-05-02T11:00:00+00:00

    # filtrando por câmera
    python example_query.py counts.json 2026-05-02T10:00:00+00:00 2026-05-02T11:00:00+00:00 loja-entrada
"""

import sys
from datetime import datetime

from analytics import count_in_interval
from storage import SnapshotStore


def main():
    if len(sys.argv) not in (4, 5):
        print("uso: python example_query.py <storage.json> <start_iso> <end_iso> [camera_id]")
        sys.exit(1)

    path, start_s, end_s = sys.argv[1], sys.argv[2], sys.argv[3]
    camera_id = sys.argv[4] if len(sys.argv) == 5 else None

    # Sem camera_id no construtor: store em modo somente-leitura.
    store = SnapshotStore(path)
    result = count_in_interval(
        store,
        datetime.fromisoformat(start_s),
        datetime.fromisoformat(end_s),
        camera_id=camera_id,
    )

    label = f"camera={camera_id}" if camera_id else "todas as câmeras"
    print(f"Intervalo: {start_s}  ->  {end_s}  ({label})")
    print(f"  Entradas: {result['in']}")
    print(f"  Saídas:   {result['out']}")
    print(f"  Total:    {result['total']}")


if __name__ == "__main__":
    main()
