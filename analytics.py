"""
Cálculo de quantas pessoas passaram em um intervalo de tempo arbitrário.

Como os contadores no storage são acumulados (sempre crescentes), o número
de eventos em [t1, t2] é simplesmente:

    contagem_no_intervalo = snapshot_em_t2 - snapshot_em_t1

Procuramos o último snapshot <= t1 (estado inicial do intervalo) e o último
snapshot <= t2 (estado final). A diferença é o que aconteceu no meio.
"""

from datetime import datetime
from typing import Optional

from storage import Snapshot, SnapshotStore


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def count_in_interval(
    store: SnapshotStore,
    start: datetime,
    end: datetime,
) -> dict:
    """Retorna {'in', 'out', 'total'} entre start e end (datetimes UTC)."""
    if end < start:
        raise ValueError("end deve ser >= start")

    baseline: Optional[Snapshot] = None  # último snapshot <= start
    final: Optional[Snapshot] = None     # último snapshot <= end

    for snap in store.iter_snapshots():
        ts = _parse(snap.timestamp)
        if ts <= start:
            baseline = snap
        if ts <= end:
            final = snap
        else:
            # Snapshots vêm em ordem cronológica — podemos parar cedo
            break

    if final is None:
        return {"in": 0, "out": 0, "total": 0}

    # Se não houve snapshot antes do início, assumimos zero como baseline.
    # Isso pode superestimar se o sistema começou a contar antes do start
    # solicitado mas ainda não havia gravado nenhum snapshot — para janelas
    # razoáveis (>> intervalo de snapshot) o erro é desprezível.
    base_in = baseline.in_count if baseline else 0
    base_out = baseline.out_count if baseline else 0

    return {
        "in": final.in_count - base_in,
        "out": final.out_count - base_out,
        "total": (final.in_count + final.out_count) - (base_in + base_out),
    }
