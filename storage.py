"""
Persistência mínima dos snapshots de contagem.

LGPD-friendly: gravamos APENAS timestamp + contadores acumulados.
Nenhum frame, nenhum recorte de pessoa, nenhum embedding facial.

Formato no disco: JSON-lines (um snapshot por linha). Esse formato é robusto
a falhas — se o processo cair no meio de uma escrita, perdemos no máximo um
snapshot, e o arquivo continua legível linha a linha.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator


@dataclass
class Snapshot:
    timestamp: str   # ISO 8601 em UTC
    in_count: int
    out_count: int

    @property
    def total(self) -> int:
        return self.in_count + self.out_count

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "in": self.in_count,
            "out": self.out_count,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        return cls(
            timestamp=d["timestamp"],
            in_count=int(d["in"]),
            out_count=int(d["out"]),
        )


class SnapshotStore:
    """Append-only store em JSON-lines."""

    def __init__(self, path: str):
        self.path = path
        # Garante que o arquivo existe — append puro não cria diretórios
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if not os.path.exists(path):
            open(path, "a").close()

    def save(self, in_count: int, out_count: int) -> Snapshot:
        snap = Snapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            in_count=in_count,
            out_count=out_count,
        )
        # Append em uma única linha — atômico do ponto de vista do leitor
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snap.to_dict()) + "\n")
        return snap

    def iter_snapshots(self) -> Iterator[Snapshot]:
        """Itera todos os snapshots em ordem de gravação."""
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield Snapshot.from_dict(json.loads(line))
