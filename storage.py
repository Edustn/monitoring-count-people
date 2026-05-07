"""
Persistência mínima dos snapshots de contagem.

LGPD-friendly: gravamos APENAS camera_id + timestamp + contadores acumulados.
Nenhum frame, nenhum recorte de pessoa, nenhum embedding facial.

Formato no disco: JSON-lines (um snapshot por linha). Esse formato é robusto
a falhas — se o processo cair no meio de uma escrita, perdemos no máximo um
snapshot, e o arquivo continua legível linha a linha.

Sobre `camera_id`: identifica a fonte (ex: "loja-entrada", "saida-fundos",
"cam-01"). Permite que múltiplas câmeras gravem no mesmo arquivo e que
análises depois filtrem por câmera.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional


@dataclass
class Snapshot:
    camera_id: str   # identificador da câmera (ex: "loja-entrada")
    timestamp: str   # ISO 8601 em UTC
    in_count: int
    out_count: int

    @property
    def total(self) -> int:
        return self.in_count + self.out_count

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "in": self.in_count,
            "out": self.out_count,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        # `camera_id` é opcional na leitura para tolerar arquivos antigos
        # gravados antes do campo existir; vira "" nesses casos.
        return cls(
            camera_id=str(d.get("camera_id", "")),
            timestamp=d["timestamp"],
            in_count=int(d["in"]),
            out_count=int(d["out"]),
        )


class SnapshotStore:
    """Append-only store em JSON-lines.

    Para gravar, é obrigatório informar `camera_id` no construtor — todo
    snapshot escrito carrega esse identificador. Para apenas ler/consultar,
    o `camera_id` é opcional.
    """

    def __init__(self, path: str, camera_id: Optional[str] = None):
        self.path = path
        self.camera_id = camera_id
        # Garante que o arquivo existe — append puro não cria diretórios
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if not os.path.exists(path):
            open(path, "a").close()

    def save(self, in_count: int, out_count: int) -> Snapshot:
        if not self.camera_id:
            raise RuntimeError(
                "save() requer um camera_id no construtor do SnapshotStore"
            )
        snap = Snapshot(
            camera_id=self.camera_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            in_count=in_count,
            out_count=out_count,
        )
        # Append em uma única linha — atômico do ponto de vista do leitor
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snap.to_dict()) + "\n")
        return snap

    def iter_snapshots(self, camera_id: Optional[str] = None) -> Iterator[Snapshot]:
        """Itera todos os snapshots em ordem de gravação.

        Se `camera_id` for informado, devolve só os snapshots daquela câmera.
        Útil quando múltiplas câmeras compartilham o mesmo arquivo.
        """
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                snap = Snapshot.from_dict(json.loads(line))
                if camera_id is not None and snap.camera_id != camera_id:
                    continue
                yield snap
