"""
Lógica de contagem por cruzamento de linha virtual.

Conceito-chave: cada pessoa rastreada tem um ID temporário (vindo do tracker).
Para cada ID, guardamos de que "lado" da linha ele estava no frame anterior.
Quando o lado muda, sabemos que houve um cruzamento — e a direção do cruzamento
nos diz se foi entrada ou saída.

Decidimos o lado usando o sinal do produto vetorial 2D entre o vetor da linha
(P2 - P1) e o vetor do ponto até P1. Isso é robusto a linhas em qualquer ângulo
(não só horizontais/verticais).
"""

from dataclasses import dataclass, field
from typing import Optional


Point = tuple[int, int]


@dataclass
class LineCounter:
    # Linha virtual definida por dois pontos em coordenadas de pixel
    line_start: Point
    line_end: Point

    # Contadores acumulados — nunca decrementam, são monotonicamente crescentes
    in_count: int = 0
    out_count: int = 0

    # Último lado NÃO-ZERO conhecido para cada track_id.
    # Guardamos só ±1 (nunca 0); frames com o centro exatamente em cima da linha
    # são ignorados em vez de zerar o estado — senão um cruzamento que passa
    # por side==0 seria perdido (prev=+1 -> 0 -> -1 nunca dispara).
    _last_side: dict[int, int] = field(default_factory=dict)

    def _side_of(self, point: Point) -> int:
        """Retorna +1, -1 ou 0 dependendo de qual lado da linha o ponto está.

        Usa o sinal do produto vetorial 2D: d = (x2-x1)*(py-y1) - (y2-y1)*(px-x1).
        Se d > 0 -> ponto à esquerda da direção P1->P2; d < 0 -> à direita.
        """
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        px, py = point
        d = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if d > 0:
            return 1
        if d < 0:
            return -1
        return 0

    def update(self, track_id: int, center: Point) -> Optional[str]:
        """Atualiza o estado para um track_id e retorna 'in', 'out' ou None.

        - 'in'  -> cruzou no sentido considerado entrada
        - 'out' -> cruzou no sentido considerado saída
        - None  -> não houve cruzamento neste frame

        Cada cruzamento (transição de lado A para lado B) conta uma vez. Idas
        e voltas legítimas da mesma pessoa contam ambas — o ID identificador
        do tracker não é "consumido" depois da primeira contagem. Isso é o que
        diferencia entrada de saída em um portal: a mesma pessoa entra e
        depois sai, e ambos os eventos são registrados.
        """
        side = self._side_of(center)

        # Centro em cima da linha: ignoramos o frame inteiro. Não atualizamos
        # _last_side — assim, quando o ponto sair para o outro lado, ainda
        # comparamos com o lado original e detectamos o cruzamento.
        if side == 0:
            return None

        prev = self._last_side.get(track_id)

        # Primeira observação não-zero deste ID: só registra o lado.
        if prev is None:
            self._last_side[track_id] = side
            return None

        # Mesmo lado da última vez: nada a fazer.
        if prev == side:
            return None

        # Cruzamento real: prev e side são ambos ±1 e diferentes.
        self._last_side[track_id] = side

        # Convenção: cruzar de -1 para +1 = entrada; o oposto = saída.
        # Para inverter, basta trocar a ordem de line_start/line_end.
        if prev == -1 and side == 1:
            self.in_count += 1
            return "in"
        else:
            self.out_count += 1
            return "out"

    @property
    def total(self) -> int:
        """Total acumulado de cruzamentos (entradas + saídas)."""
        return self.in_count + self.out_count
