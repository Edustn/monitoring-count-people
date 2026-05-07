# Monitoring Count People

Sistema de contagem de pessoas em tempo real usando visão computacional, com foco em **conformidade com a LGPD**: nenhuma imagem é armazenada, nenhuma pessoa é identificada, apenas eventos agregados de cruzamento de uma linha virtual.

Construído com **YOLOv8** (detecção) + **ByteTrack** (tracking) + **OpenCV** (vídeo). Persiste snapshots periódicos em JSON-lines no disco local.

---

## O que faz

- Detecta pessoas em vídeo (arquivo, webcam ou stream RTSP).
- Atribui IDs temporários a cada pessoa rastreada (descartados quando ela some do quadro).
- Conta quando o **centro do bounding box** atravessa uma linha virtual configurável.
- Diferencia **entrada** (`IN`) de **saída** (`OUT`) pela direção do cruzamento.
- Grava snapshots periódicos com `timestamp + IN + OUT + total` — sem frames, sem rostos, sem trilhas.
- Permite calcular quantas pessoas passaram em qualquer intervalo de tempo a partir do histórico.

## O que NÃO faz (e por quê)

- **Não usa reconhecimento facial.** YOLO detecta a classe `person` do COCO, sem qualquer descritor de identidade.
- **Não armazena imagens nem vídeo.** Só o contador acumulado e o timestamp são gravados.
- **Não persiste IDs do tracker.** Os IDs são efêmeros, vivem só na memória do processo.
- **Não usa re-identificação visual** (DeepSORT por exemplo). Optamos por ByteTrack puro justamente para evitar embeddings que poderiam ser usados pra rastrear indivíduos.

---

## Arquitetura

```
            +---------+    +-----------+    +-------------+    +----------+
 video ---> | YOLOv8  |--> | ByteTrack |--> | LineCounter |--> | Storage  |
            | (person)|    | (track_id)|    | (cruzamento)|    | (.json)  |
            +---------+    +-----------+    +-------------+    +----------+
                                                  |
                                                  v
                                            +-----------+
                                            | OpenCV UI |
                                            +-----------+
```

| Arquivo | Responsabilidade |
| --- | --- |
| [`counter.py`](counter.py) | `LineCounter` — lógica pura de cruzamento de linha (sem dependências de visão). Testável isoladamente. |
| [`storage.py`](storage.py) | `SnapshotStore` — append-only em JSON-lines. Robusto a crashes, fácil de processar com `jq`/pandas. |
| [`analytics.py`](analytics.py) | `count_in_interval` — diferença entre snapshots para responder "quantos passaram entre T1 e T2". |
| [`main.py`](main.py) | Pipeline: YOLO + ByteTrack + visualização ao vivo + persistência. |
| [`example_query.py`](example_query.py) | CLI de exemplo pra consultar um intervalo. |

---

## Como funciona o cruzamento de linha

A linha virtual é definida por dois pontos `P1=(x1,y1)` e `P2=(x2,y2)`. Para cada centro de bounding box `C=(cx,cy)`, calculamos o **produto vetorial 2D**:

```
d = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
```

- `d > 0` → ponto está **à esquerda** da direção `P1 → P2` (lado `+1`)
- `d < 0` → ponto está **à direita** (lado `-1`)
- `d == 0` → exatamente em cima da linha (frame ignorado)

Quando o lado de um `track_id` muda entre dois frames consecutivos (ex: `+1 → -1`), contamos um cruzamento. A direção da mudança define se foi `IN` ou `OUT`. Para inverter qual sentido conta como entrada, basta trocar a ordem de `P1` e `P2`.

Funciona pra linha em **qualquer ângulo** — horizontal, vertical, diagonal.

### Por que esse design

- **Cada cruzamento real conta uma vez.** Frames com o centro exatamente em cima da linha são ignorados (não zeram o estado), então cruzamentos que passam por `d==0` em algum frame não são perdidos.
- **A mesma pessoa pode entrar e sair.** O `track_id` não é "consumido" depois da primeira contagem — uma pessoa que entra (`IN`) e depois sai (`OUT`) registra os dois eventos.
- **Pessoas paradas em um lado nunca contam.** Sem mudança de lado, sem evento.
- **Pessoas que se aproximam da linha mas não cruzam não contam.** A direção `d` precisa efetivamente trocar de sinal.

---

## Instalação

Pré-requisito: **Python 3.10+**.

```bash
git clone <este-repo>
cd monitoring-count-people

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Dependências (em [`requirements.txt`](requirements.txt)):

- `ultralytics` — YOLOv8 + ByteTrack integrado
- `opencv-python` — captura de vídeo e visualização
- `numpy`, `lap` — usados pelo tracker

O peso `yolov8n.pt` (~6 MB) é baixado automaticamente na primeira execução.

---

## Uso

### Comando básico

```bash
python main.py --source <fonte> --line <x1,y1,x2,y2> --camera-id <identificador>
```

A linha aceita **pixels** (`320,0,320,480`) ou **porcentagens** (`50%,0%,50%,100%`). Porcentagens são resolvidas no primeiro frame contra a resolução real, então não é preciso saber se a webcam é 640x480 ou 1920x1080.

O `--camera-id` é uma string que identifica a fonte (ex: `"loja-entrada"`, `"saida-fundos"`, `"cam-01"`). Ele é gravado em cada snapshot e permite que múltiplas câmeras escrevam no mesmo arquivo sem conflito — depois você consulta filtrando por câmera.

### Exemplos

```bash
# Webcam (linha vertical no meio do frame)
python main.py --source 0 --line 50%,0%,50%,100% --camera-id minha-webcam

# Arquivo de vídeo (linha horizontal embaixo, simulando porta vista de cima)
python main.py --source video.mp4 --line 0%,80%,100%,80% --camera-id loja-entrada

# Stream RTSP, sem janela de visualização (servidor headless)
python main.py --source "rtsp://user:pass@cam/stream" --line 50%,0%,50%,100% --camera-id portao-1 --no-show

# Linha diagonal, snapshots a cada 2s, modelo maior (mais preciso)
python main.py --source 0 --line 0%,0%,100%,100% --camera-id cam-01 --interval 2 --model yolov8s.pt
```

### Todos os parâmetros

| Flag | Default | Descrição |
| --- | --- | --- |
| `--source` | (obrigatório) | índice de webcam (`0`), caminho de vídeo, ou URL RTSP |
| `--line` | (obrigatório) | `x1,y1,x2,y2` em pixels ou `%` |
| `--camera-id` | (obrigatório) | identificador da câmera, gravado em cada snapshot |
| `--model` | `yolov8n.pt` | peso YOLO. `n`=nano (CPU), `s`/`m`/`l`/`x` cada vez maiores e mais precisos |
| `--interval` | `10.0` | segundos entre snapshots gravados |
| `--storage` | `counts.json` | arquivo JSON-lines de saída |
| `--no-show` | (off) | desabilita a janela do OpenCV |

### Visualização ao vivo

Enquanto roda com a janela ligada:

- **Linha ciano** = a linha virtual configurada.
- **Bbox verde** (lado `L`) ou **vermelho** (lado `R`) = mostra em qual lado da linha o sistema considera cada pessoa em tempo real. Se a cor está alternando, o cruzamento vai contar.
- **Bbox amarelo** (`ON-LINE`) = centro exatamente em cima da linha. Esses frames são ignorados pelo contador (não atrapalham e nem disparam contagem indevida).
- **HUD no canto** = `IN`, `OUT`, `TOTAL` acumulados.
- **`q`** = encerra o processo (com snapshot final pra não perder os últimos eventos).

---

## Persistência

Cada snapshot é uma linha do arquivo `counts.json` (formato JSON-lines):

```json
{"camera_id": "loja-entrada", "timestamp": "2026-05-02T18:42:14.353883+00:00", "in": 12, "out": 7, "total": 19}
{"camera_id": "loja-entrada", "timestamp": "2026-05-02T18:42:24.430766+00:00", "in": 13, "out": 7, "total": 20}
{"camera_id": "saida-fundos",  "timestamp": "2026-05-02T18:42:25.118200+00:00", "in":  4, "out": 9, "total": 13}
```

- `camera_id` identifica a fonte do snapshot. Múltiplas câmeras (cada uma rodando seu próprio processo `main.py` com `--camera-id` distinto) podem gravar no mesmo arquivo sem conflito.
- `timestamp` em ISO 8601, sempre em **UTC**.
- `in` e `out` são **acumulados** desde o início da execução daquele processo (nunca decrementam). Cada câmera tem seu próprio acumulado independente.
- O arquivo é **append-only** — múltiplas execuções/câmeras vão acrescentando novas linhas.
- Robusto a crashes: se o processo morrer no meio de uma escrita, no máximo um snapshot é perdido e o arquivo continua legível.

> **Atenção:** quando o processo é reiniciado, o `LineCounter` zera (começa em `IN=0, OUT=0`), mas as linhas antigas continuam no arquivo. As ferramentas de análise lidam com isso usando **diferenças entre snapshots vizinhos**, não valores absolutos.

---

## Consultando intervalos

Como os contadores são monotonicamente crescentes dentro de uma execução, o número de pessoas que passaram entre `T1` e `T2` é simplesmente:

```
N = snapshot_em_T2 - snapshot_em_T1
```

Use o script de exemplo:

```bash
# todas as câmeras combinadas
python example_query.py counts.json 2026-05-02T10:00:00+00:00 2026-05-02T11:00:00+00:00

# filtrando por câmera específica
python example_query.py counts.json 2026-05-02T10:00:00+00:00 2026-05-02T11:00:00+00:00 loja-entrada
```

Saída:

```
Intervalo: 2026-05-02T10:00:00+00:00  ->  2026-05-02T11:00:00+00:00  (camera=loja-entrada)
  Entradas: 47
  Saídas:   42
  Total:    89
```

Ou programaticamente:

```python
from datetime import datetime
from analytics import count_in_interval
from storage import SnapshotStore

# Sem camera_id no construtor: store em modo somente-leitura
store = SnapshotStore("counts.json")

result = count_in_interval(
    store,
    datetime.fromisoformat("2026-05-02T10:00:00+00:00"),
    datetime.fromisoformat("2026-05-02T11:00:00+00:00"),
    camera_id="loja-entrada",  # opcional: filtra por câmera
)
print(result)  # {'in': 47, 'out': 42, 'total': 89}
```

---

## LGPD / privacidade

Este projeto foi desenhado com a LGPD em mente:

- **Sem dado pessoal armazenado.** O JSON contém só timestamp e contadores agregados.
- **Sem reconhecimento facial.** YOLO detecta a forma genérica de uma pessoa, não rostos.
- **Sem re-identificação entre execuções.** Os `track_id` do ByteTrack vivem só em memória e somem quando o processo encerra.
- **Sem captura de imagem persistida.** Frames são processados em RAM e descartados.

Se for usar em ambiente público, ainda é boa prática:

- Sinalizar visivelmente que há contagem em andamento.
- Documentar a finalidade do tratamento (Art. 6º LGPD).
- Manter o `counts.json` em local com acesso restrito (mesmo sendo agregado, é dado operacional).

---

## Dicas de tuning

- **Posicionamento da linha:** prefira lugares onde as pessoas naturalmente atravessem (corredores, portas). Linhas em áreas onde as pessoas ficam paradas vão gerar oscilações.
- **Ponto de referência do bbox:** por padrão usamos o **centro**. Para câmeras altas/zenitais, considere usar os **pés** (`cy + h/2`) — mais estável. Edite [`main.py`](main.py:154).
- **Modelo:** `yolov8n.pt` roda em CPU mas perde detecções em cenas movimentadas. Para uma loja com fluxo alto, troque pra `yolov8s.pt` ou `yolov8m.pt` (precisa de GPU pra fluir em tempo real).
- **Intervalo de snapshot:** 10s é um bom default. Para análises minuto-a-minuto, baixe pra 1–2s. Para histórico de longo prazo, suba pra 30–60s.
- **Streams instáveis (RTSP):** o pipeline lida bem com perda de frames, mas se a câmera cair completamente o processo encerra. Em produção, use um supervisor (`systemd`, `supervisord`) pra reiniciar.
