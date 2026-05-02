"""
Pipeline principal: detecta pessoas com YOLOv8, rastreia com ByteTrack,
conta cruzamentos de uma linha virtual e persiste snapshots periódicos.

Uso:
    # arquivo de vídeo
    python main.py --source video.mp4 --line 100,300,1180,300

    # webcam (índice 0)
    python main.py --source 0 --line 100,300,540,300

    # stream RTSP
    python main.py --source rtsp://user:pass@cam/stream --line 0,400,1920,400

A linha é informada como x1,y1,x2,y2 em pixels. Pessoas que cruzarem de
"baixo-direita" para "cima-esquerda" da direção P1->P2 contam como entrada;
o oposto, como saída. Para inverter, troque a ordem dos pontos.
"""

import argparse
import time
from typing import Iterable

import cv2
from ultralytics import YOLO

from counter import LineCounter
from storage import SnapshotStore


# COCO class id 0 = person. É a única classe que nos interessa.
PERSON_CLASS_ID = 0


LineTokens = tuple[tuple[float, bool], tuple[float, bool], tuple[float, bool], tuple[float, bool]]


def parse_line(s: str) -> LineTokens:
    """Converte 'x1,y1,x2,y2' em 4 tokens (valor, é_porcentagem).

    Aceita pixels (`320`) ou porcentagens (`50%`). Porcentagens são resolvidas
    em runtime contra a resolução real do frame, então o usuário não precisa
    saber se a webcam captura em 640x480, 1280x720 ou 1920x1080.
    """
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--line deve estar no formato x1,y1,x2,y2 (pixels ou %)")

    tokens: list[tuple[float, bool]] = []
    for p in parts:
        if p.endswith("%"):
            tokens.append((float(p[:-1]), True))
        else:
            tokens.append((float(p), False))
    return tuple(tokens)  # type: ignore[return-value]


def resolve_line(tokens: LineTokens, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Converte tokens em coordenadas absolutas usando largura/altura do frame."""
    def resolve(val: float, is_pct: bool, ref: int) -> int:
        return int(round(val / 100 * ref)) if is_pct else int(round(val))

    x1 = resolve(tokens[0][0], tokens[0][1], width)
    y1 = resolve(tokens[1][0], tokens[1][1], height)
    x2 = resolve(tokens[2][0], tokens[2][1], width)
    y2 = resolve(tokens[3][0], tokens[3][1], height)
    return (x1, y1), (x2, y2)


def parse_source(s: str):
    """Aceita índice de webcam (int) ou path/URL (string)."""
    try:
        return int(s)
    except ValueError:
        return s


def draw_overlay(
    frame,
    counter: LineCounter,
    boxes_with_ids: Iterable[tuple[tuple[float, float, float, float], int]],
):
    """Desenha a linha virtual, as bounding boxes e os contadores no frame."""
    # Linha virtual em ciano para destacar do conteúdo do vídeo
    cv2.line(frame, counter.line_start, counter.line_end, (255, 255, 0), 2)

    # BGR (não RGB) — OpenCV usa essa ordem
    SIDE_COLORS = {
        1: (0, 255, 0),     # verde
        -1: (0, 0, 255),    # vermelho
        0: (0, 255, 255),   # amarelo (em cima da linha — não conta)
    }
    SIDE_LABELS = {1: "L", -1: "R", 0: "ON-LINE"}

    for (x, y, w, h), tid in boxes_with_ids:
        # xywh -> cantos
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        # Cor do bbox indica de que lado da linha o sistema considera o objeto.
        # Se a cor está oscilando entre verde e vermelho, está cruzando = vai contar.
        side = counter._side_of((int(x), int(y)))
        color = SIDE_COLORS[side]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Centro do bbox — o ponto que efetivamente cruza a linha
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        cv2.putText(
            frame, f"id:{tid} ({SIDE_LABELS[side]})", (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    # HUD com os contadores acumulados
    hud = f"IN: {counter.in_count}   OUT: {counter.out_count}   TOTAL: {counter.total}"
    cv2.rectangle(frame, (5, 5), (5 + 8 * len(hud), 40), (0, 0, 0), -1)
    cv2.putText(
        frame, hud, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )


def process_video(
    source,
    line_tokens: LineTokens,
    model_path: str = "yolov8n.pt",
    snapshot_interval: float = 10.0,
    storage_path: str = "counts.json",
    show: bool = True,
):
    """Loop principal: detecta + rastreia + conta + persiste.

    A linha é resolvida no primeiro frame para suportar tokens em porcentagem.
    """
    model = YOLO(model_path)
    store = SnapshotStore(storage_path)
    counter: LineCounter | None = None  # criado no 1º frame quando soubermos o tamanho

    last_snapshot = time.time()

    # model.track com stream=True devolve um gerador frame-a-frame.
    # persist=True mantém os IDs estáveis entre frames consecutivos.
    # tracker='bytetrack.yaml' usa o ByteTrack que já vem na ultralytics.
    results = model.track(
        source=source,
        classes=[PERSON_CLASS_ID],
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        verbose=False,
    )

    try:
        for result in results:
            frame = result.orig_img
            boxes_with_ids: list[tuple[tuple[float, float, float, float], int]] = []

            # No primeiro frame, resolvemos a linha contra a resolução real do frame.
            # Isso permite passar `--line 50%,0%,50%,100%` sem saber a resolução.
            if counter is None:
                h, w = frame.shape[:2]
                ls, le = resolve_line(line_tokens, w, h)
                counter = LineCounter(line_start=ls, line_end=le)
                print(f"[info] frame {w}x{h} | linha: {ls} -> {le}")

            # Pode não haver detecções neste frame — checamos antes de iterar
            if result.boxes is not None and result.boxes.id is not None:
                # xywh = (centro_x, centro_y, largura, altura)
                xywh = result.boxes.xywh.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()

                for box, tid in zip(xywh, ids):
                    cx, cy = float(box[0]), float(box[1])
                    # O centro do bbox é o ponto de referência para cruzamento.
                    # Para cenas com câmera elevada, usar (cx, cy + h/2) — pés —
                    # costuma dar resultado mais estável.
                    counter.update(tid, (int(cx), int(cy)))
                    boxes_with_ids.append(
                        ((float(box[0]), float(box[1]), float(box[2]), float(box[3])), tid)
                    )

            # Snapshot periódico — independe do FPS do vídeo
            now = time.time()
            if now - last_snapshot >= snapshot_interval:
                store.save(counter.in_count, counter.out_count)
                last_snapshot = now

            if show:
                draw_overlay(frame, counter, boxes_with_ids)
                cv2.imshow("People Counter", frame)
                # 'q' encerra; o waitKey também é o que mantém a janela responsiva
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        # Snapshot final garante que não perdemos os últimos eventos
        if counter is not None:
            store.save(counter.in_count, counter.out_count)
        if show:
            cv2.destroyAllWindows()

    return counter


def main():
    parser = argparse.ArgumentParser(description="Contador de pessoas com YOLO + ByteTrack")
    parser.add_argument("--source", required=True,
                        help="caminho de vídeo, índice de webcam (ex: 0) ou URL RTSP")
    parser.add_argument("--line", required=True,
                        help="linha virtual no formato x1,y1,x2,y2 (pixels ou %, ex: '50%%,0%%,50%%,100%%')")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="peso YOLO (n=nano é o mais leve; s/m/l/x maiores e mais precisos)")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="intervalo entre snapshots em segundos")
    parser.add_argument("--storage", default="counts.json",
                        help="arquivo JSON-lines para gravar os snapshots")
    parser.add_argument("--no-show", action="store_true",
                        help="desabilita a janela de visualização (útil em servidor)")
    args = parser.parse_args()

    process_video(
        source=parse_source(args.source),
        line_tokens=parse_line(args.line),
        model_path=args.model,
        snapshot_interval=args.interval,
        storage_path=args.storage,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
