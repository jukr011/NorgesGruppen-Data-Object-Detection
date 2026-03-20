# NorgesGruppen Data: Object Detection — NM i AI 2026

Detect and classify grocery products on store shelves.
**Metric:** mAP@0.5 &nbsp;|&nbsp; **Score:** 70% detection + 30% classification

---

## Hva du skal gjøre (steg for steg)

### 1. Sett opp Python-miljø

```bash
pip install ultralytics torch torchvision pyyaml pillow numpy
```

### 2. Last ned treningsdata

Gå til competition-siden og last ned:
- `NM_NGD_coco_dataset.zip` (864 MB) → unzip til `data/coco/`
- `NM_NGD_product_images.zip` (60 MB) → unzip til `data/products/` (valgfritt)

```
data/
  coco/
    annotations.json
    images/
      img_00001.jpg
      img_00002.jpg
      ...
```

### 3. Tren modellen

```bash
python train.py \
  --data data/coco \
  --model s \
  --epochs 50 \
  --imgsz 1280 \
  --batch 8
```

Beste vekter kopieres automatisk til `best.pt` når treningen er ferdig.

**Modellstørrelser (YOLOv8):**

| Flagg | Størrelse | Anbefaling |
|-------|-----------|------------|
| `n`   | ~6 MB     | Rask test  |
| `s`   | ~22 MB    | God balanse |
| `m`   | ~52 MB    | Bedre score |
| `l`   | ~87 MB    | Høy score  |

### 4. Test inference lokalt

```bash
# Mappe med bilder:
python run.py --input data/coco/images --output predictions.json

# Enkeltbilde:
python run.py --image data/coco/images/img_00001.jpg
```

Output er COCO-format:
```json
[
  {"image_id": 1, "category_id": 42, "bbox": [141.0, 49.0, 169.0, 152.0], "score": 0.87}
]
```

### 5. Lag submission-zip og last opp

```bash
bash zip_submission.sh
# → lager submission.zip (inneholder run.py + best.pt + requirements.txt)
```

Last opp `submission.zip` på competition-siden (maks 420 MB).

---

## Filstruktur

```
.
├── run.py              # Inference-skript (kjøres av competition-serveren)
├── train.py            # Treningsskript (kjøres lokalt)
├── requirements.txt    # Python-avhengigheter
├── zip_submission.sh   # Lager submission.zip
├── best.pt             # Trente vekter (genereres av train.py)
└── data/
    ├── coco/           # Treningsdata (last ned fra competition)
    └── products/       # Produktbilder (valgfritt)
```

---

## Tips for høyere score

- Større modell (`--model m` eller `l`) gir bedre mAP
- Flere epoker (100-200) hvis du har tid
- Større bilde (`--imgsz 1280` eller `1920`) plukker opp små produkter
- Sjekk at `best.pt` + `run.py` + `requirements.txt` er under 420 MB totalt
