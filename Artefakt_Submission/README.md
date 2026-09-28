# Phishing-E-Mail-Klassifikationsartefakt

## Zweck

Dieses Paket enthält das final evaluierte, textgeführte Klassifikationsartefakt mit ergänzender URL-Evidenz. Das vortrainierte Artefakt kann unmittelbar für Inferenz verwendet werden. Nach Ergänzung der Raw-Datasets lässt sich außerdem die vollständige aktive Pipeline reproduzieren.

## Voraussetzungen

Der finale Lauf wurde mit Python 3.10.12 durchgeführt. Für die reproduzierbare Installation werden die exakt nachgewiesenen direkten Runtime-/ML-Versionen empfohlen.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-reproducible.txt
```

`requirements.txt` bleibt zusätzlich als ursprüngliche Projektdependency-Datei erhalten.

## Vortrainiertes Artefakt direkt starten

```bash
.venv/bin/python -B -m src.unified_artifact \
  --subject "Email subject" \
  --body "Email body"
```

Mit einem bereits passend eingerichteten Python-Environment ist auch der allgemeine Aufruf möglich:

```bash
python3 -B -m src.unified_artifact \
  --subject "Email subject" \
  --body "Email body"
```

## Erwartete Raw-Datasets

Raw-Daten sind nicht Bestandteil dieses Pakets. Für einen vollständigen Neuaufbau müssen sie unter den folgenden Pfaden ergänzt werden:

- Enron: `data/raw/enron/enron_mail_20150507.tar.gz`. Wissenschaftliche Referenz: Klimt & Yang, *The Enron Corpus: A New Dataset for Email Classification Research*, DOI `10.1007/978-3-540-30115-8_22`. Für den verwendeten Archiv-Snapshot ist keine konkrete Download-URL im Paket belegt.
- Nazario: `data/raw/nazario/phishing-2015.txt`, `phishing-2016.txt`, `phishing-2017.txt`, `phishing-2018`, `phishing-2019.txt` bis `phishing-2024.txt` sowie `nazario2025.txt`. Quelle: https://monkey.org/~jose/phishing/. Snapshots 2015–2024 dienen der historischen Entwicklung; 2025 ist für die temporale Evaluation reserviert.
- Phishload: `data/raw/phishload/current.zip` mit `phishload_120322.sql` und `phishload_readme.txt`. Quelle: https://www.medien.ifi.lmu.de/team/max.maurer/files/phishload/. Verwendeter Snapshot: `phishload_120322`, 2012-03-22.
- TREC: `data/raw/trec/archive.zip` mit `processed_data.csv`. Wissenschaftliche Referenz: TREC 2007 Spam Track, wie in der Thesis-Bibliography zitiert. Eine konkrete Download-URL ist im Paket nicht belegt. Der Spam/Ham-Datensatz wird für die vollständige Pipeline geladen, aber methodisch ausgeschlossen und als `trec_excluded.csv` isoliert; seine Labels werden nicht als Phishing/Legitimate-Ground-Truth umgedeutet.
- E-PhishLLM: `data/raw/E-PhishLLM/ephishLLM.json`. Quelle: https://github.com/pajola/e-phishGen. Verwendet wird der lokale JSON-Snapshot; ein Upstream-Commit oder Release-Tag wurde nicht aufgezeichnet.
- PhishingPot: `data/raw/PhishingPot/phishing_pot/email/**/*.eml`. Quelle: https://github.com/rf-peixoto/phishing_pot. Verwendeter Upstream-Commit: `49f63777126b0bdb9eb1f6e770a5c3f9df2b0306`.
- DIDECO: `data/raw/DIDECO/2026.lrec-1.542.OptionalSupplementaryMaterial.zip`. Publikation: https://aclanthology.org/2026.lrec-1.542/. Repository: https://github.com/Sneusneu/Dideco/. Verwendeter Snapshot: `2026.lrec-1.542.OptionalSupplementaryMaterial.zip`.

## Vollständige Pipeline reproduzieren

Der Production Runner verlangt einen vollständig fehlenden `outputs/`-Ordner. Das ausgelieferte vortrainierte Artefakt liegt dort, damit die Direktinferenz sofort funktioniert. Vor einem vollständigen Neuaufbau daher:

```bash
mv outputs pretrained_runtime_models
.venv/bin/python -B -m scripts.run_production
```

Optional kann der finale Modellselektions-Parallelismus gesetzt werden:

```bash
.venv/bin/python -B -m scripts.run_production \
  --text-jobs 8 \
  --url-jobs 8
```

Nach einem erfolgreichen vollständigen Lauf liegen die neu trainierten Modelle wieder unter `outputs/models/`.

Die Git-freie Submission bewahrt die fachliche Pipeline, Datenrollen, Splits, Seeds, Modellparameter und Evaluationslogik. Ein vollständiger Neuaufbau kann jedoch aufgrund entfernter repository-spezifischer Git-Provenienz andere Manifest-/Report-Metadaten und daraus abgeleitete Hashes erzeugen. Das ausgelieferte vortrainierte Runtime-Artefakt selbst bleibt byteidentisch mit dem final evaluierten Artefakt.
