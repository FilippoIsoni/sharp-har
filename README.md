# sharp-har

Human Activity Recognition da tracce WiFi CSI (Doppler) sul dataset **SHARP**
(*Environment and Person-Independent Activity Recognition with Commodity
IEEE 802.11 Access Points*), con protocollo di valutazione **LOEO**
(Leave-One-Environment-Out) e una progressione di run core C0→C4 (baseline
di riproduzione, cross-entropy, adversarial GRL, contrastivo SupCon, e la
combinazione SupCon+GRL).

## Principio: thin-notebook, logica nel package

Tutta la logica vive nel package Python versionato `sharp_har/`. I notebook
in `notebooks/` sono **runner sottili**: montano Drive, fanno staging dei
dati, chiamano le funzioni del package e mostrano output ispezionabili.
Non contengono logica propria. Questo è necessario perché la pipeline
richiede code review incrociata del dataloader e riproducibilità completa
(config YAML, seed, git hash) — cose che un diff di notebook non permette
di verificare.

## Principi non negoziabili

1. **Split per trace, mai per finestra.** Le liste di split contengono
   trace-id, non indici di finestre — evita leakage tra train/val/test.
2. **μ/σ solo dal train** della rotazione corrente, salvate nel file di
   split. Non si ricalcolano mai su val/test.
3. **Seed unico = 42** per ogni scelta stocastica del progetto.
4. **Gli split, una volta congelati, si committano su Git e non si
   toccano più.**
5. **I dati e i checkpoint non entrano mai nel repo.** Vivono su Drive
   (~762 MB); il repo contiene solo codice, config, split congelati
   (`splits/*.json`) e report (`reports/*.csv`).
6. **L'architettura di training non è decisa finché non serve.** I moduli
   dei giorni 2–9 sono stub con firma e `NotImplementedError`: la review
   è possibile, l'implementazione arriva al gate corrispondente.
7. **Ogni invocazione sul test set viene loggata**, incluso il wrapper di
   valutazione stile repo SHARP per C0 — il test set non si consuma
   accidentalmente in iterazioni di sviluppo.

## Mappa notebook → giorni

| Notebook | Giorno | Stato | Scopo |
|---|---|---|---|
| `00_setup_smoke.ipynb` | — | stub | Mount Drive + staging, verifica veloce dell'ambiente |
| `01_inventory_splits.ipynb` | 1 | **implementato** | Inventario dati + split congelati (`p2_office`, `p1_sharp`) |
| `02_smoke_gate.ipynb` | 2 | stub | Smoke test modello + gate di throughput |
| `03_train.ipynb` | 3+ | stub | Runner generico di training su una config `configs/*.yaml` |

## Dati: mai nel repo

I dati Doppler CSI (due archivi zip, ~762 MB totali) vivono su Google
Drive, path definito in `configs/paths.yaml`. I notebook li montano e li
staggiano localmente su Colab (`/content/data`); il training legge solo
dallo staging locale, mai da Drive. Checkpoint e feature cache seguono la
stessa regola: mai committati (vedi `.gitignore`).

## Come far girare il Giorno 1

```bash
pip install -r requirements.txt
```

Aprire `notebooks/01_inventory_splits.ipynb` (su Colab o in locale con i
dati già staggiati) ed eseguire le celle in ordine. Il notebook:

1. monta Drive e stagga i due zip in locale, cronometrando lo staging;
2. ispeziona i nomi file reali e chiede conferma del pattern regex prima
   di procedere;
3. costruisce `reports/inventory.csv` (una riga per file-stream) e
   `reports/name_to_arset.json`;
4. esegue le verifiche del gate del giorno 1 (assi, copertura AR-set,
   contingenza attività×AR-set, policy NaN ≤5%, conteggio finestre vs
   attesi);
5. congela `splits/p2_office.json` (rotazione primaria C1–C4) e
   `splits/p1_sharp.json` (riproduzione SHARP per C0).

Gli artefatti prodotti (`splits/*.json`, `reports/*.csv`,
`reports/name_to_arset.json`) si committano su Git: sono i deliverable
congelati del giorno 1.
