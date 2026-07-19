# Pipeline sperimentale — WiFi CSI Human Activity Recognition (NNDL Project) — v5

**Dataset:** Doppler traces pre-estratti dal dataset Meneghello et al. (IEEE DataPort, 80 MHz, 802.11ac) — copia già disponibile su Drive condiviso `DATASET_SHARP`
**Team:** 3 persone · **Durata:** 15 giorni · **Compute:** Google Colab free (T4), backup Kaggle (P100)
**Riferimenti:** SHARP (IEEE TMC 2023; arXiv 2103.09924), dataset paper (IEEE Comm. Mag. 2023), repo `francescamen/SHARP`

> **v5 — cosa cambia rispetto alla v4 (recepisce `audit_pipeline_wifi_har_v4.md`).**
> (1) **Residenza dati risolta (era E-2, unico gap critico):** il pacchetto Doppler vive nel Drive condiviso `DATASET_SHARP` — `doppler_traces.zip` ≈ **762 MB** + `doppler_traces_S4_S5.zip` ≈ 10 MB, due ordini di grandezza sotto i ~25 GB temuti. Staging = copia sequenziale da Drive + unzip su `/content`, pochi minuti, **cronometrato al gate del giorno 2** (§8.5). Niente ri-download da IEEE DataPort, mai.
> (2) **Escalation (b) di §5.2 riscritta e ricalcolata:** ~2.9 GFLOPs, non ~2.5; rimosso il residuo di bozza (E-1).
> (3) **Garanzia di stratificazione resa meccanica:** pin esplicito di 1 trace per cella rara in train + assert nello script di split (I-3).
> (4) **Vincolo di offset minimo (≥340) per il riuso di trace nel sampler P×K** (I-4).
> (5) **Regola pre-committata per la griglia checkpoint di fase A sotto escalation:** ⌈2H/3⌉ / ⌈5H/6⌉ / H (A-2; con H=60 dà esattamente 40/50/60).
> (6) **Eval di C0 instradata nel logger delle invocazioni sul test** (A-1).
> (7) Precisazioni dichiarative: unità dei conteggi finestre = campioni (finestra, antenna) (I-2); "stock" definita (I-1); probe su feature non aumentate (I-5); nota P100 corretta (I-6); Adam nella probe intenzionale (I-7); asimmetria batch GRL C2 vs C4 (A-3); μ/σ su finestre sovrapposte accettato (A-4); 57→58 finestre eval (E-3); 2–3 frasi di posizionamento related-work (§9).

> **v5.1 — ERRATA DATASET (2026-07-15, verificato su Drive `DATASET_SHARP` + repo SHARP + paper TMC).**
> La copia su Drive contiene il dataset del paper SHARP TMC (cartelle `doppler_traces/S1a … S7a`), **NON** il dataset esteso a 7 ambienti / 13 soggetti del paper IEEE Comm. Mag. 2023. Conseguenze, recepite in tutto il documento:
> (1) **Copertura reale: set S1–S7** (che questo progetto denomina AR-1…AR-7, mappa 1:1 Sn ≡ AR-n), campagne **a–c a cardinalità disomogenea**: S1a/b/c, S2a/b, S3a, S4a/b, S5a, S6a/b, S7a — 12 campagne totali. Non esistono AR-8 (ufficio) né AR-9 (semi-anechoic), né campagne d/e.
> (2) **Mappa set → dominio (Tabella 1 del TMC):** S1 = bedroom, M1, P1, LOS (train nel paper); S2 = bedroom, M1, P1, LOS (giorno diverso); S3 = bedroom, M1, P2, LOS; S4 = bedroom, M2, P1, **NLOS** (libreria); S5 = bedroom, M2, P2, **NLOS**; S6 = living room, M3, P1, LOS; S7 = **laboratorio universitario, M4, P3, LOS** — il set più sfidante: ambiente, giorno e persona mai visti in train.
> (3) **Hardware identico in tutti i set** (monitor Asus RT-AC86U, link Netgear X4S): il confound hardware Asus/Netgear citato nella v5 non esiste in questa copia. I domini differiscono per: stanza, posizione monitor (M1–M4), persona (P1–P3), giorno, LOS/NLOS.
> (4) **Rotazione primaria P2 ridefinita: leave-S7-out (laboratorio)** al posto dell'inesistente "ufficio out" (§2.2). Train = S1–S6 (6 domini per la testa GRL), test = S7. E2 ridimensionata (§10.3).
> (5) **P1/C0:** il paper addestra su S1 (tutte le campagne, cfr. README repo: train S1a+S1b+S1c) e testa su S2–S7; classi del paper core = **5** (walking, running, jumping, sitting + empty; classificatore a 5 uscite, §4.1 TMC); l'estensione TMC arriva a 7 attività + empty (aggiunge standing, arm gym, sit-down/stand-up). C0 usa il set a 5 classi; C1–C4 il set completo dell'inventario (atteso 7+empty, da confermare al giorno 1 dai file).
> (6) **Tc ≈ 6 ms confermato** dal paper (Tabella parametri: channel estimates interval Tc ≃ 6·10⁻³ s): l'assunzione hop STFT del §1.1 resta da verificare solo contro i file reali.
> (7) **Vincolo operativo: il training parte esclusivamente dai dati del Drive condiviso** (link team). Nessun ri-download, nessuna integrazione da IEEE DataPort.

> **v5.2 — CHIUSURA DEL CORE E CODA SPERIMENTALE (2026-07-17, decisioni di team; evidenze nei notebook `diagnostics/` e in STATUS.md).**
> (1) **C4 chiusa senza run, su evidenza:** le diagnostiche di dominio su C1, C2 e C3 mostrano che nessun target di dominio è linearmente leggibile dalle feature di train (ogni target alla propria baseline di maggioranza, replicato su tre encoder e due famiglie di loss), mentre il GRL ha solo costi misurati (C2: −4.6 pt val, fit ridotto, nessuna invarianza aggiuntiva). Il GRL non ha bersaglio → esito atteso di C4 = "C3 più rumore": il branch si chiude e la tabella finale ha uno stream in meno (§6-C4).
> (2) **§7 dichiarato underpowered su p2_lab:** il probe AR-set non può sostenere il claim in nessuna direzione (val = 9 trace, AR-3 assente, AR-1/AR-2 indistinguibili per metadati). L'evidenza di invarianza per §9 poggia sulle diagnostiche di dominio su feature di train (§7 aggiornato).
> (3) **Righe trasduttive pre-registrate (AdaBN, T3A)** dentro l'unica sessione di test, solo su C1, iperparametri fissati a priori, dati di test non etichettati, stesso logger (nota a §0.7; dettagli in §9).
> (4) **Estensioni ridefinite (§10.3):** E1′ = replica seed 43 di C1 e C2 (emendamento della regola 5 registrato in `splits/CHANGELOG.md`; seed 44 solo su decisione esplicita); E2′ = living-out solo C1 + replica della diagnostica di dominio sul suo train; C3 dichiarata single-seed (non si replica). Diagnostiche val-only aggiuntive su feature cachate: NCM, kNN, concatenazione C1⊕C3 con controllo ensemble C1⊕C1′ (§7).
> (5) **Grafici §9 adattati:** il "prezzo dell'invarianza" (probe AR-set) è sostituito dalla tabella delle diagnostiche di dominio; t-SNE C1 vs C3 (C4 mai addestrata), con ricetta di proiezione dichiarata.
> (6) Housekeeping: cartella scratch `C1_smoke` eliminata da Drive. Nota operativa: finora ogni collaboratore gestisce le cartelle run sul proprio Drive — la sessione di test unica richiede scorciatoie Editor a TUTTE le cartelle (incluse quelle nuove seed/S6-out) da un solo account.

---

## 0. Principi non negoziabili (leggere prima di scrivere codice)

1. **Gli split si congelano il giorno 1** e si salvano come file JSON versionati su Git. Nessuna modifica successiva, mai.
2. **Split per trace, mai per finestra.** Finestre della stessa trace in train e test = leakage = risultati invalidi.
3. **Statistiche di normalizzazione calcolate solo sul train** della rotazione corrente e riusate identiche su val/test. Ogni rotazione ha il proprio file di split e le proprie μ/σ (vale anche per E2′, §10.3).
4. **Ogni run salva:** config completa (YAML), seed, git hash, checkpoint, metriche per-set in CSV. Eval harness unica e condivisa: stessa interfaccia `checkpoint → CSV metriche` per tutti gli stream.
5. **Un solo seed (42) per tutto il core.** Il seed controlla init dei pesi, shuffling/sampler (riseedato deterministicamente per epoca, §8.2) e augmentation. Differenze tra configurazioni sotto ~2 punti percentuali si descrivono come "comparabili", non come miglioramenti. **Seed dichiarato ≠ riproducibilità bit-exact:** con cuDNN benchmark attivo (necessario per il throughput) i kernel non sono deterministici; il report lo dichiara in una riga. **Emendamento E1′ (v5.2, registrato in `splits/CHANGELOG.md`):** due repliche pre-registrate a seed 43 (`C1_s43`/`C2_s43`, §10.3); seed 42 resta il seed primario di ogni stream — probe, diagnostiche e tecniche restano solo su seed 42.
6. **Nessuna run parte se non è nella tabella budget (§8.4)** e se il gate di throughput (§10.1, giorno 2) non è passato.
7. **Il test set si valuta una volta sola, alla fine, solo con il checkpoint selezionato su val.** Nessuna eval intermedia sul test, per nessuno stream, per nessun motivo. La harness logga ogni invocazione sul test. **Anche l'eval di C0 (procedura repo SHARP, §2.1) passa dal medesimo logger** tramite wrapper: nessuna via d'accesso al test fuori dall'audit trail. **Nota v5.2:** le righe trasduttive pre-registrate (AdaBN/T3A, §9) usano dati di test **non etichettati** DENTRO la medesima e unica sessione, passano dallo stesso logger e si confrontano con la riga C1 liscia; la lista delle righe è stata congelata alla team call del 2026-07-17 e **non si estende a sessione aperta**. **Lista congelata (13 righe — emendata 2026-07-19 a sessione NON ancora aperta, team call C3-ft):** C0; C1; C1_s43; C2; C2_s43; C1-lin; C2-lin; C3 (epoch40, selezione fase B); **C3-ft (fine-tuning CE dell'encoder C3, ricetta pre-registrata in `configs/c3_ft.yaml` + CHANGELOG 2026-07-19)**; C1+AdaBN; C1+T3A; C1+AdaBN+T3A; C1 S6-out (rotazione E2′). Le righe per-antenna di appendice (§6, ablation 1) sono aggregazioni diverse delle stesse invocazioni, non righe aggiuntive. Il readiness assert del notebook 05 verifica questa lista hard-coded (implementazione nello step pre-freeze, con cross-review, §10.4).

---

## 1. Dati e preprocessing

### 1.1 Inventario (giorno 1 — blocker, prima di tutto)
- **Sorgente dati: Drive condiviso `DATASET_SHARP`** (cartella già accessibile ai 3 account; ciascuno aggiunge una *scorciatoia* al proprio Drive — le scorciatoie a cartelle condivise non consumano quota del visitatore, lo storage conta solo sull'owner). Contenuto rilevante: `doppler_traces.zip` (~762 MB) + `doppler_traces_S4_S5.zip` (~10 MB). Il resto (`Python_code_old.zip` 4.5 GB, `processed_phase/`, `Doppler_plots/`, `confusion_matrices/`) NON si stagia: il riferimento per il codice è il repo GitHub `francescamen/SHARP`.
- **Verifica di copertura e mapping dei nomi (aggiornata v5.1):** la copia su Drive usa la nomenclatura del repo SHARP (`S1a … S7a`); la mappa è 1:1 con la nostra convenzione interna (Sn ≡ AR-n). Lo script di inventario costruisce e salva la **mappa file → AR-set** e verifica che tutti i set **AR-1…AR-7** siano presenti dopo l'unione dei due zip, con le campagne attese **S1a/b/c, S2a/b, S3a, S4a/b, S5a, S6a/b, S7a** (nota: `doppler_traces_S4_S5.zip` contiene proprio i set NLOS S4–S5). Set o campagne mancanti = blocker: si discute in team PRIMA di congelare gli split (mai ri-download dal DataPort, §v5.1-7).
- Script di inventario: per ogni file → set (AR-1…AR-7), campagna (a–c), attività, persona (P1–P3), ambiente (bedroom/living/laboratory), posizione monitor (M1–M4), LOS/NLOS, shape della matrice, dtype, presenza NaN, e **hop temporale effettivo dello STFT** (dai metadati o dal codice del repo SHARP): il conteggio "~20k frame Doppler per trace da 120 s" assume hop = Tc ≈ 6 ms (confermato dal paper, da verificare sui file); se l'hop è diverso cambiano finestre/trace, volumi e step/epoca — cioè il budget. Output: `inventory.csv`. I metadati per-set vengono dalla Tabella 1 del TMC (§v5.1-2), non si inventano.
- **Verifica assi:** convenzione attesa **340 passi temporali × 100 bin di velocità** (Nw=340, ND=100). L'orientamento reale si verifica dalla shape dei file + parametri del codice SHARP e si scrive nel file di split. Nessun codice dipendente dagli assi prima di questa verifica.
- **Tabella di contingenza attività×AR-set** (conteggio trace), nel report. Serve a: (a) costruire il sampler §4; (b) valutare il rischio *label shift* per il GRL (se la distribuzione delle attività differisce molto tra AR-set, la testa ambiente può predire l'ambiente dal contenuto di attività); (c) flag "n. AR-set per classe": classi presenti in un solo set di train non hanno positivi cross-domain per costruzione; (d) verificare le celle rare per la stratificazione della val (§2.2).
- **Decisione classi (duplice, vedi anche §6-C0; aggiornata v5.1):** (i) quante e quali classi contiene l'inventario (atteso: 7 attività — walking, running, jumping, sitting, standing, sit-down/stand-up, arm gym — + empty E → n_att = 8; le sigle esatte si leggono dai nomi file al giorno 1); (ii) **le tabelle core del paper TMC usano 5 classi** (walking, running, jumping, sitting + empty; classificatore a 5 uscite, §4.1 TMC), l'esperimento esteso 7+empty. Regola: **C0 sul set a 5 classi del paper; C1–C4 sul set completo dell'inventario.** Tutta la pipeline è parametrizzata su n_att: costo zero.
- **Policy NaN:** trace con NaN esclusa e loggata con motivo; se le escluse superano il 5%, stop e decisione di imputazione insieme.

### 1.2 Windowing
- Unità dati: per ogni (trace, antenna) una matrice Doppler tempo × velocità.
- **Finestra = 340 passi temporali × tutti i 100 bin.** Finestre incomplete a fine trace si scartano.
- **Stride train = 100.** Con finestra 340, l'overlap tra finestre consecutive è **~71%** (240/340): il passaggio dallo stride 30 (overlap 91%) è una **riduzione di ridondanza, non un'eliminazione**. La motivazione dello stride 100 è il budget di compute; è sufficiente da sola.
- **Stride val/test = 340 (finestre disgiunte).** Elimina la correlazione tra le unità di valutazione (a stride 100 le finestre di test condividerebbero il 71% del contenuto e i conteggi per finestra non sarebbero campioni indipendenti), al costo di ~3.4× meno finestre di eval — irrilevante per il budget (l'eval è economica) e accettabile per la stabilità delle metriche. Registrato nel file di split.
- **Volumi attesi e unità di conteggio (da confermare al giorno 1 con hop STFT e conteggio trace; ricalibrati v5.1).** A hop 6 ms: ~197 finestre/antenna/trace a stride 100; **58 finestre/antenna/trace a stride 340** in val/test. **Convenzione: tutti i conteggi di training di questo documento sono in campioni (finestra, antenna)**, cioè già espansi ×4 sulle antenne (coerente con §1.3: ogni (finestra, antenna) è un sample). Rotazione primaria (train S1–S6 = 11 campagne × ~8 trace ≈ 88 trace): ~88 × 197 × 4 ≈ **~69k campioni di train**; con batch 256, un'epoca da 400 step (§8.1) ≈ ~1.5 passate sui campioni antenna-espansi.
- Le finestre ereditano le label (attività, persona, ambiente, AR-set, trace-id, antenna) dalla trace madre.

### 1.3 Gestione antenne
- 4 antenne → ogni trace produce 4 stream Doppler.
- **Training:** ogni (finestra, antenna) è un sample indipendente (×4 i dati).
- **Val e test:** fusione per finestra = media dei softmax sulle antenne disponibili, poi argmax. Val e test usano la stessa procedura. Per-antenna in appendice (stessa forward pass, aggregazione diversa).

### 1.4 Normalizzazione
- Input a 1 canale → μ e σ **due scalari globali**, calcolati su tutte le finestre di train di tutte e 4 le antenne (dopo il windowing, prima delle augmentation). `(x − μ_train)/σ_train`, identica su val/test. μ/σ nel file di split della rotazione corrente (ricalcolate per ogni rotazione E2′, §0.3).
- **Nota (accettata, dichiarata):** calcolare μ/σ sulle finestre sovrapposte pesa i frame centrali della trace ~3.4× più dei bordi rispetto al calcolo per-trace. Per due scalari globali l'effetto è numericamente trascurabile: si tiene il calcolo post-windowing (un solo code path) e si annota qui, senza azione.
- Sanity check dal range in dB dell'inventario.

---

## 2. Data splitting

### 2.1 Protocollo P1 — Riproduzione SHARP (solo C0) — aggiornato v5.1
- **Train:** bedroom, set S1 (≡ AR-1), **tutte le campagne a/b/c**, come nel paper/repo (README SHARP: train = S1a+S1b+S1c).
- **Validation:** 20% delle trace di S1 (per trace, seed 42, congelato), solo per early stopping. **Deviazione dichiarata:** il repo SHARP addestra su tutto S1 senza hold-out → il nostro train è ridotto del 20% rispetto al paper — va nell'elenco "riproduzione parziale" del report.
- **Test:** i set di generalizzazione di SHARP: S2 (stesso setup, giorno diverso), S3 (persona P2), S4–S5 (NLOS), S6 (living room), S7 (laboratorio, nessun elemento in comune).
- **Eval di C0 = procedura del repo SHARP** (stessa fusione antenne e aggregazione temporale), non la harness di P2 — **ma invocata attraverso il wrapper della harness che logga ogni accesso al test** (§0.7): la procedura di aggregazione è quella del repo, l'audit trail è quello comune. Dove il repo non documenta, harness comune + dichiarazione. Classi: quelle dei numeri pubblicati (§1.1).

### 2.2 Protocollo P2 — Cross-domain (protocollo principale, C1–C4) — aggiornato v5.1
- **Rotazione primaria (tutto il core):** leave-one-domain-out con **laboratorio (S7 ≡ AR-7) come test** — il set più sfidante del paper: ambiente, giorno E persona (P3) mai visti in train. Train = S1–S6 (11 campagne); val = 15% delle trace di train, stratificato per (AR-set, attività), seed 42, congelato. **Limite dichiarato:** S7 ha una sola campagna (~8 trace) → test piccolo; le metriche di test si riportano con i conteggi assoluti delle finestre accanto.
- **Garanzia celle rare (meccanica, non solo dichiarata):** per ogni cella (AR-set, attività) con **< 4 trace**, lo script di split **pinna d'ufficio 1 trace (estratta con seed 42) in train**, poi stratifica il resto (per la cella degradando a solo-AR-set). A valle, **assert bloccante:** ogni cella (AR-set, attività) presente nell'inventario di train ha ≥ 1 trace in train. Lo split non si congela se l'assert fallisce. Conseguenza accettata e dichiarata: le celle rare possono non comparire in val (la macro-F1 di val è comunque definita sulle classi presenti, §9).
- **Rotazione completa:** solo come estensione E2′ (§10.3), **solo C1 (v5.2; in origine C1 e C4)**: **living-out (test = S6, train = S1–S5+S7)**. La rotazione bedroom-out è **dichiarata infattibile** (train residuo = solo S6+S7, 3 campagne: troppo poco per addestrare) e non si esegue. Ogni rotazione = proprio file di split + proprie μ/σ.
- **Label avversariale = identità dell'AR-set** (campagne aggregate; 6 domini in train nella rotazione primaria). Nei set sono confusi: stanza, posizione del monitor (M1–M4), persona (P1–P3), giorno e LOS/NLOS — l'hardware invece è identico ovunque (v5.1-3). Nel report si parla di **"AR-set invariance"**, non di "environment invariance". La persona si misura solo diagnosticamente (§7).
- **Nota sulla rotazione primaria:** con S7 come test, la persona P3 non compare mai in train → il test misura congiuntamente generalizzazione di ambiente e di persona (come nel paper SHARP, che su S7 dichiara proprio questo). S4–S5 (NLOS) restano in train: l'invarianza appresa include la variazione LOS/NLOS. Da segnalare nell'analisi.

### 2.3 Formato file di split
```json
{"protocol": "P2-lab",
 "axes": {"time": 340, "velocity": 100, "layout": "time_x_velocity", "stft_hop_s": 0.006},
 "window": {"train_stride": 100, "eval_stride": 340},
 "classes": {"n_att": 8, "labels": ["sigle esatte dai nomi file, giorno 1"], "c0_paper_set": ["walking","running","jumping","sitting","empty"]},
 "split_seed": 42,
 "pinned_train_traces": ["..."],
 "train": ["S1a_W", "..."], "val": ["..."], "test": ["S7a_W", "..."],
 "norm": {"mu": 0.0, "sigma": 1.0}}
```
Le liste contengono trace-id, non indici di finestre. `axes`, `classes` e `stft_hop_s` registrano le verifiche del giorno 1; `pinned_train_traces` documenta i pin della garanzia celle rare (§2.2).

---

## 3. Data augmentation

On-the-fly in training, **dopo la standardizzazione**, ordine fisso: (1) time shift, (2) time masking, (3) velocity masking, (4) amplitude scaling, (5) gaussian noise. Riempimento maschere: **0** (= media post-standardizzazione). Set fisso, nessuna ablation.

| Augmentation | Parametri (tempo=340 / velocità=100) | CE (C1, C2) | Viste SupCon (C3, C4) |
|---|---|---|---|
| Time shift circolare | shift ~ U{−10, …, +10} frame | p=0.5 | p=0.5 |
| Time masking | 1–2 maschere (equiprobabile), larghezza ~ U{5, …, 20} | p=0.5 | p=0.8 |
| Velocity masking | 1–2 maschere (equiprobabile), larghezza ~ U{2, …, 10} bin | p=0.5 | p=0.8 |
| Amplitude scaling | s ~ U(0.8, 1.2) | p=0.5 | p=0.8 |
| Gaussian noise | additivo, σ = 0.05 | — | p=0.5 |

- Le larghezze assumono 340×100: se la verifica assi dà dimensioni diverse, si riscalano in proporzione. Velocità mascherata con parsimonia (≤10%): è la feature che separa walking/running.
- **Vietato:** flip dell'asse velocità (inverte il segno Doppler), flip temporale (sit-down ↔ stand-up).
- Il time shift ±10 è un'augmentation di robustezza locale; **non** ha lo scopo di compensare la diversità di offset persa con lo stride 100 (giustificato dal solo budget, §1.2).
- Per SupCon ogni sample genera **2 viste = 2 augmentation indipendenti della stessa (finestra, antenna)**.

---

## 4. Mini-batch composition

### 4.1 C1, C2 (CE)
- **Batch size 256** (finestre), shuffling uniforme sui sample (finestra, antenna).
- **CE non pesata, scelta dichiarata:** la selezione/reporting è su macro-F1 mentre il training ottimizza CE uniforme; standard e mantenuta per confrontabilità. Da rivalutare solo se la tabella di contingenza del giorno 1 mostra squilibri forti (in tal caso si discute PRIMA di lanciare C1, non dopo).
- **Nota per il report (asimmetria C2 vs C4, dichiarata):** in C2 la testa AR-set vede i batch uniformi → la composizione di dominio dei batch rispecchia i conteggi di finestre e il gradiente del GRL è dominato dai set ricchi di dati; in C4 il sampler P×K bilancia i domini via round-robin. La baseline di maggioranza nel monitoring corregge la *metrica*, non il *gradiente*: asimmetria accettata, standard, una riga nel report.
- Fallback OOM (solo CE): batch 128 + gradient accumulation ×2 (per la CE preserva il batch effettivo). Con V-B non dovrebbe servire (§5.2).

### 4.2 C3, C4 (SupCon) — sampler P×K con vincolo trace-distinte
- P = n_att del train (atteso 8), K = ⌊256/P⌋ = 32 → batch 256 finestre, **512 viste**.
- **Campionamento per classe:** (1) round-robin sugli AR-set di train in cui la classe esiste (induce positivi "stessa attività, dominio diverso"); (2) dentro il set corrente, una trace senza reimmissione; (3) dalla trace: finestra uniforme, antenna uniforme; (4) **vincolo duro: max una finestra per trace per classe nel batch** finché le trace ≥ K; sotto, riuso con finestra diversa **e offset temporale minimo: due finestre della stessa trace nello stesso batch devono avere |Δstart| ≥ 340 (disgiunte)**. Motivo: a stride 100 due finestre "diverse" della stessa trace condividono fino al 71% del contenuto — senza il vincolo di offset il riuso reintrodurrebbe i positivi banali che il vincolo trace-distinte esiste per eliminare. Se una trace non ha abbastanza finestre disgiunte (~58 disponibili: non succede con K=32), si passa alla trace successiva del round-robin.
- **Nota dichiarata:** il round-robin sovracampiona i set con poche trace rispetto alle finestre — scelta intenzionale di bilanciamento di dominio; una riga nel report.
- **Sampler riseedato deterministicamente per epoca** (`seed_epoch = f(seed, epoca)`): il resume da fine epoca riproduce la sequenza dei batch senza salvare lo stato interno del sampler (§8.2).
- **Logging obbligatorio:** composizione media dei batch per epoca (AR-set distinti e trace uniche per classe, n. riusi di trace con relativo offset minimo osservato).
- **Fallback OOM (SupCon): ridurre K** (es. K=16, 256 viste) e dichiararlo. **VIETATA la gradient accumulation in fase A:** la SupCon è intra-batch; accumulare due mezzi-batch dà due loss indipendenti, non una a batch pieno.
- **Testa ambiente (C2, C4):** CE, nessun sampler dedicato; C2 usa il batch uniforme (vedi nota asimmetria §4.1), C4 lo stesso batch P×K. Classe **empty**: i suoi positivi cross-domain sono quasi puro segnale di dominio (stanza vuota = firma dell'ambiente) → tensione SupCon↔GRL attesa in C4, da monitorare e commentare.

---

## 5. Architetture

### 5.1 Rete di riproduzione (solo C0)
Rete SHARP-like: maxpool 2×2 ∥ conv 5@(2×2) ∥ conv 3@(1×1)→6@(2×2)→9@(4×4), concat, conv 3@(1×1), flatten, dropout 0.2, dense → n classi del paper. Fedele, nessuna modifica.

### 5.2 Backbone unico per C1–C4: ResNet-18 **variante V-B** (revisione FLOPs)
**Perché V-B.** La small-input della v3 (stem 3×3 s1 senza maxpool, stage 3–4 a stride (2,1)) costa **~73 GFLOPs/finestra ≈ 28× la stock**, non 3–4×: layer1 a piena risoluzione 340×100 e velocità tenuta a 50 bin fino a layer4 (che da solo vale metà del costo). *("Stock" qui = ResNet-18 a stride standard sul NOSTRO input 340×100 ≈ 2.5 GFLOPs — da non confondere col valore canonico su 224×224, ~3.6 GFLOPs.)* A 10–20 TFLOPS sostenuti su T4, C1 sarebbe costata 16–32 h e la fase A 49–97 h: fuori budget senza appello. L'obiettivo dichiarato riguarda però solo l'asse **velocità** (non schiacciarlo a ~3 bin): si può downsamplare il **tempo** presto e la velocità poco.

**Definizione V-B:**
- **Stem:** conv 3×3, **stride (2,1)** + maxpool 3×3, **stride (2,1)** → tempo /4 prima di layer1, velocità intatta.
- **Stage:** layer2 stride (2,2); layer3–4 stride (2,1). Mappa finale ≈ **11×50** (tempo×velocità): 50 bin di velocità conservati (obiettivo raggiunto).
- **Larghezza ×0.5:** canali 32/64/128/256 → **d_enc = 256** (GAP → feature 256-d).
- Costo: **~4.7 GFLOPs/finestra** (~1.8× la stock come definita sopra). Con ~100k campioni di train, mezza larghezza non è sottodimensionata (la rete di SHARP è due ordini di grandezza più piccola).
- **Memoria:** attivazioni ~1/16 della v3 → il checkpointing **probabilmente non serve nemmeno in fase A**; si verifica al gate del giorno 3, non si assume.
- **Activation checkpointing (se necessario):** `torch.utils.checkpoint` con **`use_reentrant=False`**; nota BN: il re-forward aggiorna le running statistics due volte per step nei segmenti checkpointati (momentum effettivo alterato) — bias piccolo e identico ovunque grazie al code path unico, ma va dichiarato e verificato sulla versione PyTorch in uso.
- **Nessuna ablation di architettura.** Escalation SOLO se il gate del giorno 2 fallisce (§10.1), in ordine: (a) step/epoca 400→300; (b) **aggiungere stride (2,2) a layer3 di V-B** → mappa finale 11×25, **~2.9 GFLOPs (~0.63× V-B)**; (c) riduzione epoche (griglia checkpoint fase A ridefinita dalla regola pre-committata di §6-C3). Ogni escalation si scrive nel changelog del file di split.

### 5.3 Teste — **tutte parametrizzate su d_enc (=256), nessun numero cablato**
- **Classificatore attività (C1, C2 end-to-end):** linear d_enc → n_att.
- **Projection head SupCon (C3, C4):** MLP d_enc → d_enc → 128, ReLU, L2-normalizzata; scartata dopo il pretraining.
- **Testa AR-set avversariale (C2, C4):** GRL → MLP d_enc → d_enc/2 → n_AR-set di train (ReLU, dropout 0.3).
- **Linear probe (ricetta unica per C1-lin, C2-lin, fase B di C3/C4, probe §7):** encoder CONGELATO, feature d_enc estratte una volta e cachate su disco. Linear d_enc → n_classi, Adam lr 1e-3, wd 1e-4, batch 256, max 30 epoche, early stopping su val macro-F1 dalla harness comune (con fusione antenne), patience 5, best checkpoint. Costo: minuti.
- **Due note dichiarative (per il report):** (i) le feature cachate sono estratte **senza augmentation** → la tabella apples-to-apples confronta gli encoder su feature pulite, mentre le righe end-to-end hanno addestrato (e selezionato) con augmentation nel loop: standard, dichiarato. (ii) La probe usa **Adam** mentre il core usa AdamW (§8.1): intenzionale — la ricetta della probe è congelata e separata; nessuna implicazione sui confronti (identica per tutte le righe).

---

## 6. Configurazioni sperimentali

C1–C4 su P2 (rotazione primaria), stesso backbone V-B, stessi split, stessa harness, seed 42. C0 su P1 con la rete SHARP-like.

### C0 — Riproduzione SHARP (P1) — time-box: max 3 giorni-persona
- CE; iperparametri come da paper/repo dove documentati; altrove: Adam lr 1e-4, batch 32, max 60 epoche, early stopping su val accuracy (patience 10).
- Classi = quelle dei numeri pubblicati (§1.1); eval = procedura repo SHARP via wrapper con logging invocazioni test (§2.1, §0.7). Deviazioni dichiarate: val 20%, classi se diverse dall'inventario, tutto ciò che il repo non documenta. Discrepanze coi numeri = risultato (cfr. Cominelli et al.), non fallimento.
- Se al giorno 6 C0 non è a posto: si congela e si scrive quello che c'è (non alimenta nulla a valle).

### C1 — Baseline CE (P2)
- Una testa attività, CE con label smoothing 0.1. Max 40 epoche (epoca = 400 step, §8.1), early stopping su **val macro-F1 con fusione antenne**, patience 10, best checkpoint.
- È il denominatore di tutti i confronti.

### C2 — CE + GRL
- Teste: attività (CE, ls 0.1) + AR-set (CE dietro GRL). `L = L_att + λ(p)·L_env`.
- **Ramp fisso:** `p = min(epoca/20, 1)`, `λ(p) = 2/(1+exp(−10p)) − 1` (λ≈1 dall'epoca 20; indipendente dall'early stopping — altrimenti C2 collassa in C1). Nel report: λ al checkpoint selezionato.
- **Monitoring (definizione univoca):** accuracy della testa AR-set **misurata sui batch di train, media per epoca** (la testa ambiente è definita sui domini di train; la val in-domain contiene gli stessi AR-set). Deve scendere verso la baseline di maggioranza al crescere di λ. Se resta alta → il GRL non morde; se l'accuracy attività crolla mentre quella AR-set resta alta → label shift o λ troppo aggressivo → **unica manopola: λ_max → 0.5**.
- Stessa durata/selezione di C1.

### C3 — SupCon (due fasi)
- **Fase A:** encoder + projection head, **SupCon** (Khosla et al.), label = attività, τ = 0.1, 2 viste, sampler P×K. **60 epoche fisse** (epoca = 400 step), nessun early stopping; checkpoint a **40/50/60** + `last.ckpt` per il resume.
- **Regola checkpoint pre-committata (vale anche sotto escalation (c) di §5.2):** con orizzonte fase A = H epoche, i checkpoint di selezione sono **⌈2H/3⌉, ⌈5H/6⌉, H** (H=60 → 40/50/60). La regola è fissata ORA perché la definizione operativa di "plateau" resti identica per C3 e C4 anche se l'orizzonte cambia.
- **Fase B (selezione + classificazione):** linear probe (ricetta §5.3) per ciascun checkpoint della griglia; si seleziona la migliore **val macro-F1**; **solo quello va a test**. È la definizione operativa di "plateau", identica per C4.
- Terminologia nel report: **SupCon**, non NT-Xent.

### C4 — SupCon + CE-GRL (configurazione completa) — **CHIUSA SENZA RUN (v5.2, 2026-07-17)**
- **Chiusura su evidenza:** il GRL non ha bersaglio sotto nessuna delle due famiglie di loss — le diagnostiche di dominio su C1/C2/C3 (feature di train, split interno trace-disjoint) danno ogni target di dominio alla propria baseline di maggioranza, e il precedente C2 mostra solo costi (−4.6 pt val, fit ridotto) senza invarianza aggiuntiva. Esito atteso: "C3 più rumore". Il design resta documentato qui sotto per il record; nessuna estensione riguarda C4 (§10.3).
- **Fase A:** SupCon (attività) + testa AR-set CE dietro GRL, insieme. `L = L_supcon + β·λ(p)·L_env`, **β = 0.5 fisso**, stesso ramp λ di C2, sampler P×K invariato.
- **Contingency se diverge (in ordine, una alla volta, dichiarate):** (a) β = 0.25; (b) ramp λ ritardato all'epoca 30; (c) fallback strutturale = fine-tuning avversariale dell'encoder C3.
- **Fase B:** identica a C3 (stessa griglia checkpoint da regola pre-committata, stessa probe, stessa regola) → C3 vs C4 direttamente confrontabile.

### Valutazione unificata a linear probe (apples-to-apples)
Per OGNI encoder (C1, C2, C3, C4) si valuta anche con linear probe attività (ricetta unica, feature cachate → minuti).
- **Tabella principale (qualità delle rappresentazioni):** C1-lin vs C2-lin vs C3 vs C4 — unica variabile: come è stato addestrato l'encoder.
- **Tabella secondaria (deployment):** C1 e C2 end-to-end.

| Config | Encoder addestrato con | Avversario | Eval | Protocollo |
|---|---|---|---|---|
| C0 | CE (rete SHARP-like) | — | end-to-end (eval repo SHARP, via logger) | P1 |
| C1 | CE | — | end-to-end | P2 |
| C1-lin | CE | — | linear probe | P2 |
| C2 | CE | CE-GRL | end-to-end | P2 |
| C2-lin | CE | CE-GRL | linear probe | P2 |
| C3 | SupCon | — | linear probe | P2 |
| C4 | SupCon | CE-GRL | linear probe | P2 |

(v5.2: la riga C4 documenta il design — configurazione chiusa senza run, §6-C4; la tabella finale del report ha uno stream in meno.)

### Ablation — solo quelle gratuite
1. Fusione antenne vs per-antenna al test (stessa forward, due aggregazioni) → appendice.
2. Probe diagnostici (§7) → feature cachate.
Eliminati (fuori budget): ablation augmentation, β, architettura, TCN.

---

## 7. Linear probe diagnostici (per C1–C4)

Encoder CONGELATO, ricetta §5.3, feature cachate:
- **Probe AR-set:** linear d_enc → n_AR-set; metrica: accuracy; **riferimento: baseline di maggioranza** (non 1/n — classi sbilanciate). Atteso: alto in C1/C3, verso la baseline in C2/C4. Verifica diretta che il GRL rimuove l'informazione di dominio — grafico chiave. **(v5.2: atteso NON verificabile su p2_lab — vedi esito sotto.)**
- **Probe persona:** linear d_enc → n_persone, stessa baseline. Interpretazione SOLO qualitativa (persona e ambiente confusi nei set AR).
- **Esito v5.2 (2026-07-17) — probe AR-set dichiarato underpowered su p2_lab:** la val ha 9 trace su 5 AR-set (AR-3 assente, AR-5 con 1 trace) e AR-1/AR-2 sono indistinguibili per costruzione (metadati identici, 55% delle finestre di val) → la baseline di maggioranza è irraggiungibile e l'atteso qui sopra non è verificabile in nessuna direzione. L'evidenza di invarianza per §9 poggia sulle **diagnostiche di dominio su feature di train** (split interno trace-disjoint dei 81 trace di train, target multipli — ar_set, ambiente, direct_path, persona, monitor — ciascuno contro la propria baseline di maggioranza, con controllo positivo `y`; notebook in `notebooks/diagnostics/`), replicate su C1, C2 e C3. Il probe persona resta qualitativo come da design. **Refinement 2026-07-18 (implementazione):** lo strumento della diagnostica di dominio (split interno trace-disjoint, derivazione dei target da `AR_SET_METADATA`, loop di probe e lettura fusa) è promosso a `sharp_har.diagnostics.domain_probe` — la replica S6-out pre-registrata (§10.3) lo rende codice ri-eseguito tra sessioni, lo stesso criterio della promozione NCM/kNN; matematica verificata **byte-identica** alla cella notebook-local delle tre sessioni archiviate (le righe C1/C2/C3 registrate restano valide); le sessioni successive (S6-out incluse) lo importano e restano thin.
- **Diagnostiche aggiuntive val-only (v5.2; feature cachate, solo seed 42, iperparametri fissati a priori e dichiarati):** accanto al probe lineare, **NCM** (centroidi per classe) e **kNN** (k=20) per C1/C2/C3 — confronto di classificatori a encoder fisso; kNN è il metro naturale della letteratura contrastive per C3; **concatenazione C1⊕C3** (512-d, ricetta probe invariata) con **controllo ensemble C1⊕C1′** (seed 42 ⊕ seed 43) per distinguere complementarità CE/SupCon da semplice diversità di ensemble. **Iperparametri dichiarati (fissati qui, a priori; ricetta identica per tutti gli encoder — è ciò che rende il confronto apples-to-apples):** feature L2-normalizzate e similarità coseno (metrica standard della letteratura contrastive; su vettori normalizzati equivale alla euclidea); NCM = un centroide per classe = media delle feature di train normalizzate, **rinormalizzata L2** (senza rinormalizzazione, con centroidi di norma diversa, argmax coseno e argmin euclidea divergerebbero), punteggio di classe = similarità al centroide; kNN: k=20 (dalla team call), punteggio di classe = frazione di voti fra i k vicini; fusione antenne = media dei punteggi di classe sulle 4 antenne, poi argmax (analogo della media dei softmax, §1.3); tie-break = classe con similarità media più alta. **Refinement 2026-07-18:** riferimento (centroidi NCM e vicini kNN) = pool unico dei campioni (finestra, antenna) di train sulle 4 antenne — la fusione dichiarata opera solo lato query, analogo §1.3 (era implicito nella prima esecuzione, ora dichiarato); il codice degli scorer vive in `sharp_har/diagnostics.py` (promosso da notebook-local con cross-review 2026-07-18, matematica verificata identica: i numeri C1 già registrati restano validi), il notebook diagnostico lo importa e resta thin; la concatenazione passa da `diagnostics.concat_caches` (assert bloccanti di allineamento per-riga tra le cache — trace/finestra/antenna/etichette; un mismatch è un errore di file, mai un riordino da riparare), ricetta probe invariata sul risultato. Nessun contatto col test; eventuale promozione a riga di test solo per decisione di team (nessuna prevista).

---

## 8. Ottimizzazione e training (comune a C1–C4 salvo dove indicato)

### 8.1 Iperparametri ed epoca
- **Epoca = 400 step fissi** (col sampler P×K "vedere tutto il dataset" è mal definito; la definizione a step rende i budget calcolabili). Orizzonti: CE max 40 epoche (16k step), fase A 60 epoche (24k step).
- AdamW, wd 1e-4; LR 1e-3, cosine decay sull'orizzonte della fase, warmup 5 epoche. (Il cosine è schedulato sull'orizzonte pieno anche se l'early stopping taglia prima: standard, nessuna azione.)
- AMP sempre attiva; gradient clipping a norma 1.0 (essenziale con GRL); cuDNN benchmark attivo (vedi nota determinismo, §0.5).

### 8.2 Checkpoint e resume (Colab free disconnette)
- Su Drive per run: **`last.ckpt` (sovrascritto ogni epoca) + `best.ckpt`** (+ i 3 checkpoint della griglia in fase A). Niente storico completo (Drive free = 15 GB, e ora ospita solo checkpoint: i dati stanno nella cartella condivisa dell'owner, §8.5).
- **Contenuto del checkpoint (completo per il resume):** pesi + optimizer + scheduler + **stato GradScaler AMP** + epoca + config + **stati RNG (torch, cuda, numpy, python)**. Il sampler P×K non salva stato interno: è **riseedato deterministicamente per epoca** (§4.2), quindi il resume da fine epoca riproduce la sequenza dei batch. Resume automatico da `last.ckpt` senza intervento manuale.

### 8.3 Compute (con V-B, da CONFERMARE al gate del giorno 2)
- ~4.7 GFLOPs/finestra → a 10–20 TFLOPS sostenuti su T4: **C1 ≈ 1–2.5 h; fase A ≈ 3–6 h**; fasi B e probe ≈ minuti.
- **Backup Kaggle (P100, 30 h/settimana per account):** in pratica l'AMP su P100 dà uno speedup misurato piccolo — mancano i tensor core e PyTorch AMP sfrutta raramente il picco fp16 della P100 (aritmetica half2, limiti di banda) — quindi si stimano tempi ≈ T4, non migliori. Se il canale Kaggle si attiva davvero, il pacchetto dati (piccolo, §8.5) si carica come **Kaggle Dataset privato** (10 minuti, elimina lo staging su quel canale).
- **Model selection in DG (limite dichiarato):** val in-domain (15% del train) → checkpoint potenzialmente sub-ottimi per l'OOD; con ~5 ambienti di train, sacrificarne uno come val OOD costa troppo. Si dichiara nel report.

### 8.4 Tabella budget run (core, seed 42 — nessuna run fuori tabella; ore da ricalibrare col gate)
| # | Run | Protocollo | Ore stimate (T4, V-B) | Owner |
|---|---|---|---|---|
| 1 | C0 | P1 | 1–2 | A |
| 2 | C1 | P2-lab | 1–2.5 | A |
| 3 | C2 | P2-lab | 1.5–3 | C |
| 4 | C3 fase A | P2-lab | 3–6 | B |
| 5 | ~~C4 fase A~~ — **chiusa senza run (v5.2, §6-C4)** | P2-lab | 0 (~7 h liberate) | — |
| — | Fasi B, C1-lin, C2-lin, probe | P2-lab | < 1 (totale) | C |
| | **Totale core** | | **≈ 10–20 GPU-h** | |

Con 3 account Colab il core sta nei giorni 4–9 con margine per re-run. Tutto il resto è estensione (§10.3).

**Emendamento v5.2 (cfr. `splits/CHANGELOG.md`):** la riga C4 non si esegue; le estensioni pre-registrate E1′ (2 run ≈ 4.6 h) + E2′ (1 run ≈ 2.5 h) ≈ 7.1 h — dello stesso ordine del budget liberato da C4 (proiezione di gate 6.87 h) e molto sotto l'inviluppo estensioni pre-v5.2 (E1+E2+E3: 15–35 h). Nessun'altra run si aggiunge alla tabella. **Emendamento 2026-07-19 (team call, cfr. CHANGELOG):** aggiunta l'unica run C3-ft ≈ 2 h (fine-tuning CE dell'encoder C3 da `C3/epoch40.ckpt`, seed 42) — totale estensioni ≈ 9.1 h, ancora sotto l'inviluppo pre-v5.2; con questa la tabella è definitivamente chiusa.

### 8.5 Residenza dati, I/O e staging (Colab) — RISOLTO in v5
- **Residenza:** il pacchetto vive nella cartella Drive condivisa **`DATASET_SHARP`** (`doppler_traces.zip` ~762 MB + `doppler_traces_S4_S5.zip` ~10 MB). Ogni membro aggiunge una **scorciatoia** alla cartella nel proprio Drive: lo storage conta solo sull'owner, i 15 GB free di ciascun account restano interi per i checkpoint. **Nessun ri-download da IEEE DataPort, mai.**
- **Staging a inizio sessione:** mount di Drive → copia dei due zip su `/content` (lettura *sequenziale* di file grandi: il caso in cui Drive va bene) → unzip locale. Atteso: pochi minuti. **Il tempo di staging si misura al gate del giorno 2 e si scrive nel go/no-go** come ogni altra assunzione; contingency (solo se lo staging misurato supera ~15 min o la dimensione decompressa sorprende): repack in fp16 `.npy` dopo l'inventario del giorno 1, previa verifica di adeguatezza fp16 contro il range in dB.
- **Durante il training: lettura SOLO da `/content`.** Drive montato = solo checkpoint. Lettura random da Drive in training = GPU idle + rate limiting: vietata.
- Con ~762 MB compressi, il dataset decompresso potrebbe perfino stare in RAM (~12 GB): si verifica al giorno 1; il default resta il **caricamento lazy per-trace da disco locale** (robusto comunque, costo nullo).
- Lo staging (mount → copia → unzip → primo batch) fa parte dello smoke test del giorno 2.

---

## 9. Valutazione

- **Metriche:** accuracy e **macro-F1**, per set di test (mai solo aggregate); confusion matrix per la rotazione primaria.
- **Macro-F1 con classi mancanti:** media solo sulle classi presenti nel ground truth del set valutato; il report elenca le assenti per set. Stessa definizione in val e test.
- **Unità di valutazione: la finestra a stride 340 (disgiunta, §1.2).** Predizione = argmax della media dei softmax sulle antenne. Per-antenna in appendice. (Eccezione: C0 usa l'aggregazione del repo SHARP, via logger.)
- **Reporting a seed singolo:** un numero per cella, seed 42 dichiarato, niente ±std con n=1; differenze < ~2 punti = "comparabili". Se E1/E3 aggiungono seed: media ± (min–max).
- **Grafici chiave (aggiornati v5.2):** (1) barre accuracy per configurazione e dominio; (2) tabella/figura delle **diagnostiche di dominio** (target × encoder, delta vs baseline di maggioranza propria — sostituisce il "prezzo dell'invarianza": il probe AR-set è underpowered, §7); (3) t-SNE/UMAP delle embedding per attività e per AR-set, **C1 vs C3** (C4 mai addestrata), **su feature di train** (refinement 2026-07-18: unico split con tutti e 6 gli AR-set e 2 ambienti — la val ha 9 trace su 5 AR-set con AR-3 assente, il test è un solo dominio (S7): nessuna struttura cross-domain da visualizzare; stessa portata dichiarata delle diagnostiche §7, confound di memorizzazione incluso, producibile senza contatto col test) — ricetta unica dichiarata: L2-norm delle feature, subsampling per trace (**8 campioni (finestra, antenna) per trace** — punto medio del range 5–10 della bozza, fissato per unicità della ricetta; tutti se la trace ne ha meno; refinement 2026-07-18: il conteggio è in campioni (finestra, antenna) — l'unità di riga delle cache — e NON in finestre ×4 antenne: la dicitura precedente "8 finestre" era ambigua; contare campioni tiene il tetto per trace a 8 punti e non include sistematicamente le 4 viste correlate della stessa finestra, servendo meglio la motivazione anti-duplicati del subsampling; è la semantica con cui la figura committata `reports/embeddings_c1_vs_c3.png` è stata prodotta), PCA-50 → t-SNE perplexity 30, seed 42, identica per tutti gli encoder; figura qualitativa, ancorata ai numeri dei probe.
- **Righe trasduttive pre-registrate (v5.2, team call 2026-07-17):** nella singola sessione di test (§0.7), **solo su C1** (baseline), righe **C1+AdaBN**, **C1+T3A** e **C1+AdaBN+T3A** (ordine di composizione: AdaBN → T3A; riga incondizionata — l'"opzionale" della call originaria avrebbe lasciato una scelta a sessione aperta, contro la nota §0.7: si esegue sempre). AdaBN = sostituzione delle statistiche BN dell'encoder con quelle del test set, senza etichette e senza toccare i pesi. **Specifica fissata (refinement 2026-07-18):** reset delle running statistics → un singolo forward di tutti i sample (finestra, antenna) di test (stride 340) in train-mode BN con `momentum=None` (media cumulativa = stimatore AdaBN standard), batch 256, ordine deterministico (trace in ordine di inventario, finestre in ordine temporale) → statistiche congelate per l'eval e per il caching post-AdaBN. Motivazione: col momentum di default (0.1) le statistiche pesano gli ultimi batch esponenzialmente → dipendenza forte e arbitraria dall'ordine; la media cumulativa è deterministica dati ordine e batch size dichiarati; T3A = aggiustamento a pseudo-prototipi della **testa lineare end-to-end di C1** (Iwasawa & Matsuo, NeurIPS 2021): prototipi inizializzati dai pesi della testa, aggiornati con le feature di test filtrate per entropia, predizione = argmax del prodotto scalare coi prototipi L2-normalizzati. Unico iperparametro: la dimensione del filtro M (support conservati per classe; il paper la esplora su griglia {1, 5, 20, 50, 100, ∞} selezionando via validazione — per noi sarebbe tuning): **fissato a priori M=20** (valore centrale della griglia), nessuna selezione su val o test. **Specifica operativa fissata (refinement 2026-07-18) — variante batch dichiarata, su feature cachate (numpy):** (1) pseudo-label ed entropia di ogni sample (finestra, antenna) coi prototipi iniziali (= righe della matrice pesi della testa C1, L2-normalizzate; bias scartato — nel paper i template iniziali sono i soli pesi); (2) per ogni classe si tengono gli M=20 sample a entropia minima fra quelli pseudo-etichettati a quella classe; (3) prototipo aggiustato = media di {prototipo iniziale} ∪ {support filtrati}, rinormalizzata L2; (4) predizione = identico percorso §1.3 (softmax dei logit sui prototipi, media sulle 4 antenne, argmax) — la riga C1+T3A differisce dalla riga C1 **solo nella testa**. Motivazione della variante batch: il protocollo online del paper è order-dependent — andrebbe scelto e dichiarato un ordine arbitrario da cui il risultato dipende, un grado di libertà in più senza beneficio sotto l'assunzione di deployment batch già dichiarata; la variante batch è deterministica e order-free (deviazione dal protocollo online, dichiarata nel report). Support pooled sulle 4 antenne: un solo insieme di prototipi, più support per classe, un solo code path di eval. Dichiarazioni nel report: regime trasduttivo (assunzione di deployment batch, contratto diverso dalla riga C1 liscia), attribuzione per differenza dalla stessa baseline, regola dei ~2 punti (§0.5). La combinazione richiede il caching delle feature post-AdaBN dentro la sessione (previsto nel template del notebook 05). **Refinement 2026-07-18 (implementazione, cross-review pre-freeze richiesta §10.4):** T3A vive in `sharp_har/transductive.py` (`t3a_head`, numpy puro sulle cache; la testa restituita — prototipi aggiustati come `weight`, `bias` = 0 — passa da `harness.evaluate_features`, quindi la riga differisce da C1 solo nella testa, come dichiarato). Precisazioni fissate contro il repo ufficiale (griglia filter_K confermata, ∞ = -1): i support sono **L2-normalizzati prima della media** (l'ufficiale somma support normalizzati e rinormalizza — media e somma coincidono dopo la rinormalizzazione); deviazioni già dichiarate rese esplicite: pseudo-label/entropie dai prototipi iniziali **senza bias** (l'ufficiale usa il classificatore originale col bias) e prototipo iniziale **mai soggetto al filtro** (l'ufficiale filtra anche i warmup support); tie deterministici (entropia pari → ordine di riga della cache, argmax → prima classe). AdaBN = flag `adapt_bn` su `harness.evaluate`/`harness.cache_features`: una passata di adattamento deterministica sullo stesso set prima dell'eval/estrazione, pesi intatti (solo i moduli BatchNorm in train-mode, `momentum=None`), stem/nomi di output con infisso `_adabn` (le righe/cache lisce non si sovrascrivono mai), flag registrato nel log §0.7 e nei metadati CSV/npz, batch fissato da assert (`ADAPT_BN_BATCH = 256`). Precisazione sulla dicitura "ordine di inventario": l'ordine operativo è l'ordine deterministico del dataset (trace_id ordinati, antenne crescenti, finestre in ordine temporale — l'eval loader con `shuffle=False`, l'unico ordine già canonico in ogni percorso di valutazione); con lo stimatore cumulativo l'ordine incide comunque solo tramite la composizione dell'ultimo batch parziale. Verifica sintetica locale registrata in STATUS (formula §9 vs re-implementazione indipendente, pesi intatti, stimatore cumulativo, determinismo bitwise).
- **Posizionamento nel report (previene l'obiezione SOTA):** "metodi consolidati di domain-invariant representation learning (SupCon, DANN/GRL) applicati a un protocollo LOEO rigoroso con diagnostica diretta dell'invarianza — non una proposta architetturale nuova". **Più 2–3 frasi di related work:** esistono linee più recenti per la DG in WiFi sensing — meta-learning (RF-Net), domain generalization dedicata (AirFi), pretraining self-supervised su CSI (AutoFi) — dichiarate fuori scope per budget e per il focus su protocollo e diagnostica (e, v5.2, con evidenza empirica: le loss di allineamento training-time condividono col GRL l'assenza di bersaglio su questo train); la **test-time adaptation è invece misurata in forma minima** (righe trasduttive v5.2, sopra); la scelta di metodi battle-tested con failure mode noti è deliberata. Converte "non abbiamo fatto SOTA" in "abbiamo scelto di non farlo, ed ecco la mappa".

---

## 10. Piano temporale e divisione del lavoro (ownership verticale)

### 10.1 Giorni 1–3 (tutti insieme): fondamenta e GATE
- **Giorno 1 (blocker):** scorciatoie Drive + staging su /content + inventario (con hop STFT, dtype, **mapping nomi Drive → AR-set, copertura AR-1…AR-7 (12 campagne attese, v5.1)**) + verifica assi + decisione classi (inventario E paper per C0) + tabella contingenza + policy NaN + split congelati (JSON su Git, con pin celle rare + assert) + μ/σ.
- **Giorno 2 (milestone obbligatoria): smoke test end-to-end CON GATE DI THROUGHPUT.** Run C1 di 2 epoche su mini-subset attraverso TUTTA la pipeline (mount → copia zip → unzip → dataloader → training → checkpoint su Drive → resume → harness → CSV). In più: **misura dei s/step reali** (CE, batch 256), **misura del tempo di staging**, e **ricalcolo della tabella §8.4 dai tempi misurati**, con regola scritta di go/no-go: **se C1 proiettata > 4 h o fase A proiettata > 8 h → escalation §5.2 PRIMA di lanciare qualsiasi run; se staging > ~15 min → repack fp16 (§8.5).** Il throughput è l'assunzione già fallita due volte (v2: volume dati; v3: costo modello): ora si misura, non si stima.
- **Giorno 3:** harness completa (fusione, macro-F1 per set, CSV, logging invocazioni test incluso wrapper C0) + caching feature + sampler P×K con logging (inclusi riusi e offset) + **test di memoria e throughput della fase A: un forward+backward a batch pieno (512 viste) col sampler reale.** Il percorso di memoria della fase A non deve arrivare vergine al giorno 4.

### 10.2 Giorni 4–9: core (budget §8.4, ricalibrato dal gate)
- **Persona A:** C0 (time-box 3 giorni) + C1.
- **Persona B:** C3 end-to-end (sampler, augmentation, fase A+B), poi C4 con C.
- **Persona C:** C2 end-to-end (GRL, ramp λ, monitoring), poi C4 con B; probe e C1-lin/C2-lin.

### 10.3 Giorni 10–12: estensioni a tempo residuo (in ordine; solo se il core è completo) — **ridefinite v5.2 (C4 chiusa)**
1. **E1′ — Replica seed 43 di C1 e C2** (rotazione primaria; emendamento della regola 5 registrato in `splits/CHANGELOG.md`): 2 run ≈ 4.6 GPU-h, dentro il budget liberato da C4. Misura il rumore di seed della pipeline (calibrazione empirica della banda §0.5, riportata come range osservato — nessun claim di significatività con n=2) e la robustezza dei finding C2 all'inizializzazione. Config `c1_ce_s43`/`c2_grl_s43` (identici ai gemelli seed-42 salvo name/seed), **cartelle Drive di run separate** (l'auto-resume da `last.ckpt` non deve incrociare i seed), push del codice prima di ogni lancio; **una riga di test pre-registrata ciascuna** (cella media ± range col gemello seed-42); probe/diagnostiche restano su seed 42 (asimmetria dichiarata). Le feature di C1_s43 si cachano (controllo C1⊕C1′, §7). Seed 44: solo su decisione esplicita di team.
2. **E2′ — Seconda rotazione living-out, solo C1** (test = S6, train = S1–S5+S7; bedroom-out infattibile, §2.2), seed 42 ≈ 2.5 GPU-h: **proprio split file congelato su Git PRIMA della run + proprie μ/σ + propri pin celle rare e assert**; contingenza della nuova rotazione ispezionata e dichiarata prima del lancio (classi assenti in val, celle vuote). Include la **replica della diagnostica di dominio sul train S6-out** (primo train con un ambiente non-bedroom intero: il test più pulito della causa strutturale, quasi gratis su feature cachate). Il suo test (= S6) si valuta nella medesima unica sessione §0.7. **Artefatti pre-dichiarati (v5.2 refinement):** `splits/p2_living.json` (protocol `P2-living`), `configs/c1_ce_s6out.yaml` (name `c1_ce_s6out`), cartella Drive di run `C1_s6out`.
3. **C3 non si replica (dichiarato):** costo ~3× per seed, finding già replicati cross-encoder/cross-macchina, griglia 40/50/60 piatta (0.7 pt) come evidenza di stabilità della rappresentazione; nel report C3 è dichiarata single-seed.
Nulla del core si rifà; estensioni a metà si riportano dichiarandolo. Le run seed si valutano end-to-end nella sessione unica; reporting media ± (min–max), §9.

### 10.4 Giorni 13–15: report e presentazione
- Freeze del codice a **deadline −2 giorni**: **deadline 2026-07-30 → freeze 2026-07-28** (v5.2: sostituisce "giorno 12" — i giorni-piano non tracciano più il calendario reale; date anche in STATUS). Il freeze include (v5.2): T3A, l'aggiunta AdaBN alla harness, `viz.plot_embeddings` e l'estensione del template del notebook 05 (righe trasduttive, caching post-AdaBN, assert della lista righe §0.7) — tutto con cross-review. Il report dichiara: seed singolo (o quanti raggiunti), val in-domain, "AR-set invariance", riproduzione parziale C0 (incluse classi e val 20%), CE non pesata, probe su feature non aumentate, asimmetria batch GRL C2 vs C4, nota checkpointing↔BN se usato, no bit-exactness, ogni contingency ed escalation attivata. **Aggiunte v5.2:** C4 chiusa su evidenza senza run; §7 underpowered (evidenza §9 dalle diagnostiche di dominio); probe e diagnostiche solo su seed 42; righe trasduttive con assunzione di deployment dichiarata, iperparametri fissati senza tuning e variante batch di T3A (deviazione dichiarata dal protocollo online del paper); macro-F1 di val a 5 classi effettive vs test a 8 (non scale-comparabili — audit split); fase A a 300 step/epoca vs 400 CE (escalation (a)); E2′ con proprio split e μ/σ.

---

## 11. Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Pacchetto Doppler incompleto/diverso (shape, hop STFT, **copertura set nella copia Drive**) | Inventario giorno 1 con checklist estesa + mapping nomi → AR-set, prima di ogni decisione |
| **Residenza dati tra sessioni** (era il gap E-2) | **Risolto:** cartella Drive condivisa (762 MB), scorciatoie senza costo quota, staging sequenziale misurato al gate giorno 2; contingency repack fp16 |
| **Throughput reale < stime (assunzione già fallita 2 volte)** | **Gate del giorno 2: s/step e staging misurati → budget ricalcolato → go/no-go scritto; escalation §5.2 predefinita** |
| OOM/instabilità fase A scoperti tardi | Test memoria fase A a batch pieno anticipato al giorno 3 |
| Bug di integrazione tra i 3 stream | Smoke test end-to-end obbligatorio al giorno 2 |
| Leakage da windowing | Split per trace, code review incrociata del dataloader |
| Sbirciata sul test a metà progetto | Principio 7 + logging invocazioni harness sul test, **C0 incluso (wrapper)** |
| Cella rara (AR-set, attività) svuotata dallo split | **Pin 1 trace per cella in train + assert bloccante nello script di split** |
| Label shift attività×AR-set → GRL rimuove attività | Tabella contingenza; monitoring C2 (definito su train, per epoca); λ_max→0.5; caveat |
| Positivi banali nei batch SupCon | Vincolo trace-distinte + **offset minimo ≥340 sui riusi** + logging composizione batch |
| OOM fase A | Ridurre K (mai gradient accumulation in fase A); checkpointing `use_reentrant=False` con nota BN |
| ~~C4 instabile~~ (chiusa v5.2 senza run, §6-C4) | Contingency ordinate (mai attivate): β=0.25 → ramp ritardato → fine-tuning avversariale di C3 |
| Colab disconnette / GPU debole | `last.ckpt` completo (GradScaler+RNG) + resume automatico; sampler riseedato per epoca; backup Kaggle (+ Kaggle Dataset privato se attivato) |
| Drive pieno / I/O lento | Solo last+best(+3 fase A); dati sulla cartella condivisa dell'owner; staging su /content, mai lettura da Drive in training |
| Esplosione di run | Tabella budget vincolante; estensioni solo nell'ordine §10.3 |
| Risultati C0 ≠ paper | Classi del paper + eval del repo (via logger) + deviazioni dichiarate; è un risultato, non un fallimento |
| Risultato a seed singolo rumoroso | Soglia dichiarata (<2 punti = comparabili); E1/E3 sulle config chiave |
