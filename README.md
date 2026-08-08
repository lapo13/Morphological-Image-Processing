# Parallel Computing — Esame

Benchmark di operazioni di morfologia matematica su immagini in scala di grigi
(erosione, dilatazione, opening, closing), applicate a mosaici composti da
immagini del [Brain Cancer MRI dataset](brain-cancer-mri-dataset/), parallelizzate
con OpenMP e misurate al variare del numero di thread e della dimensione
dell'immagine di input.

## Struttura del repository

```
Esame/
├── brain-cancer-mri-dataset/   # dataset di immagini MRI usate come tile per i mosaici
└── OpenMP/                     # implementazione C + OpenMP e strumenti di analisi
    ├── src/                    # sorgenti C
    ├── external/                # librerie header-only di terze parti (stb_image)
    ├── analysis/                 # script Python per l'analisi dei risultati
    └── experiment_run/           # output dei benchmark (CSV + grafici, non versionato)
```

## Come funziona

### Pipeline dati

1. `build_mosaic_image` ([image.c](OpenMP/src/image.c)) compone un'immagine di
   test incollando tile quadrati caricati dal dataset, producendo
   immagini via via più grandi al crescere della griglia.
2. Ogni operazione morfologica (`image_erosion`, `image_dilation` in
   [morphologies.c](OpenMP/src/morphologies.c)) segue lo schema classico
   "pad → sliding window sulla struttura → crop": l'immagine viene prima
   espansa (`pad_image`) di metà lato dell'elemento strutturante, poi si scorre
   una finestra calcolando min (erosione) o max (dilatazione)
   dei pixel coperti dall'elemento strutturante, infine si ritaglia (`crop_image`)
   il bordo aggiunto dal padding.
3. `image_opening` = erosione poi dilatazione; `image_closing` = dilatazione poi
   erosione, entrambe implementate come composizione delle due primitive.

### Parallelizzazione OpenMP

Tutto il lavoro (allocazione, caricamento tile, loop centrale delle operazioni)
gira dentro un'unica regione `#pragma omp parallel`, distribuita con
`#pragma omp for` sulle righe dell'immagine; le sezioni non parallelizzabili
(allocazioni, aggiornamento di puntatori condivisi) usano `#pragma omp single`
o `#pragma omp master` per essere eseguite da un solo thread, sfruttando le
barriere implicite di questi costrutti per la sincronizzazione fra le fasi.

### Metodologia di benchmark ([main.c](OpenMP/src/main.c))

Per ogni combinazione di **thread count** (`TEST_THREAD_COUNTS`: 1, 2, 4, 6, 8,
10) e **dimensione immagine** (`TEST_GRID_SIZES`: da 1×1 a 5×4 tile), il
programma:

1. apre una regione `#pragma omp parallel num_threads(N)` con `N` thread;
2. costruisce il mosaico della dimensione corrente;
3. esegue `WARMUP_RUNS` iterazioni di riscaldamento (scartate) per ciascuna
   delle 4 operazioni **evitando l'effetto della cold-cache**;
4. esegue `TIMED_RUNS` iterazioni cronometrate, misurando **solo il loop
   centrale** dell'operazione (padding e cropping sono esclusi dal tempo
   misurato, per le operazioni di `opening` e `closing` si utilizza una somma dei tempi delle computazioni);
5. appende ogni singola misurazione (non solo media/minimo) al CSV dei
   risultati tramite [logger.c](OpenMP/src/logger.c).

Il file `experiment_run/results.csv` ha quindi una riga per ogni run
cronometrata:

```
run_id,threads,image_rows,image_cols,operation,run_index,seconds
```

`run_id` è il timestamp fissato a inizio programma e condiviso da tutte le
righe prodotte nella stessa esecuzione.

## Come eseguire

### Build ed esecuzione del benchmark

```sh
cd OpenMP
make            # compila in build/, produce l'eseguibile ./main
./main          # esegue tutte le combinazioni thread x dimensione, scrive experiment_run/results.csv
```
