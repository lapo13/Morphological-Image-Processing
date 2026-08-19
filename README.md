# Parallel Computing — Esame

Benchmark di operazioni di morfologia matematica su immagini in scala di grigi
(erosione, dilatazione, opening), applicate a mosaici composti da
immagini del [Brain Cancer MRI dataset](brain-cancer-mri-dataset/), parallelizzate
con OpenMP e misurate al variare del numero di thread e della dimensione
dell'immagine di input.

## Struttura del repository

```
Esame/
├── brain-cancer-mri-dataset/   # dataset di immagini MRI usate come tile per i mosaici
├── OpenMP/                     # implementazione C + OpenMP e strumenti di analisi
│   ├── app/                    # sorgente del main
│   ├── src/                    # sorgenti C
│   ├── include/                # header dei sorgenti
│   ├── external/               # librerie header-only di terze parti (stb_image)
│   └── experiment_run/         # output dei benchmark (CSV + grafici, non versionato)
└── CUDA/                       # implementazione CUDA (in sviluppo), stesso layout
    ├── app/ src/ include/ external/
    └── ...
```

## Come funziona

### Pipeline dati

1. `scan_dataset` ([image.c](OpenMP/src/image.c)) enumera a runtime i tile `.jpg`
   sotto la radice del dataset (6056 immagini 512×512 nelle tre classi
   `brain_menin`/`brain_tumor`/`brain_glioma`), ordinandoli per path.

2. `build_mosaic_image` ([image.c](OpenMP/src/image.c)) compone un'immagine di
   test incollando tile quadrati caricati dal dataset, producendo
   immagini via via più grandi al crescere della griglia. Le operazioni composte
   lavorano su un **batch di `PIPELINE_BATCH` mosaici distinti**: ognuno usa una
   fetta diversa del dataset, per un totale di `righe × colonne × PIPELINE_BATCH`
   tile per configurazione.

3. Ogni operazione morfologica (`image_erosion`, `image_dilation`) in
   [morphologies.c] segue lo schema classico "pad → sliding window sulla struttura → crop": l'immagine viene prima
   espansa (`pad_image`) di metà lato dell'elemento strutturante, poi si scorre
   una finestra calcolando min (erosione) o max (dilatazione)
   dei pixel coperti dall'elemento strutturante, infine si ritaglia (`crop_image`)
   il bordo aggiunto dal padding.

4. `image_opening` = erosione poi dilatazione. Non è però una semplice
   composizione sequenziale delle due primitive: è realizzata come **pipeline
   a due stadi**.

### Parallelizzazione OpenMP — erosione e dilatazione

Tutto il lavoro (allocazione, caricamento tile, loop centrale delle operazioni)
gira dentro un'unica regione `#pragma omp parallel`, distribuita con
`#pragma omp for` sulle righe dell'immagine; le sezioni non parallelizzabili
(allocazioni, aggiornamento di puntatori condivisi) usano `#pragma omp single`
o `#pragma omp master` per essere eseguite da un solo thread, sfruttando le
barriere implicite di questi costrutti per la sincronizzazione fra le fasi.

### Pipeline — opening

Eseguire i due stadi in sequenza sull'intero batch lascia il team in gran parte
fermo durante le fasi seriali (allocazioni, `single`) di ogni immagine.
L'opening usa invece una **pipeline a due stadi**.

Lo split è **manuale**, via `omp_get_thread_num()`, non con `#pragma omp task`:
un task è eseguito da un singolo thread, quindi per assegnarne `N/2` a uno
stadio servirebbe parallelismo annidato che comporta il respawn del team che è un'operazione costosa.
Il partizionamento manuale riusa il team già aperto.

Ne discende un vincolo preciso: `#pragma omp for`, `single` e `master` sono
costrutti di worksharing che **richiedono l'intero team**. Per questo:

- gli helper di calcolo della pipeline (`erosion_helper`/`dilation_helper` in
  [morphologies.c](OpenMP/src/morphologies.c)) non contengono alcun costrutto di
  worksharing: il proprio intervallo di righe lo ricavano da `row_range()`;
- `pad_image` e `crop_image` restano invece **in contesto team completo**, fuori
  dallo split, dove i loro `single`/`for` sono perfettamente leciti. È quindi
  sovrapposta la sola fase di calcolo — che è anche quella dominante, dato che
  con un elemento strutturante 5×5 sono 25 letture per pixel di output.

Ogni immagine avanza *in place* (grezza → stadio 1 → stadio 2) e i due stadi
usano buffer distinti, così le due metà non si sovrappongono mai sulla stessa
memoria. Con un solo thread si ricade su una composizione sequenziale.

### Baseline sequenziale e decomposizione dello speedup

[sequential.c](OpenMP/src/sequential.c) contiene un'implementazione in **C puro,
senza alcuna direttiva OpenMP** (il tempo si misura con `clock_gettime`).
Il kernel è identico a quello parallelo —
stesso schema pad → finestra scorrevole con maschera → crop — così l'unica
variabile fra le due versioni è la parallelizzazione.

Lo stesso sorgente viene compilato **due volte** con flag diversi, perché
vettorizzazione e threading sono due forme distinte di parallelismo e vanno
misurate separatamente:

| variante | come | ruolo |
|---|---|---|
| `sequential_scalar` | `-fno-vectorize -fno-slp-vectorize` | nessun parallelismo di alcun tipo → riferimento per lo speedup **totale** |
| `sequential_simd` | flag standard, auto-vettorizzato da clang | riferimento per lo speedup da **threading** |

Da cui la decomposizione moltiplicativa:

```
Speedup_vettorizzazione = T_scalar / T_simd
Speedup_threading       = T_simd   / T_parallel(p)
Speedup_totale          = T_scalar / T_parallel(p)   = prodotto dei due
```

Servono entrambi i riferimenti. Il **totale** va misurato contro il codice
scalare, perché il SIMD *è* parallelismo (data parallelism) e non appartiene al
baseline. La curva di **thread scaling** va invece riferita a `T_simd`: usare
quella scalare vi incorporerebbe un fattore costante di ~3,5×, spingendo
l'efficienza per-thread sopra 1 ovunque e rendendo l'analisi di Amdahl priva di
significato.

Nessuno dei due riferimenti è la versione OpenMP eseguita a 1 thread, che paga
comunque barriere, regioni `single` e dispatch del runtime: usare quella
nasconderebbe parte del costo della parallelizzazione.

### Metodologia di benchmark ([main.c](OpenMP/app/main.c))

Per ogni **dimensione immagine** il programma esegue prima il baseline
sequenziale, poi la versione OpenMP per ogni **thread count**. Ogni misura
prevede `WARMUP_RUNS`(per evitare l'effetto della cold-cache) seguite da `TIMED_RUNS` cronometrate;
**ogni singola misurazione** viene salvata in un CSV.

La versione OpenMP ha `#pragma omp simd` **sempre attivo** sui loop interni: il
SIMD non è un asse crociato con gli altri, il suo contributo si isola dal
confronto fra i due baseline.

> **"C puro senza pragma" non significa scalare.** Clang a `-O2`
> auto-vettorizza comunque: sulla macchina di test lo stesso sorgente sequenziale,
> privo di qualsiasi direttiva, è già più veloce della propria compilazione
> con `-fno-vectorize`. Per avere codice davvero scalare serve il flag esplicito.

#### Parametri

| Parametro | Valori | Dove |
|---|---|---|
| Thread count | 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 | `TEST_THREAD_COUNTS` |
| Dimensione immagine | 512², 1024², 2048², 4096² (griglie 1×1 … 8×8 di tile 512²) | `TEST_GRID_SIZES` |
| Elemento strutturante | 3×3, 5×5, 9×9 (tutti 1) | `TEST_SE_SIZES` |
| Profondità pipeline | 8 immagini | `PIPELINE_BATCH` |
| Warm-up / misure | 2 / 10 per configurazione | `WARMUP_RUNS`, `TIMED_RUNS` |
| Scheduling | `guided` sul loop centrale (vedi sotto) | [morphologies.c](OpenMP/src/morphologies.c) |

Il tetto di 20 thread è 2× i core fisici della macchina di test, così da coprire
anche la regione di oversubscription.

L'elemento strutturante varia il **carico aritmetico** del kernel: 9, 25 o 81
letture per pixel di output a parità di traffico di memoria. È l'asse che sposta
l'operazione da memory-bound verso compute-bound, e infatti lo scaling migliora
in modo monotono al crescere dell'SE — su `opening` a 512², lo speedup da
threading passa da 1,3× (SE 3×3) a 2,9× (SE 9×9).

#### Cosa viene cronometrato

Tutte e tre le operazioni misurano la **stessa cosa**: il wall-clock
dell'intero batch, padding e cropping inclusi, diviso per il numero di immagini.
I tempi sono quindi un valore per-immagine e sono direttamente confrontabili fra
loro.

Cronometrare il batch anziché la singola immagine serve anche alla qualità della
misura: misurare 8nimmagini per volta riducen il peso relativo del rumore di scheduling.
La stabilità è recuperata per via statistica,mediando su `TIMED_RUNS`
misure di cui il CSV conserva ogni singolo campione.

#### Scelta dello scheduling

Le politiche sono state confrontate sperimentalmente sul loop centrale
(erosione, batch di 8 immagini 1024², SE 5×5, tempi in ms per immagine):

| schedule | 6 thread | 10 thread | 16 thread | 20 thread |
|---|---|---|---|---|
| `static` | 1,119 | 1,306 | 1,338 | 1,396 |
| `static,1` | 1,120 | 1,310 | 1,372 | 1,430 |
| `static,16` | 1,108 | 1,289 | 1,356 | 1,411 |
| `dynamic,1` | 1,125 | 1,206 | 1,328 | 1,415 |
| `dynamic,16` | 1,107 | 1,196 | **1,292** | **1,367** |
| `guided` | **1,066** | **1,177** | 1,298 | 1,407 |

`guided` è la scelta adottata: vince fino a 16 thread — con circa **10% di
vantaggio su `static` a 10 thread** — e resta vicino al migliore altrove. La
ragione è la topologia eterogenea della CPU di test: con `static` i 4 efficiency core
ricevono lo stesso numero di righe dei 6 performance core e diventano i
ritardatari su cui l'intero team si sincronizza alla barriera, mentre `guided`
assegna blocchi via via più piccoli e assorbe lo sbilanciamento. `dynamic,16`
prevale solo con 20 thread, dove il costo di sincronizzazione
per blocco pesa meno dello sbilanciamento residuo.

I `for` di copia in `pad_image`/`crop_image`/`copy_matrix` mantengono lo
scheduling di default: sono operazioni di sola memoria, con costo per iterazione
uniforme, dove la politica incide molto meno del loop centrale.

Il file `experiment_run/results.csv` ha quindi una riga per ogni run
cronometrata:

```
run_id,implementation,threads,se_size,image_rows,image_cols,operation,run_index,seconds
```

`run_id` è il timestamp fissato a inizio programma e condiviso da tutte le righe
prodotte nella stessa esecuzione. `implementation` vale `sequential_scalar`,
`sequential_simd` (entrambi baseline, sempre con `threads=1`) o `parallel`, ed è
la colonna su cui si costruisce la decomposizione dello speedup.

## Validazione della correttezza

La versione parallela è verificata **pixel per pixel** contro il baseline
sequenziale **scalare**, perché se i due coincidono
allora né la vettorizzazione né il threading hanno alterato l'immagine. Il
confronto copre tutte e tre le operazioni, 3 dimensioni di elemento
strutturante (3×3, 5×5, 9×9) e 6 configurazioni di thread, per 54 configurazioni totali.

## Ambiente di test

| | |
|---|---|
| CPU | Apple M2 Pro — 10 core (6 performance + 4 efficiency) |
| RAM | 16 GB unified memory |
| OS | macOS 26.6.2 |
| Compilatore | Apple clang 21.0.0, `-O2 -Xclang -fopenmp` |
| Runtime OpenMP | LLVM libomp 22.1.8 (Homebrew) |

## Dataset

[Brain Cancer MRI dataset](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset)
— 6056 immagini JPEG 512×512 in scala di grigi, tre classi (`brain_glioma`,
`brain_menin`, `brain_tumor`). Nessun preprocessing è applicato oltre alla conversione a singolo canale
fatta da `stb_image` in fase di caricamento, che resta fuori dalle misure.

## Come eseguire

### Build ed esecuzione del benchmark

```sh
cd OpenMP
make            # compila in build/, produce l'eseguibile ./main
./main          # baseline sequenziale + sweep thread x dimensione -> experiment_run/results.csv
make test       # validazione parallelo vs sequenziale
```
