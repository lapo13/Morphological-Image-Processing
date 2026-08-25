# Parallel Computing — Esame

Benchmark di operazioni di morfologia matematica su immagini in scala di grigi
(erosione e opening), applicate a mosaici composti da
immagini del [Brain Cancer MRI dataset](brain-cancer-mri-dataset/). Il repository
contiene un'implementazione OpenMP per CPU e una CUDA per GPU, analizzate con
metodologie distinte e coerenti con i rispettivi modelli di esecuzione.

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
└── CUDA/                       # implementazione CUDA e analisi naïve/shared
    ├── app/ src/ include/ external/
    ├── analysis/               # analizzatore dei tempi kernel e grafici
    └── experiment_run/         # risultati CSV (non versionati)
```

## Come funziona

### Pipeline dati e operazioni OpenMP

1. `scan_dataset` ([image.c](OpenMP/src/image.c)) enumera a runtime i tile `.jpg`
   sotto la radice del dataset (6056 immagini 512×512 nelle tre classi
   `brain_menin`/`brain_tumor`/`brain_glioma`), ordinandoli per path.

2. `build_mosaic_image` ([image.c](OpenMP/src/image.c)) compone un'immagine di
   test incollando tile quadrati caricati dal dataset, producendo
   immagini via via più grandi al crescere della griglia. In OpenMP le operazioni
   composte lavorano su un **batch di `PIPELINE_BATCH` mosaici distinti**: ognuno usa una
   fetta diversa del dataset, per un totale di `righe × colonne × PIPELINE_BATCH`
   tile per configurazione.

3. In OpenMP ogni operazione morfologica segue lo schema classico
   "pad → sliding window sulla struttura → crop": l'immagine viene prima
   espansa (`pad_image`) di metà lato dell'elemento strutturante, poi si scorre
   una finestra calcolando min (erosione) o max (dilatazione)
   dei pixel coperti dall'elemento strutturante, infine si ritaglia (`crop_image`)
   il bordo aggiunto dal padding.

   CUDA non materializza il padding: i kernel applicano direttamente il valore
   neutro quando una coordinata della finestra cade fuori dall'immagine.

4. `image_opening` = erosione poi dilatazione. Non è però una semplice
   composizione sequenziale delle due primitive: è realizzata come **pipeline
   a due stadi**.

Le operazioni **misurate** sono due, erosione e opening.
La dilatazione resta nel codice, perché è il secondo stadio
dell'opening, ma non viene cronometrata a sé: ha la stessa struttura di loop
dell'erosione e ne segue l'andamento, quindi come
misura non aggiungerebbe informazione.

### Parallelizzazione OpenMP — erosione

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

Da questa scelta deriva un vincolo: `#pragma omp for`, `single` e `master` sono
costrutti che **richiedono l'intero team**. Per questo:

- gli helper di calcolo della pipeline (`erosion_rows`/`dilation_rows` in
  [morphologies.c](OpenMP/src/morphologies.c)) non contengono alcun costrutto:
  il proprio intervallo di righe lo ricavano da `row_range()`;
- `pad_image` e `crop_image` restano invece fuori dallo split,
   dove i loro `single`/`for` sono perfettamente leciti. È quindi
  sovrapposta la sola fase di calcolo — che è anche quella dominante, dato che
  con un elemento strutturante 5×5 sono 25 letture per pixel di output.

Ogni immagine avanza *in place* (grezza → stadio 1 → stadio 2) e i due stadi
usano buffer distinti, così le due metà non si sovrappongono mai sulla stessa
memoria. Con un solo thread si ricade su una composizione sequenziale.

### Baseline sequenziale e decomposizione dello speedup

[reference.c](OpenMP/src/reference.c) contiene un'implementazione in **C puro,
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

Il **totale** va misurato contro il codice
scalare, perché il SIMD *è* parallelismo e non appartiene al baseline.
La curva di **thread scaling** va invece riferita a `T_simd`: usare
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
contributo del SIMD si isola dal confronto fra i due baseline.

> **"C puro senza pragma" non significa scalare.** Clang a `-O2`
> auto-vettorizza comunque: sulla macchina di test lo stesso sorgente sequenziale,
> privo di qualsiasi direttiva, è già più veloce della propria compilazione
> con `-fno-vectorize`. Per avere codice davvero scalare serve il flag esplicito.

#### Parametri

| Parametro | Valori | Dove |
|---|---|---|
| Thread count | 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 | `TEST_THREAD_COUNTS` |
| Dimensione immagine | 512², 1024², 2048², 4096² (griglie 1×1 … 8×8 di tile 512²) | `TEST_GRID_SIZES` |
| Elemento strutturante | 3×3, 5×5, 9×9 (tutti 1) | `TEST_SE_SIZES` |
| Profondità pipeline | 16 immagini | `PIPELINE_BATCH` |
| Weak scaling | mosaico di `p × 2` tile, batch 8 | `WEAK_COLS`, `WEAK_BATCH` |
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

Entrambe le operazioni misurano la **stessa cosa**: il wall-clock
dell'intero batch, padding e cropping inclusi, diviso per il numero di immagini.
I tempi sono quindi un valore per-immagine e sono direttamente confrontabili fra
loro.

Cronometrare il batch anziché la singola immagine serve anche alla qualità della
misura: misurare più immagini per volta riduce il peso relativo del rumore di scheduling.
La stabilità è recuperata per via statistica, mediando su `TIMED_RUNS`
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
run_id,mode,implementation,threads,se_size,image_rows,image_cols,operation,run_index,seconds
```

`run_id` è il timestamp fissato a inizio programma e condiviso da tutte le righe
prodotte nella stessa esecuzione. `implementation` vale `sequential_scalar`,
`sequential_simd` (entrambi baseline, sempre con `threads=1`) o `parallel`, ed è
la colonna su cui si costruisce la decomposizione dello speedup. `mode` distingue
i due esperimenti di scalabilità descritti sotto.

### Strong scaling e weak scaling

Il programma esegue **due esperimenti distinti**:

| `mode` | cosa varia | legge di riferimento |
|---|---|---|
| `strong` | problema fisso, thread crescenti | **Amdahl**: lo speedup satura sulla frazione seriale |
| `weak` | il carico cresce con i thread | **Gustafson-Barsis**: `S(p) = p − f·(p−1)` |

Nel weak scaling a `p` thread il mosaico ha `p` righe di tile invece di una, così
il **lavoro per thread resta costante**: si scalano le righe perché sono l'asse su
cui il loop parallelo distribuisce le iterazioni. Il batch resta fisso
(`WEAK_BATCH`), altrimenti cambierebbe anche la profondità della pipeline.

La metrica riportata è lo **scaled speedup**, `p · T_base / T(p)` con `T_base` il
tempo del baseline sequenziale vettorizzato sul carico base: idealmente il tempo
resterebbe costante e lo scaled speedup varrebbe `p`. Dai punti misurati si stima
la frazione seriale `f` della legge di Gustafson, che il
grafico riporta accanto alla curva.

## Implementazione CUDA

L'esperimento CUDA non confronta direttamente CPU e GPU: sono architetture con
modelli di esecuzione differenti e il server CUDA non condivide una piattaforma
hardware di riferimento con i test OpenMP. La GPU viene quindi studiata
separatamente, osservando il tempo dei kernel al crescere del problema e
confrontando due implementazioni dello stesso algoritmo morfologico generico:

- `naive`, che legge direttamente dalla global memory;
- `shared`, che carica un tile con halo in shared memory e riusa i pixel fra
  finestre adiacenti.

Non vengono impiegati filtri separabili o deque monotone. Queste ottimizzazioni
dipenderebbero dalla forma dell'elemento strutturante e cambierebbero la
complessità dell'algoritmo soltanto in una delle due implementazioni, rendendo
impossibile attribuire lo speedup all'uso della shared memory. Entrambi i kernel
rispettano invece la maschera generica memorizzata in constant memory.

### Kernel naïve

Il mapping è quello CUDA naturale, senza griglie persistenti o grid-stride loop:

```c
col   = blockIdx.x * blockDim.x + threadIdx.x;
row   = blockIdx.y;
chunk = blockIdx.z;
```

Ogni thread calcola un pixel e attraversa tutte le celle attive dell'elemento
strutturante. Il blocco del benchmark contiene 256 thread, cioè 8 warp, e la
griglia ha dimensioni `ceil(width / 256) × height × batch`. Il numero di blocchi
non è un parametro del benchmark e non viene registrato: è una conseguenza
della dimensione dell'immagine e CUDA li distribuisce autonomamente sugli SM.

### Kernel tiled/shared

Anche il kernel shared usa blocchi da 256 thread, ma ogni thread produce quattro
colonne adiacenti per 8 righe di output. Un blocco copre quindi un tile di output
da `1024 × 8` pixel:

```text
larghezza output = 256 thread × 4 pixel = 1024 pixel
altezza output   = SHARED_TILE_ROWS = 8 pixel
```

Il tile di input comprende l'halo richiesto dall'elemento strutturante. L'halo
orizzontale viene arrotondato a quattro byte e il caricamento cooperativo usa
`uchar4`: i thread consecutivi trasferiscono word consecutive e gli accessi
globali risultano coalescenti. I pixel esterni all'immagine sono sostituiti con
l'elemento neutro della riduzione, 255 per l'erosione e 0 per la dilatazione.

Dopo una sola barriera `__syncthreads()`, ogni thread calcola i propri quattro
output orizzontali. Le quattro finestre sovrapposte leggono `K+3` byte distinti
per ogni riga dell'SE invece di quattro gruppi separati da `K`, ma ciascun
output continua a eseguire la riduzione generica sulle celle attive della
maschera. La griglia è
`ceil(width / 1024) × ceil(height / 8) × batch`.

Con l'SE massimo 9×9 il tile occupa:

```text
(8 + 2×4) righe × (1024 + 2×4) byte = 16.512 byte di shared memory
```

Sulla build osservata per `sm_86`, la compilazione con `-Xptxas -v` riporta 33
registri per thread, nessuno spill e una barriera per il kernel shared; il naïve
usa 25 registri e nessuna barriera.

### Esperimento CUDA

La GPU e il numero di SM restano fissi mentre cresce la dimensione dell'input.
Questo esperimento è indicato nel CSV come `problem_size`: non è strong scaling,
perché non varia il numero di risorse, e non è weak scaling formale, perché non
esiste un parametro di parallelismo controllato rispetto al quale mantenere
costante il lavoro per risorsa.

| Parametro | Valori |
|---|---|
| Larghezza immagine | 4096 pixel, 8 tile |
| Altezza immagine | 512, 1024, 2048, 4096, 8192, 16384, 32768 pixel |
| Batch | 8 immagini |
| Elemento strutturante | 3×3, 5×5, 9×9; forma piena e invariata |
| Operazioni | erosione, opening |
| Implementazioni | naïve/global, tiled/shared |
| Thread per blocco | 256 |
| Righe per blocco shared | 8 |
| Warm-up / misure | 2 / 10 per configurazione |

La dimensione dell'SE modifica il lavoro per pixel mantenendone fissa la forma:
non è quindi un confronto fra differenti tipi di elemento strutturante, ma fra
tre dimensioni della stessa maschera quadrata piena. Il codice dei kernel resta
comunque generico e verifica `c_se.values` per ogni cella.

### Cosa viene cronometrato in CUDA

La sola metrica è il tempo della regione dei kernel, espresso in secondi per
immagine. Un timer host basato su `clock_gettime` viene avviato immediatamente
prima dei lanci e fermato dopo `cudaDeviceSynchronize()`.

Sono esclusi:

- allocazioni e deallocazioni device;
- trasferimenti host-to-device e device-to-host;
- scansione del dataset e costruzione dei mosaici.

Per l'erosione la regione contiene un kernel; per l'opening contiene erosione e
dilatazione consecutive, mantenendo il risultato intermedio sulla GPU. Il tempo
totale della regione viene diviso per le 8 immagini del batch.

Il riferimento CPU è utilizzato esclusivamente
per verificare gli output e non viene cronometrato né scritto nel CSV.

## Validazione della correttezza

La versione OpenMP è verificata **pixel per pixel** contro il baseline
sequenziale **scalare** su un caso rappresentativo: batch di 2 immagini
512×512, elemento strutturante 3×3 e 2 thread OpenMP. Il confronto copre erosione
e opening. La validazione viene eseguita prima degli esperimenti di scaling: se
trova almeno una differenza, il programma termina senza produrre dati da analizzare.

La versione CUDA viene confrontata con un riferimento CPU separato, non incluso
nei benchmark. La validazione copre erosione e opening per SE 3×3, 5×5 e 9×9,
più dimensioni del blocco e tile shared alti 1, 4, 8 e 16 righe. Anche in questo
caso ogni differenza interrompe il programma prima della scrittura dei tempi.

## Ambiente di test

### OpenMP

| | |
|---|---|
| CPU | Apple M2 Pro — 10 core (6 performance + 4 efficiency) |
| RAM | 16 GB unified memory |
| OS | macOS 26.6.2 |
| Compilatore | Apple clang 21.0.0, `-O2 -Xclang -fopenmp` |
| Runtime OpenMP | LLVM libomp 22.1.8 (Homebrew) |

### CUDA

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| Streaming Multiprocessor | 82 SM |
| Compute capability | 8.6 |
| Architettura di compilazione | `sm_86` |
| Compilatore | `nvcc -O2 -arch=sm_86` |
| CPU host | Dipende dal server CUDA esterno; non usata come riferimento prestazionale |
| OS | UBUNTU 22.04 |

## Dataset

[Brain Cancer MRI dataset](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset)
— 6056 immagini JPEG 512×512 in scala di grigi, tre classi (`brain_glioma`,
`brain_menin`, `brain_tumor`). Nessun preprocessing è applicato oltre alla conversione a singolo canale
fatta da `stb_image` in fase di caricamento, che resta fuori dalle misure.

## Build ed esecuzione del benchmark

```sh
cd OpenMP
make            # compila in build/, produce l'eseguibile ./main
./main          # baseline sequenziale + sweep thread x dimensione -> experiment_run/results.csv
```

Per CUDA, sul server dotato di RTX 3090:

```sh
cd CUDA
make clean       # necessario dopo modifiche ai flag di compilazione
make             # compila per sm_86 e mostra l'uso delle risorse ptxas
./main           # validazione + benchmark -> experiment_run/kernel_results.csv
```

Il CSV è append-only. Prima di eseguire una versione con uno schema del log
diverso occorre archiviare o rimuovere il file precedente, evitando di mescolare
righe con intestazioni incompatibili.
