# 🗂️ Indice Schemi d'Esame - Metodi Numerici per l'IA

Questo documento funge da mappa e punto di accesso rapido per tutti gli schemi di laboratorio ottimizzati per l'esame. Cliccando sul nome di una tipologia verrai reindirizzato direttamente al file `.md` corrispondente contenente i criteri scientifici di scelta, la teoria da scrivere e il codice Python pre-impostato.

---

## 📊 Tabella Riassuntiva delle Tipologie

| # | Tipologia Esercizio | Breve Descrizione | Link Diretto al File |
| :-: | :--- | :--- | :-: |
| **1** | **Condizionamento e Stabilità** | Analisi degli errori, cancellazione di cifre significative e stabilizzazione delle formule (es. equazioni di secondo grado con segni concordi). | [Apri Schema](./Tipologia1_Condizionamento_Stabilita.md) |
| **2** | **Sistemi Lineari Quadrati** | Metodi diretti (Fattorizzazione LU, Cholesky, QR). Scelta ottimizzata basata su condizionamento, simmetria e definizione positiva via *try-except*. | [Apri Schema](./Tipologia2_Metodi_Diretti.md) |
| **3** | **Sistemi Lineari Sparsi** | Metodi iterativi stazionari (Jacobi e Gauss-Seidel). Ottimizzazione computazionale delle matrici di iterazione senza inversione esplicita. | [Apri Schema](./Tipologia3_Metodi_Iterativi.md) |
| **3b** | **Metodi di Discesa** | Metodi iterativi non stazionari (Gradiente e Gradiente Coniugato) per matrici sparse simmetriche e definite positive. | [Apri Schema](./Tipologia3bis_Metodi_Discesa.md) |
| **4** | **Calcolo degli Autovalori** | Metodo delle Potenze (autovalore massimo) e Potenze Inverse (autovalore minimo via shift) con stabilizzazione tramite Quoziente di Rayleigh. | [Apri Schema](./Tipologia4_Autovalori.md) |
| **5** | **Approssimazione e Interpolazione** | Interpolazione polinomiale esatta di Lagrange (nodi equispaziati vs Chebyshev) ottimizzata con la formula baricentrica stabile. | [Apri Schema](./Tipologia5_Approssimazione_e_Interpolazione.md) |
| **5b** | **Minimi Quadrati** | Risoluzione di sistemi sovradeterminati (Equazioni Normali via Cholesky/cho_solve e fattorizzazione QR economica). | [Apri Schema](./Tipologia_5bis_Minimi_Quadrati.md) |
| **6** | **Integrazione Numerica** | Formule composite (Trapezi e Simpson) accoppiate alla routine di ricerca dinamica del numero minimo di nodi $N$ per soddisfare la tolleranza. | [Apri Schema](./Tipologia6_Integrazione_Numerica.md) |
| **7** | **Equazioni Non Lineari** | Ricerca degli zeri di una funzione $f(x)=0$ confrontando il Metodo di Bisezione (convergenza globale) e il Metodo di Newton (convergenza locale quadratica). | [Apri Schema](./Tipologia7_Equazioni_Non_Lineari.md) |
| **8** | **Equazioni Differenziali (ODE)** | Risoluzione del problema di Cauchy confrontando metodi espliciti (Eulero Avanti, RK4) e impliciti (Eulero Indietro) con analisi di stabilità. | [Apri Schema](./Tipologia8_Equazioni_Differenziali.md) |

---

## 🚀 Note per l'Uso Rapido in Laboratorio
1. **Verifiche preliminari:** Apri sempre per primo il file del sistema lineare (`Tipologia 2`) o dell'approssimazione (`Tipologia 5b`) a seconda dell'input del file `.mat`.
2. **Grafici:** Tutti i file contengono già i corretti setup di `matplotlib` impostati con scale logaritmiche (`plt.loglog`) o marcatori discreti dove strettamente richiesto dai professori.
3. **Risposte Teoriche:** In fondo a ciascun file trovi un blocco di testo pre-impostato da inserire nei commenti del Notebook per motivare scientificamente le scelte algoritmiche effettuate.