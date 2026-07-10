# 📋 Tipologia 1: Condizionamento e Stabilità Numerica
*(Analisi degli errori, cancellazione di cifre significative e spacing)*

---

## 🎯 Obiettivo dell'Esercizio
In questa tipologia d'esame, la prof testa la tua capacità di riconoscere il fenomeno della **cancellazione numerica** (o perdita di cifre significative) e di saper **stabilizzare algebricamente** una formula matematica instabile, verificando i risultati sia graficamente sia numericamente attraverso uno Jupyter Notebook.

---

## 🧠 1. La Teoria da Scrivere nel Commento (Punti Sicuri)

Quando il testo chiede di *"spiegare perché il calcolo diretto soffre di instabilità numerica, dove avviene la cancellazione e richiamare il concetto di spacing"*, devi rispondere usando queste precise parole chiave:

1. **Cancellazione Numerica:** Quando si esegue la sottrazione tra due numeri molto vicini tra loro (es. $1 - \cos(x)$ quando $x \to 0$, oppure $k - \sqrt{k^2 - 1}$ quando $k \to \infty$), le cifre decimali identiche si annullano a vicenda. Il risultato finale perde precisione e conterrà solo cifre decimali di "rumore", generando un errore relativo altissimo (prossimo al 100%).
2. **Spacing (Spaziatura):** Lo *spacing* rappresenta la minima distanza rappresentabile tra un numero floating-point e il suo successivo nella memoria del computer (legato alla precisione di macchina $\epsilon_m \approx 2.22 \times 10^{-16}$ per la doppia precisione). Quando una quantità diventa inferiore alla radice quadrata della precisione di macchina ($\approx 10^{-8}$), il computer non riesce più a distinguere la differenza matematica tra il valore calcolato (es. $\cos(x)$) e il valore limite (es. $1$). Il valore viene memorizzato come identico, e la sottrazione restituisce un errore distruttivo.

---

## 🛠️ 2. I Due Pattern Algebrici di Stabilizzazione (Da Imparare a Memoria)

### 🔹 Caso A: Funzioni Trigonometriche ($1 - \cos(x)$)
Si moltiplica e si divide per il coniugato $(1 + \cos(x))$ per eliminare la sottrazione:

$$y_k = 1 - \cos(x_k) = \frac{(1 - \cos(x_k))(1 + \cos(x_k))}{1 + \cos(x_k)}$$

Sviluppando il prodotto notevole al numeratore:

$$y_k = \frac{1 - \cos^2(x_k)}{1 + \cos(x_k)} = \frac{\sin^2(x_k)}{1 + \cos(x_k)}$$

*Perché è stabile?* Perché al numeratore il $\sin(x_k)$ va a $0$ senza sottrazioni distruttive, e al denominatore abbiamo una somma stable $1 + \cos(x_k) \approx 2$.

### 🔹 Caso B: Equazioni di Secondo Grado ($x^2 - 2kx + 1 = 0$)
L'instabilità per cancellazione numerica dipende dal segno di $k$: se $k \to +\infty$ l'instabilità è sulla sottrazione in $x_1$, mentre se $k \to -\infty$ l'instabilità si sposta sulla somma in $x_2$. 

Per stabilizzare l'algoritmo in modo universale, si calcola prima la radice sicuramente stabile concordando il segno di $k$ con quello davanti alla radice quadrata (somma di valori concordi):
$$x_{\text{stabile}} = k + \operatorname{sgn}(k)\sqrt{k^2 - 1}$$

La seconda radice si ricava poi senza alcuna cancellazione distruttiva sfruttando la relazione del prodotto delle radici ($x_1 \cdot x_2 = \frac{c}{a} = 1$):
$$x_{\text{alternativa}} = \frac{1}{x_{\text{stabile}}}$$

$$x_1 = \frac{(k - \sqrt{k^2 - 1})(k + \sqrt{k^2 - 1})}{k + \sqrt{k^2 - 1}} = \frac{k^2 - (k^2 - 1)}{k + \sqrt{k^2 - 1}} = \frac{1}{k + \sqrt{k^2 - 1}}$$

Poiché il denominatore è esattamente la radice stabile $x_2$, la formula finale da usare a codice è:

$$x_1 = \frac{1}{x_2}$$

---

## 💻 3. Il Codice Python Standard per il Notebook

Ecco la struttura del codice da scrivere nella cella del Jupyter Notebook per il Caso A (Trigonometrico). Il meccanismo per il Caso B è identico.

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Generazione dei dati (passo k da 1 a 16)
k = np.arange(1, 17)
x = 10.0**(-k)

# 2. Implementazione della formula DIRETTA (Instabile)
y_diretta = 1.0 - np.cos(x)

# 3. Implementazione della formula ALTERNATIVA (Stabile / Riferimento)
y_stabile = np.sin(x)**2 / (1.0 + np.cos(x))

# 4. Calcolo dell'Errore Relativo della formula instabile rispetto alla stabile
errore_relativo = np.abs(y_diretta - y_stabile) / np.abs(y_stabile)

# 5. Rappresentazione grafica in scala logaritmica (MANDATORIA ALL'ESAME)
plt.figure(figsize=(10, 5))
plt.loglog(x, errore_relativo, 'ob-', linewidth=2, label='Errore Relativo (Formula Diretta)')
plt.xlabel('Valori di x (Scala Logaritmica)')
plt.ylabel('Errore Relativo')
plt.title('Analisi dell\'Instabilità Numerica e Cancellazione')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
```

---

## 💬 3. La Risposta Teorica Standard da scrivere nel Notebook

Quando la prof ti chiede di *"analizzare l'andamento dell'errore relativo a seguito della perturbazione dei dati d'ingresso, distinguendo il contributo del condizionamento del problema dalla stabilità dell'algoritmo"*, rispondi così:

> *"L'analisi numerica effettuata tramite l'introduzione di perturbazioni controllate sui dati d'ingresso permette di separare nettamente le proprietà intrinseche del problema da quelle dell'algoritmo implementato. L'errore relativo finale riscontrato sulla soluzione è governato dalla precisione finita di macchina $\epsilon_m$ (legata alla rappresentazione floating point a 64 bit) e dall'indice di condizionamento del problema. Se a fronte di una piccola perturbazione sui dati l'errore sulla soluzione cresce in modo controllato e lineare, il problema si definisce ben condizionato e l'algoritmo dimostra stabilità numerica, confinando l'errore entro le limitazioni imposte dall'arrotondamento di macchina. Viceversa, un'esplosione dell'errore relativo sperimentale evidenzia due possibili criticità: un mal condizionamento intrinseco del problema (in cui la sensibilità ai dati è geometricamente elevata e indipendente dal codice) oppure un'instabilità dell'algoritmo stesso, tipicamente causata da fenomeni di cancellazione numerica nelle operazioni intermedie (come la sottrazione di quantità quasi identiche), che distruggono le cifre significative ereditando e amplificando gli errori di troncamento."*
---

## 🏁 Checklist d'Esame: Come completare l'esercizio
1. [ ] Esegui la prima cella per definire il vettore dei nodi $x_k$ [cite: 1].
2. [ ] Scrivi la formula diretta instabile e quella geometricamente modificata stabile [cite: 1].
3. [ ] Calcola l'errore relativo impostando la formula stabile come denominatore [cite: 1].
4. [ ] Genera il grafico usando rigorosamente `plt.loglog` (scala logaritmica su entrambi gli assi) [cite: 1].
5. [ ] Scrivi il commento teorico nella cella di testo sottostante menzionando **cancellazione** e **spacing** [cite: 1].
