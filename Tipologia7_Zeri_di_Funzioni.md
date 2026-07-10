# 📋 Tipologia 7: Zeri di Funzioni ed Equazioni Non Lineari
*(Metodo di Bisezione, Metodo di Newton-Raphson, Criteri di Arresto e Ordine di Convergenza)*

---

## 🎯 Obiettivo dell'Esercizio
In questa tipologia d'esame, l'obiettivo è calcolare l'approssimazione numerica della radice $\\alpha$ di una funzione non lineare, ovvero risolvere l'equazione:
$$f(x) = 0$$
Viene richiesto di confrontare un metodo globale ma lento (**Bisezione**) con un metodo locale ma quadratico (**Newton-Raphson**), valutando il rispettivo numero di iterazioni necessarie a raggiungere una tolleranza imposta (`tol`) e verificando sperimentalmente l'ordine di convergenza.

---

## 🔍 1. Criteri Scientifici e Teoria d'Esame (Da Imparare a Memoria)

### 📂 METODO 1: Metodo di Bisezione
*   **Condizione di Esistenza (Teorema degli Zeri):** Richiede una funzione continua $f(x)$ in un intervallo $[a, b]$ tale che ai bordi cambi di segno: $f(a) \cdot f(b) < 0$.
*   **Proprietà:** Converge **sempre** (convergenza globale), ma in modo molto lento.
*   **Velocità e Ordine di Convergenza:** Ha un ordine di convergenza pari a **1** (lineare). Ad ogni iterazione l'ampiezza dell'intervallo si dimezza; il numero di iterazioni $k$ necessarie per garantire un errore inferiore a `tol` si può calcolare *a priori* con la formula:
    $$k > \log_2(b - a) - \log_2(\text{tol})$$

### 📂 METODO 2: Metodo di Newton-Raphson
*   **Condizione di Convergenza:** È un metodo locale. Converge se il punto iniziale $x_0$ è sufficientemente vicino alla radice $\alpha$, se $f(x)$ è derivabile e se $f'(x) \neq 0$ nell'intorno della radice.
*   **Velocità e Ordine di Convergenza:** Se la radice è semplice ($f'(\alpha) \neq 0$), l'ordine di convergenza è **2** (quadratica). Questo significa che il numero di cifre decimali esatte raddoppia ad ogni iterazione, rendendolo infinitamente più veloce di Bisezione in prossimità della soluzione.

---

## 💻 2. Il Codice Python Completo d'Esame

```python
import numpy as np

# ==========================================
# 1. DEFINIZIONE DELLE FUNZIONI E DERIVATE
# ==========================================
# Esempio d'esame classico: f(x) = x^3 - x - 2  (La radice è compresa tra 1 e 2)
f = lambda x: x**3 - x - 2.0
df = lambda x: 3.0 * x**2 - 1.0  # Derivata prima necessaria per Newton

a, b = 1.0, 2.0
tol = 1e-10
max_it = 100

# ==========================================
# 2. IMPLEMENTAZIONE METODO DI BISEZIONE
# ==========================================
def bisezione(f, a, b, tol, max_it):
    if f(a) * f(b) >= 0:
        raise ValueError("Il Teorema degli zeri non è applicabile nell'intervallo fornito.")
        
    it = 0
    # Criterio di arresto basato sull'ampiezza dell'intervallo
    while (b - a) > tol and it < max_it:
        c = a + (b - a) / 2.0  # Evita overflow rispetto a (a+b)/2
        if f(c) == 0:
            return c, it
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        it += 1
    return a + (b - a) / 2.0, it

# ==========================================
# 3. IMPLEMENTAZIONE METODO DI NEWTON
# ==========================================
def newton(f, df, x0, tol, max_it):
    x = x0
    it = 0
    incremento = tol + 1
    
    while incremento > tol and it < max_it:
        fx = f(x)
        dfx = df(x)
        
        if np.abs(dfx) < 1e-15:
            raise ZeroDivisionError("Derivata prima nulla o troppo vicina allo zero.")
            
        x_new = x - fx / dfx
        # Criterio di arresto basato sullo scarto tra iterazioni successive
        incremento = np.abs(x_new - x)
        x = x_new
        it += 1
        
    return x, it

# ==========================================
# 4. ESECUZIONE E CONFRONTO
# ==========================================
x_bis, it_bis = bisezione(f, a, b, tol, max_it)
x_newt, it_newt = newton(f, df, x0=1.5, tol=tol, max_it=max_it)

print(f"--- CONFRONTO METODI NON LINEARI ---")
print(f"Bisezione: Radice = {x_bis:.10f} | Iterazioni = {it_bis}")
print(f"Newton:    Radice = {x_newt:.10f} | Iterazioni = {it_newt}")
```

---

## 💬 3. La Risposta Teorica Standard da scrivere nel Notebook

Quando viene richiesto di *"commentare i risultati ottenuti evidenziando le differenze di convergenza e l'efficienza computazionale dei due metodi applicati"*, la struttura ideale della risposta è:

> *"Il confronto numerico evidenzia in modo nitido la discrepanza nell'efficienza asintotica dei due algoritmi. Il metodo di Bisezione mostra una convergenza globale robusta ma estremamente lenta, richiedendo un numero elevato di iterazioni in quanto il suo errore decade in modo strettamente lineare (ordine di convergenza $p = 1$), dimezzando l'intervallo di incertezza ad ogni passo secondo una progressione geometrica indipendente dalla struttura della funzione. Di contro, il metodo di Newton-Raphson raggiunge la tolleranza impostata in una frazione minima di passaggi grazie al suo ordine di convergenza quadratico ($p = 2$). Poiché il punto iniziale $x_0$ è stato scelto opportunamente all'interno del bacino di attrazione della radice semplice (dove la derivata prima si mantiene distante dallo zero), lo scarto tra le iterazioni decade in modo quadratico, raddoppiando il numero di cifre decimali esatte ad ogni ciclo e garantendo un'efficienza computazionale nettamente superiore a parità di accuratezza richiesta."*

---

## 🏁 Checklist d'Esame: Come completare l'esercizio
1. [ ] Definisci accuratamente la funzione $f(x)$ e calcola analiticamente la sua derivata $f'(x)$.
2. [ ] Per la bisezione, verifica che $f(a) \cdot f(b) < 0$ prima di far partire qualsiasi ciclo.
3. [ ] Come punto di partenza $x_0$ per Newton, se il testo non lo specifica, usa il punto medio dell'intervallo di bisezione o l'estremo in cui $f(x) \cdot f''(x) > 0$ (Condizioni di Fourier).
4. [ ] Controlla che le tolleranze richieste siano applicate correttamente come ampiezza dell'intervallo per Bisezione e come scarto assoluto $|x_{\text{new}} - x|$ per Newton.