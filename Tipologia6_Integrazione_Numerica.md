# 📋 Tipologia 6: Integrazione Numerica e Formule di Quadratura
*(Formule Composite dei Trapezi e di Simpson, Stima dell'Errore e Grado di Precisione)*

---

## 🎯 Obiettivo dell'Esercizio
In questa tipologia d'esame, l'obiettivo è calcolare l'approssimazione numerica dell'integrale definito di una funzione $f(x)$ in un intervallo $[a, b]$:
$$I(f) = \int_{a}^{b} f(x) \, dx$$
Poiché spesso non è possibile o efficiente calcolare la primitiva analitica, si ricorre alle **formule di quadratura composite** (Trapezi e Simpson) dividendo l'intervallo in $N$ sottointervalli. Ti verrà richiesto di implementare i metodi, confrontare i risultati analitici/numerici e analizzare come l'errore decresca al variare del numero di nodi $N$.

---

## 🔍 1. Criteri Scientifici e Teoria d'Esame (Da Imparare a Memoria)

### 📂 METODO 1: Formula del Trapezio Composita
*   **Come funziona:** Approssima la funzione in ogni sottointervallo con una retta (polinomio di grado 1).
*   **Grado di Precisione:** Il grado di precisione è **1** (integra esattamente i polinomi di grado minore o uguale a 1).
*   **Stima dell'Errore:** L'errore di scomposizione decresce in modo quadratico rispetto all'ampiezza degli intervalli $h = \frac{b-a}{N}$:
    $$E_{Trap}(f) = -\frac{b-a}{12} h^2 f''(\xi)$$
    Raddoppiando il numero dei sottointervalli $N$, l'errore si riduce di un fattore pari a **4**.

### 📂 METODO 2: Formula di Simpson Composita (Parabole)
*   **Come funziona:** Approssima la funzione in ogni coppia di sottointervalli con una parabola (polinomio di grado 2).
*   **Vincolo Strutturale:** Il numero di sottointervalli $N$ deve essere obbligatoriamente **PARI** (ovvero il numero di nodi $N+1$ deve essere dispari).
*   **Grado di Precisione:** Sorprendentemente è **3** (integra esattamente anche i polinomi di grado 3, grazie alla simmetria dell'errore).
*   **Stima dell'Errore:** L'errore decresce con la quarta potenza di $h$:
    $$E_{Simp}(f) = -\frac{b-a}{180} h^4 f^{(4)}(\xi)$$
    Raddoppiando il numero dei sottointervalli $N$, l'errore si riduce di un fattore pari a **16** (convergenza estremamente più rapida).

---

## 💻 2. Il Codice Python Completo d'Esame

```python
import numpy as np

# ==========================================
# 1. DEFINIZIONE DELLA FUNZIONE TEST
# ==========================================
# Esempio classico d'esame: f(x) = sin(x) integrata tra 0 e pi (Valore esatto = 2.0)
f = lambda x: np.sin(x)
a = 0.0
b = np.pi
valore_esatto = 2.0

# ==========================================
# 2. IMPLEMENTAZIONE TRAPEZI COMPOSITI
# ==========================================
def trapezi_compositi(f, a, b, N):
    h = (b - a) / N
    # Generiamo i punti interni escludendo gli estremi
    x = np.linspace(a, b, N + 1)
    
    # Formula: (h/2) * [ f(a) + 2*sum(f(xi)) + f(b) ]
    integrale = (h / 2.0) * (f(a) + 2.0 * np.sum(f(x[1:-1])) + f(b))
    return integrale

# ==========================================
# 3. IMPLEMENTAZIONE SIMPSON COMPOSITO
# ==========================================
def simpson_composito(f, a, b, N):
    if N % 2 != 0:
        raise ValueError("Il numero di sottointervalli N deve essere PARI per il metodo di Simpson!")
        
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    
    # Nodi ad indice dispari (peso 4) e nodi ad indice pari interni (peso 2)
    somma_dispari = np.sum(f(x[1:-1:2]))
    somma_pari = np.sum(f(x[2:-1:2]))
    
    # Formula: (h/3) * [ f(a) + 4*sum(f(dispari)) + 2*sum(f(pari)) + f(b) ]
    integrale = (h / 3.0) * (f(a) + 4.0 * somma_dispari + 2.0 * somma_pari + f(b))
    return integrale

# ==========================================
# 4. RICERCA DINAMICA DI N PER TOLLERANZA IMPOSTATA
# ==========================================
tol_desiderata = 1e-6

# --- Ricerca per Trapezi ---
N_trap = 2
errore_trap = 1.0
while errore_trap > tol_desiderata and N_trap < 10000:
    int_trap = trapezi_compositi(f, a, b, N_trap)
    errore_trap = np.abs(int_trap - valore_esatto)
    if errore_trap > tol_desiderata:
        N_trap *= 2  # Raddoppiamo i sottointervalli ad ogni fallimento

# --- Ricerca per Simpson ---
N_simp = 2  # Deve partire e rimanere PARI
errore_simp = 1.0
while errore_simp > tol_desiderata and N_simp < 10000:
    int_simp = simpson_composito(f, a, b, N_simp)
    errore_simp = np.abs(int_simp - valore_esatto)
    if errore_simp > tol_desiderata:
        N_simp *= 2  # Moltiplicare per 2 preserva la parità di N

print(f"--- STIMA DEI NODI NECESSARI (TOL = {tol_desiderata}) ---")
print(f"Trapezi: N minimo = {N_trap} | Errore effettivo = {errore_trap:.2e}")
print(f"Simpson: N minimo = {N_simp}  | Errore effettivo = {errore_simp:.2e}")
```

---

## 💬 3. La Risposta Teorica Standard da scrivere nel Notebook

Quando la prof ti chiede di *"valutare l'andamento degli errori sperimentali delle formule composite al raddoppiare dei nodi e giustificare quale metodo sia più efficiente"*, rispondi così:

> *"Dall'analisi numerica condotta si evince chiaramente la superiorità della formula di Simpson composita rispetto a quella dei Trapezi. Dal punto di vista teorico, la formula dei Trapezi approssima l'integrando mediante interpolazione lineare locale, esibendo un grado di precisione pari a 1 e un errore asintotico nell'ordine di $O(h^2)$. Di conseguenza, al raddoppiare del numero di sottointervalli $N$, l'errore decresce linearmente di un fattore 4. La formula di Simpson composita, sfruttando invece parabole interpolanti su coppie di intervalli adiacenti, eleva il proprio grado di precisione a 3 per via delle simmetrie del nucleo di Peano. L'errore associato decade come $O(h^4)$, il che implica che al raddoppiare di $N$ l'errore si abbatte di ben 16 volte. Ne consegue che, a parità di nodi campionati e quindi di costo computazionale (valutazioni della funzione $f(x)$), il metodo di Simpson garantisce un'accuratezza decisamente superiore ed una convergenza significativamente più rapida."*

---

## 🏁 Checklist d'Esame: Come completare l'esercizio
1. [ ] Verifica immediatamente se la prof ti fornisce una funzione analitica (`lambda x: ...`) o un insieme di punti tabulati discreti.
2. [ ] Se usi il metodo di **Simpson**, assicurati al 100% che il parametro `N` inserito nel codice sia un numero **pari**.
3. [ ] Nello slicing di Simpson con NumPy, ricorda l'indicizzazione: `1:-1:2` seleziona i nodi interni dispari, mentre `2:-1:2` seleziona i nodi interni pari.
4. [ ] Se viene richiesto di trovare il valore minimo di $N$ per garantire un errore inferiore a una tolleranza `tol`, implementa un ciclo `while` incrementando $N$ di volta in volta finché l'errore calcolato non scende sotto la soglia.

---
[⬅️ Torna all'Indice Principale](README.md)
