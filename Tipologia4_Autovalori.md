# 📋 Tipologia 4: Calcolo degli Autovalori
*(Metodo delle Potenze, Metodo delle Potenze Inverse, Rinvio Teorico e Velocità di Convergenza)*

---

## 🎯 Obiettivo dell'Esercizio
In questa tipologia d'esame, la prof ti chiede di approssimare numericamente gli autovalori estremi di una matrice $A$ (il più grande in modulo o quello più vicino a un determinato valore) senza calcolare l'intero spettro. Lo farai implementando il **Metodo delle Potenze** (per l'autovalore massimo) e il **Metodo delle Potenze Inverse con Shift** (per l'autovalore più vicino a un valore target o il minimo).

---

## 🔍 1. Criteri di Utilizzo e Velocità di Convergenza (Da Imparare a Memoria)

### 📂 METODO 1: Metodo delle Potenze (Power Method)
*   **Quando si usa:** Per trovare l'autovalore **massimo in modulo** ($\lambda_1$) e il corrispondente autovettore.
*   **Condizione di convergenza:** La matrice $A$ deve avere un unico autovalore dominante in modulo:
    $$|\lambda_1| > |\lambda_2| \ge \dots \ge |\lambda_n|$$
*   **Velocità di Convergenza:** Il metodo converge in modo **lineare**. La velocità dipende dal rapporto di dominanza:
    $$\text{Velocità} \propto \left(\frac{|\lambda_2|}{|\lambda_1|}\right)^k$$
    Più $\lambda_2$ è vicino a $\lambda_1$, più il rapporto tende a $1$ e più il metodo sarà lento.

### 📂 METODO 2: Metodo delle Potenze Inverse (Inverse Power Method)
*   **Quando si usa:** Per trovare l'autovalore **più vicino a un valore specifico $\sigma$ (chiamato Shift)**. Se impostiamo lo shift $\sigma = 0$, il metodo trova l'autovalore **minimo in modulo** ($\lambda_n$).
*   **Come funziona teoricamente:** Si applica il metodo delle potenze standard alla matrice inversa modificata $(A - \sigma I)^{-1}$.
*   **Velocità di Convergenza:** Anch'esso lineare, ma dipende dal rapporto:
    $$\text{Velocità} \propto \left(\frac{|\lambda_{vicino1} - \sigma|}{|\lambda_{vicino2} - \sigma|}\right)^k$$
    Se lo shift $\sigma$ è molto vicino a un autovalore, la convergenza diventa estremamente rapida.

---

## 💻 2. Il Codice Python Completo d'Esame

Ecco lo script pronto che implementa entrambe le funzioni con i relativi criteri di stop e controlli.

```python
import numpy as np
import scipy.linalg as spl
from scipy.io import loadmat

# ==========================================
# 1. CARICAMENTO DATI
# ==========================================
dati = loadmat('testIV.mat')  # Nome del file dell'esame
A = dati["A"].astype(float)
n = A.shape[0]

# Vettore iniziale x0 standard casuale o a uni
x0 = np.ones((n, 1))

# ==========================================
# 2. IMPLEMENTAZIONE METODO DELLE POTENZE
# ==========================================
def potenze(A, x0, tol=1e-6, max_it=500):
    x = x0 / np.linalg.norm(x0)  # Normalizzazione iniziale
    lambda_old = 0
    it = 0
    errore = tol + 1
    
    while errore > tol and it < max_it:
        y = np.dot(A, x)
        lambda_new = np.dot(x.T, y)[0, 0]  # Quoziente di Rayleigh
        
        x = y / np.linalg.norm(y)  # Normalizzazione per evitare overflow
        errore = np.abs(lambda_new - lambda_old)
        lambda_old = lambda_new
        it += 1
        
    return lambda_new, x, it

# ==========================================
# 3. IMPLEMENTAZIONE POTENZE INVERSE (CON SHIFT)
# ==========================================
def potenze_inverse(A, x0, sigma=0, tol=1e-6, max_it=500):
    x = x0 / np.linalg.norm(x0)
    lambda_old = 0
    it = 0
    errore = tol + 1
    
    # Pre-fattorizziamo la matrice con lo shift (A - sigma*I) per non rifarlo nel ciclo!
    # Questo ottimizza i tempi computazionali in modo drastico.
    M = A - sigma * np.eye(A.shape[0])
    P, L, U = spl.lu(M)
    
    while errore > tol and it < max_it:
        # Invece di fare l'inversa, risolviamo il sistema lineare M * y = x usando LU
        y_temp = spl.solve_triangular(L, np.dot(P.T, x), lower=True)
        y = spl.solve_triangular(U, y_temp, lower=False)
        
        # Quoziente di Rayleigh su M^-1: lambda_inv = (x.T @ y) / (x.T @ x), ma x ha norma 1
        lambda_nuovo_inv = np.dot(x.T, y)[0, 0]
        
        # Formula di conversione corretta per tornare allo spettro di A
        lambda_new = sigma + (1.0 / lambda_nuovo_inv)
        
        x = y / np.linalg.norm(y)
        errore = np.abs(lambda_new - lambda_old)
        lambda_old = lambda_new
        it += 1
        
    return lambda_new, x, it

# ==========================================
# 4. ESECUZIONE, OUTPUT E VERIFICA
# ==========================================
lam_max, v_max, it_max = potenze(A, x0)
lam_min, v_min, it_min = potenze_inverse(A, x0, sigma=0)

# Calcolo dello spettro esatto con Scipy per il confronto d'esame
autovalori_esatti = spl.eigvals(A)
lam_max_esatto = np.max(np.abs(autovalori_esatti))
lam_min_esatto = np.min(np.abs(autovalori_esatti))

print(f"--- RISULTATI METODO DELLE POTENZE ---")
print(f"Autovalore Massimo calcolato: {lam_max:.6f} ({it_max} iterazioni)")
print(f"Autovalore Massimo esatto:    {lam_max_esatto:.6f}")
print(f"Errore Assoluto Max:           {np.abs(lam_max - lam_max_esatto):.2e}\n")

print(f"--- RISULTATI POTENZE INVERSE ---")
print(f"Autovalore Minimo calcolato:  {lam_min:.6f} ({it_min} iterazioni)")
print(f"Autovalore Minimo esatto:     {lam_min_esatto:.6f}")
print(f"Errore Assoluto Min:           {np.abs(lam_min - lam_min_esatto):.2e}")
```

---

## 💬 3. La Risposta Teorica Standard da scrivere nel Notebook

Quando la prof chiede di *"commentare i risultati, la velocità e l'importanza della scelta dello shift"*, rispondi così:

> *"Il metodo delle potenze ha consentito di isolare stabilmente l'autovalore dominante in quanto la matrice soddisfa la condizione di separazione spettrale $|\lambda_1| > |\lambda_2|$. La velocità di convergenza è asintoticamente legata al fattore di riduzione $|\lambda_2/\lambda_1|$. Per quanto riguarda il metodo delle potenze inverse, l'introduzione dello shift $\sigma$ ha permesso di traslare lo spettro della matrice; questo accelera fortemente la convergenza poiché rende l'autovalore d'interesse estremamente dominante nella matrice modificata $(A - \sigma I)^{-1}$. Dal punto di vista computazionale, per evitare il calcolo esplicito della matrice inversa ad ogni iterazione (operazione inefficiente e instabile), si è preferito pre-calcolare la fattorizzazione LU della matrice shiftata una sola volta fuori dal ciclo, riducendo il costo di ogni iterazione a due semplici sostituzioni triangolari di costo $O(n^2)$."*

---

## 🏁 Checklist d'Esame: Come completare l'esercizio
1. [ ] Normalizza sempre il vettore di partenza `x0` all'inizio e dentro il ciclo per impedire l'esplosione dei valori numerici.
2. [ ] Nel metodo delle potenze standard, usa il Quoziente di Rayleigh (`x.T @ A @ x`) per estrarre l'approssimazione dell'autovalore.
3. [ ] Nel metodo delle potenze inverse, applica lo shift alla diagonale tramite `A - sigma * np.eye(n)`.
4. [ ] **MANDATORIO:** Non usare `np.linalg.inv()`. Applica la scomposizione `spl.lu()` prima del ciclo `while` per risolvere i sistemi interni.
5. [ ] Ricorda la formula finale di ripristino dell'autovalore reale: $\lambda = \sigma + \frac{1}{\lambda_{inv}}$.
