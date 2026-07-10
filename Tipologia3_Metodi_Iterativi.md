# 📋 Tipologia 3: Metodi Iterativi per Sistemi Lineari Sparse
*(Jacobi, Gauss-Seidel, Criteri di Convergenza e Raggio Spettrale)*

---

## 🎯 Obiettivo dell'Esercizio
In questa tipologia d'esame, la prof ti fornisce una matrice $A$ tipicamente di **grandi dimensioni e sparsa** (molti zeri) [cite: 1]. L'obiettivo è verificare *a priori* se i metodi iterativi di **Jacobi** e **Gauss-Seidel** convergeranno alla soluzione esatta, implementare gli algoritmi con un criterio di arresto basato su una tolleranza (`tol`) e sul numero massimo di iterazioni (`max_it`), e confrontare la loro velocità di convergenza [cite: 1].

---

## 🔍 1. Criteri di Convergenza (Quando e perché un metodo converge)

Prima di scrivere il ciclo dell'algoritmo, devi verificare matematicamente se i metodi convergono usando due criteri principali [cite: 1]:

### 🔹 Condizione Sufficiente (Dominanza Diagonale)
Se la matrice $A$ è a **dominanza diagonale stretta per righe**, allora **sia Jacobi sia Gauss-Seidel convergono sicuramente**.
$$|a_{ii}| > \sum_{j \neq i} |a_{ij}| \quad \forall i$$

### 🔹 Condizione Necessaria e Sufficiente (Il Raggio Spettrale $
ho$)
Un metodo iterativo converge se e solo se il **raggio spettrale** della sua matrice di iterazione $B$ è strettamente minore di $1$ ($
ho(B) < 1$) [cite: 1]. Il raggio spettrale è il valore assoluto dell'autovalore più grande della matrice di iterazione [cite: 1].
*   **Velocità:** Più $
ho(B)$ è vicino a $0$, più il metodo è veloce a convergere [cite: 1].
*   **Matrice di Jacobi ($B_J$):** Scomponendo $A = D - L - U$ (dove $D$ è la diagonale, $L$ la parte inferiore e $U$ la superiore), si ha $B_J = D^{-1}(L + U)$ [cite: 1].
*   **Matrice di Gauss-Seidel ($B_{GS}$):** Si ha $B_{GS} = (D - L)^{-1}U$ [cite: 1].

---

## 💻 2. Il Codice Python Completo d'Esame

Ecco lo script standard da usare nel notebook per calcolare le matrici di iterazione, verificare la convergenza tramite il raggio spettrale ed eseguire le iterazioni [cite: 1].

```python
import numpy as np
import scipy.linalg as spl
from scipy.io import loadmat

# ==========================================
# 1. CARICAMENTO DATI
# ==========================================
dati = loadmat('testIII.mat')  # Nome del file dell'esame
A = dati["A"].astype(float)
b = dati["b"].astype(float)

n = A.shape[0]

# ==========================================
# 2. COSTRUZIONE DELLE MATRICI DI ITERAZIONE
# ==========================================
D = np.diag(np.diag(A))
L = -np.tril(A, -1)
U = -np.triu(A, 1)

# Matrice di Jacobi: B_J = D^-1 * (L + U)
inv_D = np.diag(1.0 / np.diag(A))
B_J = np.dot(inv_D, (L + U))
raggio_J = np.max(np.abs(np.linalg.eigvals(B_J)))

# Matrice di Gauss-Seidel: risolviamo (D - L) * B_GS = U sfruttando la struttura triangolare
B_GS = spl.solve_triangular(D - L, U, lower=True)
raggio_GS = np.max(np.abs(np.linalg.eigvals(B_GS)))

print(f"--- ANALISI DI CONVERGENZA ---")
print(f"Raggio Spettrale Jacobi: {raggio_J:.4f} -> CONVERGE? {raggio_J < 1}")
print(f"Raggio Spettrale Gauss-Seidel: {raggio_GS:.4f} -> CONVERGE? {raggio_GS < 1}\n")

# ==========================================
# 3. FUNZIONE DI ITERAZIONE STANDARD
# ==========================================
def risolvi_iterativo(B, c, b, tol=1e-6, max_it=500):
    x = np.zeros((b.shape[0], 1)) # Vettore iniziale x0 a zeri
    it = 0
    errore = tol + 1
    
    while errore > tol and it < max_it:
        x_new = np.dot(B, x) + c
        # Calcolo del residuo relativo come criterio di arresto
        errore = np.linalg.norm(np.dot(A, x_new) - b) / np.linalg.norm(b)
        x = x_new
        it += 1
        
    return x, it

# Vettori costanti per i metodi
c_J = np.dot(inv_D, b)
# Vettore costante Gauss-Seidel: risolviamo (D - L) * c_GS = b
c_GS = spl.solve_triangular(D - L, b, lower=True)

# Esecuzione dei metodi
if raggio_J < 1:
    x_J, it_J = risolvi_iterativo(B_J, c_J, b)
    print(f"Jacobi ha convertito in {it_J} iterazioni.")
else:
    print("Jacobi non converge.")

if raggio_GS < 1:
    x_GS, it_GS = risolvi_iterativo(B_GS, c_GS, b)
    print(f"Gauss-Seidel ha convertito in {it_GS} iterazioni.")
else:
    print("Gauss-Seidel non converge.")
```

---

## 💬 3. La Risposta Teorica Standard da scrivere nel Notebook

Quando la prof chiede di *"commentare la convergenza e la velocità dei due metodi alla luce della teoria"*, rispondi così [cite: 1]:

> *"Dall'analisi numerica effettuata sulle matrici di iterazione, si evince che il metodo converge poiché il rispettivo raggio spettrale $
ho(B)$ risulta strettamente minore di $1$ [cite: 1]. Confrontando i due valori, si nota che $
ho(B_{GS}) < 
ho(B_J) < 1$, il che giustifica teoricamente perché il metodo di Gauss-Seidel richieda un numero significativamente inferiore di iterazioni rispetto a Jacobi per raggiungere la medesima tolleranza impostata [cite: 1]. Per le matrici d'esame (spesso P-matrici o a dominanza diagonale), vale infatti il teorema di Stein-Rosenberg, il quale stabilisce che i due metodi sono o entrambi convergenti o entrambi divergenti, e in caso di convergenza Gauss-Seidel risulta sistematicamente più veloce [cite: 1]."*

---

## 🏁 Checklist d'Esame: Come completare l'esercizio
1. [ ] Estrai le componenti diagonali (`D`), triangolari inferiori (`L`) e superiori (`U`) di $A$ [cite: 1].
2. [ ] Calcola le matrici d'iterazione $B_J$ e $B_{GS}$ [cite: 1].
3. [ ] Trova il massimo valore assoluto degli autovalori con `np.max(np.abs(np.linalg.eigvals(B)))` per calcolare il raggio spettrale [cite: 1].
4. [ ] Se $
ho < 1$, avvia il ciclo `while` per aggiornare il vettore della soluzione [cite: 1].
5. [ ] Usa il residuo relativo corretto ($\frac{\|Ax - b\|}{\|b\|}$) come criterio di controllo dello stop [cite: 1].
6. [ ] Scrivi il commento finale confrontando il numero di iterazioni reali con i raggi spettrali teorici calcolati [cite: 1].
