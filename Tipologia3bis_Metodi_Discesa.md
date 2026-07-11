# 📋 Tipologia 3 bis: Metodi di Discesa per Sistemi Sparsi (Non Stazionari)
*(Metodo del Gradiente, Gradiente Coniugato, Ottimizzazione Quadratica ed Esame 7 Maggio 25)*

---

## 🎯 Obiettivo dell'Esercizio
Risolvere sistemi lineari di grandi dimensioni e sparsi della forma $Ax = b$ sfruttando metodi iterativi basati sulla minimizzazione della funzione energia quadratica:
$$\Phi(x) = \frac{1}{2} x^T A x - x^T b$$

Il punto di minimo di $\Phi(x)$ coincide esattamente con la soluzione del sistema lineare $Ax = b$.

---

## 🔍 1. Criteri Scientifici e Teoria d'Esame (Da Imparare a Memoria)

### 🚨 Requisito Strutturale Fondamentale
I metodi di discesa richiedono rigorosamente che la matrice $A$ sia **Simmetrica e Definita Positiva**. Se la matrice non soddisfa queste due proprietà, i metodi non convergono o non garantiscono il minimo globale.

### 📂 METODO 1: Gradiente Standard (Steepest Descent)
*   **Direzione di discesa:** Corrisponde al gradiente negativo della funzione energia, che coincide con il residuo corrente $r_k = b - Ax_k$.
*   **Comportamento:** Ad ogni passo la nuova direzione è ortogonale solo a quella precedente. Questo causa un andamento "a zigzag". Se la matrice $A$ è mal condizionata ($K(A)$ alto), la convergenza rallenta drasticamente, richiedendo un numero elevatissimo di iterazioni.

### 📂 METODO 2: Gradiente Coniugato (Conjugate Gradient - CG)
*   **Direzione di discesa:** Le direzioni di ricerca $p_k$ non sono semplicemente ortogonali, ma sono **$A$-coniugate** (ortogonali rispetto al prodotto scalare indotto da $A$, ovvero $p_k^T A p_j = 0$ per $k \neq j$).
*   **Proprietà d'Oro d'Esame:** In aritmetica esatta, il metodo garantisce la convergenza alla soluzione esatta in un numero di passi **al massimo pari alla dimensione $n$ della matrice**. Elimina l'effetto zigzag ed è straordinariamente più efficiente del Gradiente standard su matrici mal condizionate.

---

## 💻 2. Il Codice Python Completo d'Esame

```python
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. IMPLEMENTAZIONE DEI ALGORITMI
# ==========================================

def metodo_gradiente(A, b, x0, tol=1e-6, max_it=500):
    """Risoluzione di Ax=b tramite il Metodo del Gradiente Standard"""
    x = x0.copy()
    r = b - A @ x
    it = 0
    errori = []
    
    while np.linalg.norm(r) / np.linalg.norm(b) > tol and it < max_it:
        # Calcolo del passo ottimale alfa
        alfa = np.dot(r.T, r) / np.dot(r.T, A @ r)
        x = x + alfa * r
        r = b - A @ x  # Aggiornamento del residuo
        
        errori.append(np.linalg.norm(r) / np.linalg.norm(b))
        it += 1
        
    return x, it, errori

def metodo_gradiente_coniugato(A, b, x0, tol=1e-6, max_it=500):
    """Risoluzione di Ax=b tramite il Metodo del Gradiente Coniugato"""
    x = x0.copy()
    r = b - A @ x
    p = r.copy() # La direzione iniziale coincide con il residuo
    it = 0
    errori = []
    
    while np.linalg.norm(r) / np.linalg.norm(b) > tol and it < max_it:
        Ap = A @ p
        # Calcolo del passo ottimale alfa lungo la direzione p
        alfa = np.dot(r.T, r) / np.dot(p.T, Ap)
        x = x + alfa * p
        
        r_old = r.copy()
        r = r - alfa * Ap  # Aggiornamento del residuo
        
        # Calcolo del parametro beta per rendere le direzioni A-coniugate
        beta = np.dot(r.T, r) / np.dot(r_old.T, r_old)
        p = r + beta * p  # Nuova direzione di ricerca
        
        errori.append(np.linalg.norm(r) / np.linalg.norm(b))
        it += 1
        
    return x, it, errori

# ==========================================
# 2. CONFIGURAZIONE DEL TEST D'ESAME
# ==========================================
# Esempio di matrice sparsa, simmetrica e def. positiva
n = 100
A = np.diag(np.ones(n)*4) + np.diag(np.ones(n-1)*-1, 1) + np.diag(np.ones(n-1)*-1, -1)
b = np.ones((n, 1))
x0 = np.zeros((n, 1))

# Esecuzione
x_g, it_g, err_g = metodo_gradiente(A, b, x0)
x_cg, it_cg, err_cg = metodo_gradiente_coniugato(A, b, x0)

print(f"--- CONFRONTO CONVERGENZA ---")
print(f"Gradiente Standard:  Iterazioni = {it_g}")
print(f"Gradiente Coniugato: Iterazioni = {it_cg}")

# Plot di confronto in scala logaritmica (Richiesto all'esame)
plt.figure(figsize=(8, 5))
plt.semilogy(range(it_g), err_g, 'r-', label='Gradiente Standard')
plt.semilogy(range(it_cg), err_cg, 'b--', label='Gradiente Coniugato')
plt.xlabel('Iterazioni')
plt.ylabel('Residuo Relativo')
plt.title('Confronto dei Metodi di Discesa')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

```

---

## 💬 3. La Risposta Teorica Standard da scrivere nel Notebook

Quando nel tema d'esame (come quello del 7 Maggio) viene chiesto di *"giustificare la scelta del metodo di discesa più idoneo valutando le performance grafiche"*, la risposta da inserire è:

> *"I metodi di discesa affrontano la risoluzione del sistema lineare interpretandolo come un problema di minimizzazione della funzione quadratica energia $\Phi(x)$. Entrambi gli algoritmi richiedono come vincolo strutturale tassativo che la matrice dei coefficienti sia simmetrica e definita positiva. Il monitoraggio numerico e grafico evidenzia una netta superiorità del Metodo del Gradiente Coniugato rispetto al Gradiente Standard (Steepest Descent). Il Gradiente Standard, muovendosi esclusivamente lungo la direzione del residuo corrente (gradiente negativo), soffre di una convergenza fortemente penalizzata dal condizionamento della matrice, procedendo con un andamento a zigzag che ne dilata il numero di iterazioni. Il Gradiente Coniugato risolve questa inefficienza geometrica imponendo che le direzioni di ricerca siano reciprocamente $A$-coniugate. Tale proprietà teorica garantisce l'annullamento dell'errore lungo le direzioni già esplorate, determinando la convergenza in un numero di passi nettamente inferiore e cinematicamente limitato, in aritmetica esatta, dalla dimensione stessa del sistema."*

---

[⬅️ Torna all'Indice Principale](README.md)