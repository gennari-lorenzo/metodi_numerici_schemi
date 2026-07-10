# 📋 Tipologia 5 bis: Sistemi Lineari Sovradeterminati e Minimi Quadrati
*(Sistemi Rettangolari, Equazioni Normali, Fattorizzazione QR e Calcolo del Residuo)*

---

## 🎯 Obiettivo dell'Esercizio
In questa tipologia d'esame, si affronta un sistema lineare $Ax = b$ **sovradeterminato**, ovvero dove la matrice $A$ è rettangolare di dimensioni $m \times n$ con $m > n$ (più equazioni che incognite). 
Poiché un tale sistema è generalmente impossibile nel senso classico, l'obiettivo è trovare la soluzione $x^*$ nel senso dei **minimi quadrati**, ovvero il vettore che minimizza la norma euclidea del residuo:
$$\min_{x} \|Ax - b\|_2$$

Viene richiesto di risolvere il problema sia tramite le **Equazioni Normali** (metodo più diretto ma instabile se la matrice è mal condizionata) sia tramite la **Fattorizzazione QR** (metodo numericamente robusto), confrontando i risultati e calcolando la norma del residuo finale.

---

## 🔍 1. Criteri Scientifici e Teoria d'Esame (Da Imparare a Memoria)

### 📂 METODO 1: Equazioni Normali
*   **Principio Teorico:** Minimizzare il residuo equivale geometricamente a imporre che il residuo stesso sia ortogonale allo spazio generato dalle colonne di $A$. Ciò porta al sistema quadrato ($n \times n$):
    $$(A^T A)x = A^T b$$
*   **Condizione di Esistenza:** La matrice quadrata $A^T A$ è simmetrica e definita positiva se e solo se la matrice $A$ ha rango massimo ($rank(A) = n$, ovvero colonne linearmente indipendenti). In tal caso, si applica la fattorizzazione di **Cholesky** ($LL^T$) per risolvere il sistema in modo efficiente.
*   **Svantaggio Computazionale:** Il condizionamento del sistema si deteriora drasticamente: $\kappa(A^T A) = [\kappa(A)]^2$. Se $A$ è moderatamente mal condizionata, $A^T A$ diventa numericamente singolare o quasi, amplificando gli errori di arrotondamento.

### 📂 METODO 2: Fattorizzazione QR
*   **Principio Teorico:** Si decompone la matrice $A$ nel prodotto $A = QR$, dove $Q$ è una matrice ortogonale ($m \times m$) tale che $Q^T Q = I$, e $R$ è una matrice trapezoidale superiore ($m \times n$). 
*   **Risoluzione del Problema:** Sfruttando l'invarianza della norma 2 rispetto alle trasformazioni ortogonali, il problema si riduce a:
    $$\|Ax - b\|_2 = \|Q R x - b\|_2 = \|R x - Q^T b\|_2$$
    Separando la matrice $R$ e il termine noto trasformato $c = Q^T b$ nelle componenti superiori ($n$ righe) e inferiori ($m-n$ righe nulle di $R$), si ottiene un sistema triangolare superiore quadrato:
    $$R_1 x = c_1$$
    risolvibile istantaneamente tramite sostituzione all'indietro.
*   **Vantaggio Computazionale:** Preserva il condizionamento originale del problema: $\kappa(R_1) = \kappa(A)$. È il metodo standard per l'algebra computazionale robusta.

---

## 💻 2. Il Codice Python Completo d'Esame

```python
import numpy as np
import scipy.linalg as spl

# ==========================================
# 1. GENERAZIONE DEI DATI (Esempio d'Esame)
# ==========================================
# Matrice A (m x n) con m > n, e vettore b (m)
np.random.seed(42)  # Per riproducibilità
m, n = 10, 4

A = np.random.rand(m, n)
b = np.random.rand(m, 1)

print(f"Dimensioni di A: {A.shape} | Rango di A: {np.linalg.matrix_rank(A)}")

# ==========================================
# 2. RISOLUZIONE TRAMITE EQUAZIONI NORMALI
# ==========================================
# Calcolo dei componenti del sistema quadrato (A^T * A)x = A^T * b
ATA = A.T @ A
ATb = A.T @ b

# Risoluzione ottimale sfruttando la simmetria (Cholesky o fattorizzazione standard)
# Nota: Usiamo Cholesky supponendo che ATA sia definita positiva
L = spl.cholesky(ATA, lower=True)
# Risoluzione del sistema tramite i fattori di Cholesky pre-calcolati (L, lower=True)
x_eq_normali = spl.cho_solve((L, True), ATb)

# ==========================================
# 3. RISOLUZIONE TRAMITE FATTORIZZAZIONE QR
# ==========================================
# Calcolo della fattorizzazione QR in modalità "economy" (Q: m x n, R: n x n)
Q, R = spl.qr(A, mode='economic')

# Trasformazione del termine noto: c = Q^T * b
c = Q.T @ b

# Risoluzione del sistema triangolare superiore R * x = c
x_qr = spl.solve_triangular(R, c)

# ==========================================
# 4. CALCOLO E CONFRONTO DEI RESIDUI
# ==========================================
# Residuo standard: r = b - A*x
residuo_normali = b - A @ x_eq_normali
residuo_qr = b - A @ x_qr

norma_res_normali = np.linalg.norm(residuo_normali, 2)
norma_res_qr = np.linalg.norm(residuo_qr, 2)

print(f"\n--- CONFRONTO MINIMI QUADRATI ---")
print(f"Sol. Equazioni Normali (primi 3 elementi): \n{x_eq_normali[:3].flatten()}")
print(f"Sol. QR (primi 3 elementi):             \n{x_qr[:3].flatten()}")
print(f"Norma Residuo (Eq. Normali):            {norma_res_normali:.14f}")
print(f"Norma Residuo (QR):                     {norma_res_qr:.14f}")
print(f"Distanza tra le due soluzioni:          {np.linalg.norm(x_eq_normali - x_qr, 2):.14e}")
```

---

## 💬 3. La Risposta Teorica Standard da scrivere nel Notebook

Quando viene richiesto di *"commentare l'accuratezza e la stabilità dei metodi numerici utilizzati per risolvere il problema ai minimi quadrati"*, la struttura ideale della risposta è:

> *"L'analisi numerica del sistema sovraterminato conferma le proprietà di stabilità asintotica attese dalla teoria. Il metodo delle Equazioni Normali, pur essendo computazionalmente diretto ed efficiente (richiedendo solo la risoluzione di un sistema simmetrico definito positivo tramite Cholesky), risente intrinsecamente del peggioramento quadratico del condizionamento della matrice di partenza, in quanto $\kappa(A^T A) = [\kappa(A)]^2$. Questa amplificazione rende l'algoritmo vulnerabile in presenza di matrici quasi-singolari o fortemente sbilanciate. Al contrario, l'approccio basato sulla Fattorizzazione QR si dimostra numericamente robusto e superiore; operando una trasformazione ortogonale che preserva la geometria delle lunghezze (norma 2), il condizionamento del sistema triangolare superiore risultante rimane pari a $\kappa(A)$. Sebbene su matrici ben condizionate i due metodi convergano a residui geometricamente equivalenti e a scarti trascurabili nell'ordine della precisione di macchina, l'utilizzo della decomposizione QR è da preferirsi sistematicamente nei problemi di regressione e fitting industriale per garantire la massima accuratezza algebrica."*

---

## 🏁 Checklist d'Esame: Come completare l'esercizio
1. [ ] Verifica le dimensioni della matrice `A` ($m > n$) per confermare che si tratti di un problema sovraterminato.
2. [ ] Controlla che il rango di `A` sia massimo (`matrix_rank(A) == n`) per assicurare l'univocità della soluzione.
3. [ ] Nelle equazioni normali, esegui i prodotti matriciali nell'ordine corretto (`A.T @ A`) ed evita l'inversione esplicita.
4. [ ] Nella fattorizzazione QR, usa l'argomento `mode='economic'` in `scipy.linalg.qr` per estrarre direttamente la porzione quadrata superiore di $R$ e ottimizzare i tempi di calcolo.
5. [ ] Ricorda che la norma del residuo numerico finale non sarà mai nulla (a differenza dei sistemi quadrati determinati), ma rappresenterà la minima distanza euclidea geometricamente possibile.
