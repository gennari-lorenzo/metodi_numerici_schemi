# 📋 Tipologia 2: Metodi Diretti per Sistemi Lineari Quadrati
*(Fattorizzazioni LU, Cholesky, QR e Analisi del Condizionamento)*

---

## 🎯 Obiettivo dell'Esercizio
In questa tipologia d'esame, ti viene fornito un file `.mat` contenente una matrice $A$ e un termine noto $b$. L'obiettivo è analizzare le proprietà numeriche e strutturali di $A$ tramite verifiche preliminari al computer, **scegliere matematicamente il metodo migliore** per risolvere il sistema $Ax = b$, implementarlo e commentare i risultati.

---

## 🔍 1. Criteri Scientifici di Scelta del Metodo (Da Imparare a Memoria)

La scelta del metodo non si fa ad intuito, ma dipende dall'esito di tre verifiche numeriche precise da effettuare sul notebook.

```
                  [ Calcolo K(A) e Proprietà ]
                               |
            ---------------------------------------
           |                                       |
     K(A) >= 10^5                            K(A) < 10^5
           |                                       |
     📂 SCELGO QR                            -----------------------
 (Matrice Mal Condizionata)                 |                       |
                                      Simmetrica &            NON Simmetrica /
                                   Definita Positiva          Def. Positiva
                                            |                       |
                                      📂 SCELGO CHOLESKY        📂 SCELGO LU
                                    (Costo Computazionale     (Gauss con Pivoting)
                                           Dimezzato)
```

### 📂 OPZIONE 1: Fattorizzazione LU (Gauss con Pivoting Parziale)
*   **Quando si usa:** Quando la matrice è **ben condizionata** ($K(A) < 10^5$) e **NON** è simmetrica e definita positiva.
*   **Soglia Numerica:** Numero di condizionamento $K(A) < 10^5$.
*   **Giustificazione Teorica:** *"Scelgo la fattorizzazione LU con pivoting parziale poiché la matrice è ben condizionata. Il pivoting scherma l'algoritmo dalla crescita dei coefficienti, garantendo stabilità algoritmica e un errore finale proporzionale alla precisione di macchina."*

### 📂 OPZIONE 2: Fattorizzazione di Cholesky ($A = L L^T$)
*   **Quando si usa:** Quando la matrice è **simmetrica** ($A = A^T$) e **definita positiva** (tutti gli autovalori $\lambda_i > 0$).
*   **Soglia Numerica:** $K(A) < 10^5$, $A = A^T$ e $\min(\lambda_i) > 0$.
*   **Giustificazione Teorica:** *"Scelgo la fattorizzazione di Cholesky poiché la matrice è simmetrica e definita positiva. Questo metodo è computazionalmente ottimale poiché richiede la metà delle operazioni fluttuanti rispetto a LU ($\frac{n^3}{3}$ contro $\frac{2n^3}{3}$) e dimezza lo spazio di memoria necessario, preservando la stabilità senza bisogno di pivoting."*

### 📂 OPZIONE 3: Fattorizzazione QR ($A = Q R$)
*   **Quando si usa:** Quando la matrice è **mal condizionata** (indipendentemente dal fatto che sia simmetrica o meno).
*   **Soglia Numerica:** Numero di condizionamento $K(A) \ge 10^5$ (spesso nei testi d'esame arriva a $10^{12}$ o oltre).
*   **Giustificazione Teorica:** *"Scelgo la fattorizzazione QR poiché la matrice presenta un forte mal condizionamento ($K(A) \ge 10^5$). Il metodo QR utilizza matrici ortogonali $Q$ tramite trasformazioni di Householder o rotazioni di Givens; poiché le matrici ortonormali conservano la norma euclidea ($\|Qx\|_2 = \|x\|_2$), esse non amplificano gli errori di arrotondamento preesistenti, garantendo una stabilità numerica robusta dove LU fallirebbe."*

---

## 💻 2. Il Codice Python Completo d'Esame

Ecco il blocco di codice unico, strutturato in modo da effettuare i test e applicare automaticamente la scomposizione corretta.

```python
import numpy as np
import scipy.linalg as spl
from scipy.io import loadmat

# ==========================================
# 1. CARICAMENTO DATI DAL FILE .MAT
# ==========================================
dati = loadmat('testII.mat')  # Sostituisci con il nome del file dell'esame
A = dati["A"].astype(float)
b = dati["b"].astype(float)

# ==========================================
# 2. VERIFICHE PRELIMINARI DI SCELTA
# ==========================================
cond_A = np.linalg.cond(A)
is_simmetrica = np.allclose(A, A.T)

print(f"--- ANALISI DELLA MATRICE ---")
print(f"Numero di condizionamento K(A): {cond_A:.2e}")
print(f"La matrice è simmetrica?: {is_simmetrica}\n")

# ==========================================
# 3. APPLICAZIONE DEL METODO IDONEO
# ==========================================
if cond_A >= 1e5:
    print("-> RISOLUZIONE TRAMITE METODO QR (Matrice Mal Condizionata)")
    Q, R = spl.qr(A, mode='economic')
    y = Q.T @ b
    x_sol = spl.solve_triangular(R, y, lower=False)

elif is_simmetrica:
    # Verifichiamo se è definita positiva tentando direttamente Cholesky
    try:
        L_ch = spl.cholesky(A, lower=True)
        print("-> RISOLUZIONE TRAMITE METODO CHOLESKY (Simmetrica + Def. Positiva)")
        y = spl.solve_triangular(L_ch, b, lower=True)
        x_sol = spl.solve_triangular(L_ch.T, y, lower=False)
    except np.linalg.LinAlgError:
        print("-> RISOLUZIONE TRAMITE METODO LU (Simmetrica ma NON Def. Positiva)")
        P, L, U = spl.lu(A)
        y = spl.solve_triangular(L, P.T @ b, lower=True)
        x_sol = spl.solve_triangular(U, y, lower=False)

else:
    print("-> RISOLUZIONE TRAMITE METODO LU (Standard, Ben Condizionata)")
    P, L, U = spl.lu(A)
    y = spl.solve_triangular(L, P.T @ b, lower=True)
    x_sol = spl.solve_triangular(U, y, lower=False)

# ==========================================
# 4. VALUTAZIONE ACCURATEZZA (ERRORE E RESIDUO)
# ==========================================
# Soluzione di riferimento interna stocastica
x_true = np.linalg.solve(A, b)

# Errore Relativo sulla soluzione
err_rel = np.linalg.norm(x_sol - x_true) / np.linalg.norm(x_true)
# Residuo Relativo sull'equazione originaria
residuo_rel = np.linalg.norm(np.dot(A, x_sol) - b) / np.linalg.norm(b)

print(f"\n--- RISULTATI NUMERICI ---")
print(f"Errore Relativo rispetto al riferimento: {err_rel:.2e}")
print(f"Residuo Relativo finale: {residuo_rel:.2e}")
```
---

## 💬 3. La Risposta Teorica Standard da scrivere nel Notebook

Quando la prof chiede di *"giustificare la scelta della scomposizione effettuata sulla base dell'analisi del condizionamento e della struttura algebra della matrice"*, rispondi a seconda del caso riscontrato:

> **[Se hai scelto CHOLESKY]:** *"L'analisi preliminare della matrice ha mostrato un numero di condizionamento $K(A) < 10^5$, denotando un problema intrinsecamente ben condizionato, accoppiato a una struttura simmetrica e definita positiva. Si è pertanto optato per la scomposizione di Cholesky $A = LL^T$. Questa scelta è computazionalmente ottimale poiché richiede solamente $\frac{n^3}{3}$ operazioni fluttuanti (pari alla metà del costo di una scomposizione LU standard) e dimezza lo stoccaggio di memoria richiesto memorizzando solo il triangolo inferiore. Inoltre, la stabilità numerica intrinseca dell'algoritmo elimina la necessità di operazioni di pivoting, preservando l'accuratezza senza overhead strutturale."*
>
> **[Se hai scelto LU]:** *"Dai test effettuati risulta che la matrice è ben condizionata ($K(A) < 10^5$), ma non presenta simultaneamente i requisiti di simmetria e definizione positiva necessari per Cholesky. Si è scelto quindi il metodo standard di scomposizione LU con pivoting parziale tramite matrice di permutazione $P$ ($PA = LU$). Il ricorso al pivoting parziale scherma l'algoritmo da divisioni per pivot nulli o prossimi allo zero, limitando la crescita dei coefficienti ed assicurando la stabilità dell'algoritmo. Le sostituzioni triangolari successive mantengono l'errore finale entro i limiti imposti dall'accuratezza di macchina."*
>
> **[Se hai scelto QR]:** *"I controlli numerici indicano che la matrice presenta un forte mal condizionamento ($K(A) \ge 10^5$). In tale regime di instabilità intrinseca, i metodi LU e Cholesky fallirebbero a causa dell'amplificazione distruttiva degli errori di arrotondamento. Si è preferito adottare la fattorizzazione $A=QR$ basata su trasformazioni ortogonali (Householder/Givens). Poiché le matrici ortonormali conservano la norma euclidea ($\|Qx\|_2 = \|x\|_2$), esse godono di un numero di condizionamento pari a 1, garantendo che le operazioni di trasformazione del termine noto non propaghino l'errore, preservando la massima accuratezza possibile sulla soluzione numerica finale."*

---

## 🏁 Checklist d'Esame: Come completare l'esercizio
1. [ ] Carica la matrice $A$ e il vettore $b$ con `loadmat`.
2. [ ] Stampa a schermo il valore di `np.linalg.cond(A)`.
3. [ ] Controlla simmetria e positività degli autovalori se il condizionamento è inferiore a $10^5$.
4. [ ] Incolla il blocco risolutivo triangolare (`spl.solve_triangular`) corrispondente alla scelta fatta.
5. [ ] Calcola e mostra l'errore relativo e il residuo relativo.
6. [ ] Scrivi la spiegazione teorica precompilata nella cella Markdown motivando la scelta in base alle soglie numeriche ottenute.

---
[⬅️ Torna all'Indice Principale](README.md)
