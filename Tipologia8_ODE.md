# 📋 Tipologia 8: Risoluzione Numerica di Equazioni Differenziali (ODE)
*(Problema di Cauchy, Metodo di Eulero Avanti, Eulero Indietro, Runge-Kutta 4 e Analisi di Stabilità)*

---

## 🎯 Obiettivo dell'Esercizio
In questa tipologia d'esame si affronta un problema di Cauchy (equazione differenziale ordinaria del primo grado con condizione iniziale):
$$\begin{cases} y'(t) = f(t, y(t)) \quad t \in [t_0, t_{\text{end}}] \\\\ y(t_0) = y_0 \end{cases}$$

L'obiettivo è calcolare l'approssimazione numerica della funzione soluzione $y(t)$ su una griglia discreta di tempi con passo $h$, confrontando metodi **espliciti** (Eulero Avanti, RK4) e **impliciti** (Eulero Indietro), valutandone l'accuratezza rispetto alla soluzione analitica esatta e discutendo i vincoli di stabilità algoritmica.

---

## 🔍 1. Criteri Scientifici e Teoria d'Esame (Da Imparare a Memoria)

### 📂 METODO 1: Eulero Avanti (Esplicito)
*   **Formula:** $y_{k+1} = y_k + h \cdot f(t_k, y_k)$
*   **Ordine di Convergenza:** Pari a **1** (lineare rispetto al passo $h$). L'errore globale decade come $O(h)$.
*   **Stabilità (Il vero tranello):** È un metodo **condizionatamente stabile**. Se il problema è *stiff* (ovvero la derivata parziale $\frac{\partial f}{\partial y} = \lambda$ ha componenti negative molto grandi), Eulero Avanti converge **solo se** il passo $h$ soddisfa il vincolo:
    $$h < \frac{2}{|\lambda|}$$
    Se usi un $h$ troppo grande, la soluzione numerica oscilla selvaggiamente ed esplode all'infinito.

### 📂 METODO 2: Eulero Indietro (Implicito)
*   **Formula:** $y_{k+1} = y_k + h \cdot f(t_{k+1}, y_{k+1})$
*   **Ordine di Convergenza:** Pari a **1** (lineare).
*   **Stabilità:** È un metodo **A-stabile** (incondizionatamente stabile). Converge strutturalmente per **qualsiasi** passo $h > 0$, indipendentemente da quanto il problema sia instabile o *stiff*.
*   **Difficoltà Computazionale:** Poiché $y_{k+1}$ compare sia a destra che a sinistra del segno di uguale, ad ogni singolo passo temporale bisogna risolvere un'equazione non lineare (spesso tramite il metodo di Newton-Raphson o con la funzione `scipy.optimize.fsolve`).

### 📂 METODO 3: Runge-Kutta 4 (RK4 - Esplicito)
*   **Formula:** Calcola 4 stime intermedie della pendenza ($k_1, k_2, k_3, k_4$) per fare una media pesata accurata:
    $$k_1 = f(t_k, y_k), \quad k_2 = f\left(t_k + \frac{h}{2}, y_k + \frac{h}{2}k_1\right)$$
    $$k_3 = f\left(t_k + \frac{h}{2}, y_k + \frac{h}{2}k_2\right), \quad k_4 = f(t_k + h, y_k + h k_3)$$
    $$y_{k+1} = y_k + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$
*   **Ordine di Convergenza:** Pari a **4** ($O(h^4)$). È l'algoritmo esplicito standard per l'altissima precisione a basso costo computazionale.

---

## 💻 2. Il Codice Python Completo d'Esame

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ==========================================
# 1. DEFINIZIONE DEL PROBLEMA (Esempio d'Esame)
# ==========================================
# Esempio: y'(t) = -2*y(t), t in [0, 2], y(0) = 1. (Sol. Esatta: y(t) = exp(-2t))
f = lambda t, y: -2.0 * y
sol_esatta = lambda t: np.exp(-2.0 * t)

t0, t_end = 0.0, 2.0
y0 = 1.0
N = 20  # Numero di passi temporali
h = (t_end - t0) / N
t_griglia = np.linspace(t0, t_end, N + 1)

# ==========================================
# 2. IMPLEMENTAZIONE METODI
# ==========================================

# --- Eulero Avanti (Esplicito) ---
y_avanti = np.zeros(N + 1)
y_avanti[0] = y0
for k in range(N):
    y_avanti[k+1] = y_avanti[k] + h * f(t_griglia[k], y_avanti[k])

# --- Eulero Indietro (Implicito con fsolve) ---
y_indietro = np.zeros(N + 1)
y_indietro[0] = y0
for k in range(N):
    # Definiamo l'equazione implicita: F(x) = x - y_k - h*f(t_k+1, x) = 0
    equazione_implicita = lambda x: x - y_indietro[k] - h * f(t_griglia[k+1], x)
    # Risolviamo l'equazione non lineare usando y_k come punto di partenza
    y_indietro[k+1] = fsolve(equazione_implicita, x0=y_indietro[k])[0]

# --- Runge-Kutta 4 (RK4) ---
y_rk4 = np.zeros(N + 1)
y_rk4[0] = y0
for k in range(N):
    tk = t_griglia[k]
    yk = y_rk4[k]
    
    k1 = f(tk, yk)
    k2 = f(tk + h/2.0, yk + (h/2.0)*k1)
    k3 = f(tk + h/2.0, yk + (h/2.0)*k2)
    k4 = f(tk + h, yk + h*k3)
    
    y_rk4[k+1] = yk + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

# ==========================================
# 3. VALUTAZIONE DEGLI ERRORI FINALI
# ==========================================
y_true_finale = sol_esatta(t_end)

err_avanti = np.abs(y_avanti[-1] - y_true_finale)
err_indietro = np.abs(y_indietro[-1] - y_true_finale)
err_rk4 = np.abs(y_rk4[-1] - y_true_finale)

print(f"--- CONFRONTO ERRORI ALL'ISTANTE FINALE T = {t_end} ---")
print(f"Eulero Avanti:   Sol = {y_avanti[-1]:.6f} | Errore Assoluto = {err_avanti:.2e}")
print(f"Eulero Indietro: Sol = {y_indietro[-1]:.6f} | Errore Assoluto = {err_indietro:.2e}")
print(f"Runge-Kutta 4:   Sol = {y_rk4[-1]:.6f} | Errore Assoluto = {err_rk4:.2e}")
```

---

## 💬 3. La Risposta Teorica Standard da scrivere nel Notebook
Quando la prof chiede di *"commentare le differenze tra i metodi espliciti ed impliciti analizzando l'accuratezza e il comportamento in termini di stabilità al variare del passo $h$"*, la struttura ideale della risposta è:

> *"L'analisi numerica evidenzia le classiche proprietà di stabilità e consistenza delle famiglie di solutori ODE. Il metodo di Eulero Avanti, essendo un algoritmo esplicito, risulta computazionalmente leggero ad ogni passo ma soffre di una severa limitazione di stabilità condizionata; se il passo $h$ violasse la barriera di stabilità legata ai coefficienti di rigidezza del problema (stiffness), l'errore di troncamento verrebbe amplificato distruttivamente producendo oscillazioni spurie esplosive. Al contrario, il metodo di Eulero Indietro mitiga completamente questo rischio grazie alle sue proprietà di A-stabilità; pagando il prezzo computazionale di dover risolvere un'equazione non lineare (tramite fsolve) ad ogni step temporale, esso garantisce la convergenza asintotica per qualsiasi ampiezza di passo $h$. Entrambi i metodi di Eulero manifestano comunque un errore globale lineare $O(h)$ dovuto al loro ordine di accuratezza pari a 1. La netta superiorità in termini di precisione è mostrata dal metodo Runge-Kutta 4 (RK4); sfruttando quattro valutazioni intermedie della pendenza, esso innalza l'ordine di consistenza a 4, abbattendo l'errore globale come $O(h^4)$ e garantendo risultati estremamente accurati anche su griglie temporali grossolane."*

---

## 🏁 Checklist d'Esame: Come completare l'esercizio
1. [ ] Controlla se il testo ti fornisce la soluzione analitica esatta per calcolare gli errori; in caso contrario, la soluzione di riferimento su cui calcolare l'errore relativo si ottiene eseguendo RK4 con un numero di passi altissimo (es. $N=1000$).
2. [ ] Verifica se il problema è stiff calcolando lo Jacobiano (o la derivata semplice) per giustificare nei commenti l'eventuale instabilità di Eulero Avanti.
3. [ ] All'interno del ciclo di Eulero Indietro, ricordati di passare la variabile x come incognita e il valore y_indietro[k] come termine noto agganciato alla funzione lambda di fsolve.
4. [ ] Quando plotti i grafici, usa sempre curve continue per le soluzioni numeriche e marcatori discreti (es. 'o') per evidenziare i nodi della griglia temporale.