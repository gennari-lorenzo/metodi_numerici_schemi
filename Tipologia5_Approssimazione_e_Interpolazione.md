# 📋 Tipologia 5: Approssimazione e Interpolazione di Dati
*(Minimi Quadrati, Polinomi di Lagrange/Newton, Fenomeno di Runge e Nodi di Chebyshev)*

---

## 🎯 Obiettivo dell'Esercizio
In questa tipologia d'esame ti vengono forniti dei punti sperimentali $(x_i, y_i)$. L'obiettivo è duplice:
1.  **Approssimazione ai Minimi Quadrati:** Trovare un polinomio di grado basso ($m \ll n$) che catturi l'andamento generale dei dati senza necessariamente passare per ogni singolo punto (ideale per dati affetti da rumore).
2.  **Interpolazione Polinomiale:** Trovare l'unico polinomio di grado $n$ che passa *esattamente* per tutti gli $n+1$ punti forniti. Sperimenterai anche come la scelta dei nodi (equispaziati vs. Chebyshev) influenzi drasticamente la stabilità dell'approssimazione al crescere del grado (Fenomeno di Runge).

---

## 🔍 1. Criteri Scientifici e Teoria d'Esame (Da Imparare a Memoria)

### 📂 CASO 1: Retta o Polinomio dei Minimi Quadrati
*   **Quando si usa:** Quando il numero di punti $n$ è molto elevato e i dati contengono errori di misura. Non vogliamo che la curva oscilli selvaggiamente per toccare ogni punto.
*   **Equazioni Normali:** Matematicamente si risolve il sistema lineare legato alla matrice di Vandermonde ridotta:
    $$A^T A \alpha = A^T y$$
    Dove $A_{ij} = x_i^j$ e $\alpha$ contiene i coefficienti del polinomio ottimale.

### 📂 CASO 2: Interpolazione Polinomiale e Nodali di Chebyshev
*   **Il problema dei Nodi Equispaziati:** Se si tenta di interpolare una funzione su nodi equispaziati usando un polinomio di grado elevato ($n \ge 10$), si verifica il **Fenomeno di Runge**: il polinomio manifesta forti oscillazioni distruttive vicino agli estremi dell'intervallo $[a, b]$, facendo schizzare l'errore a infinito.
*   **La Soluzione (Nodi di Chebyshev):** Per annullare il fenomeno di Runge, si distribuiscono i nodi in modo non uniforme, addensandoli verso gli estremi dell'intervallo. I nodi di Chebyshev nell'intervallo $[a, b]$ si calcolano con la formula:
    $$x_i = \frac{a+b}{2} + \frac{b-a}{2} \cos\left(\frac{2i+1}{2n+2} \pi\right) \quad \text{per } i = 0, \dots, n$$

---

## 💻 2. Il Codice Python Completo d'Esame

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spl

# ==========================================
# 1. DATI CAMPIONE (Tipico esercizio d'esame)
# ==========================================
# Supponiamo che l'esame dia dei vettori x e y o una funzione da campionare
x_dati = np.array([-2.0, -1.5, -0.5, 0.5, 1.0, 1.5, 2.0])
y_dati = np.array([4.1, 2.3, 0.5, 0.2, 1.1, 2.4, 4.2])

# Vettore di ascisse fitte per visualizzare le curve in modo fluido
x_fitto = np.linspace(min(x_dati), max(x_dati), 200)

# ==========================================
# 2. APPROSSIMAZIONE AI MINIMI QUADRATI (Grado m = 2)
# ==========================================
grado_m = 2
# np.polyfit calcola i coefficienti ottimizzando lo scarto quadratico medio
coeff_mq = np.polyfit(x_dati, y_dati, grado_m)
# np.polyval valuta il polinomio nei punti desiderati
y_mq_fitto = np.polyval(coeff_mq, x_fitto)

# ==========================================
# 3. INTERPOLAZIONE DI LAGRANGE (Passaggio Esatto)
# ==========================================
# Interpolazione polinomiale stabile mediante formula baricentrica
# Valuta direttamente il polinomio interpolante sui punti fitti senza passare per Vandermonde
y_interp_fitto = spi.barycentric_interpolate(x_dati, y_dati, x_fitto)

# ==========================================
# 4. CALCOLO DEI NODI DI CHEBYSHEV (Richiesto esplicitamente)
# ==========================================
def nodi_chebyshev(a, b, n_nodi):
    i = np.arange(n_nodi)
    # Formula matematica di Chebyshev mappata sull'intervallo [a, b]
    nodi = (a + b) / 2 + (b - a) / 2 * np.cos(((2 * i + 1) / (2 * n_nodi)) * np.pi)
    return nodi

# Esempio su intervallo [-2, 2] con 10 nodi per abbattere Runge
nodi_cheb = nodi_chebyshev(-2, 2, 10)
print("Nodi di Chebyshev calcolati:\n", nodi_cheb)

# ==========================================
# 5. GRAFICO COMPARATIVO DI CONTROLLO
# ==========================================
plt.figure(figsize=(9, 5))
plt.scatter(x_dati, y_dati, color='red', zorder=5, label='Dati Sperimentali')
plt.plot(x_fitto, y_mq_fitto, '--', color='blue', label=f'Minimi Quadrati (Grado {grado_m})')
plt.plot(x_fitto, y_interp_fitto, '-', color='green', label='Polinomio Interpolante (Esatto)')
plt.title("Analisi Numerica: Minimi Quadrati vs Interpolazione")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
```

---

## 💬 3. La Risposta Teorica Standard da scrivere nel Notebook

Quando la prof ti chiede di *"commentare la differenza tra l'approccio dei minimi quadrati e l'interpolazione polinomiale, facendo riferimento alle oscillazioni e al posizionamento dei nodi"*, rispondi così:

> *"L'interpolazione polinomiale classica richiede che il polinomio passi esattamente per tutti gli $n+1$ punti dati, imponendo un grado pari a $n$. All'aumentare dei punti su nodi equispaziati, questo approccio soffre pesantemente del Fenomeno di Runge, manifestando forti oscillazioni fittizie e instabilità numerica ai bordi del dominio. Per mitigare tale comportamento distruttivo è indispensabile adottare i Nodi di Chebyshev, i quali addensano i punti di campionamento verso gli estremi dell'intervallo riducendo drasticamente la norma dell'operatore di Lebesgue. Al contrario, l'approccio ai minimi quadrati non vincola il polinomio al passaggio esatto per i nodi, ma minimizza la norma euclidea del vettore dei residui utilizzando un grado $m$ deliberatamente basso rispetto al numero di punti ($m \ll n$). Questo scherma l'approssimazione dal rumore sperimentale e previene l'insorgenza di oscillazioni spurie, garantendo un trend globale stabile."*

---

## 🏁 Checklist d'Esame: Come completare l'esercizio
1. [ ] Controlla il grado richiesto per i minimi quadrati: solitamente è lineare ($m=1$) o parabolico ($m=2$).
2. [ ] Usa `np.polyfit(x, y, grado)` per estrarre i coefficienti ordinati dal grado più alto al termine noto.
3. [ ] Usa `np.polyval(coeff, x_punti)` per plottare o calcolare i valori nei punti d'interesse.
4. [ ] Se la prof chiede l'interpolazione esatta tramite risoluzione del sistema, ricorda che la matrice di Vandermonde si implementa con `np.vander(x)`.
5. [ ] Se compare la funzione di Runge ($f(x) = \frac{1}{1 + 25x^2}$), scrivi immediatamente la formula dei nodi di Chebyshev per ottenere il massimo dei voti.
