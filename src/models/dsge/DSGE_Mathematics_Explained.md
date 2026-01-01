# DSGE Mathematics: A Beginner's Guide

This document explains DSGE models step by step, assuming only basic calculus and linear algebra.

---

## 1. What Is a DSGE Model?

DSGE stands for **Dynamic Stochastic General Equilibrium**.

- **Dynamic**: The model tracks how the economy evolves over time (today, tomorrow, next quarter...)
- **Stochastic**: Random shocks hit the economy (oil prices, productivity, policy surprises)
- **General Equilibrium**: All markets clear - supply equals demand everywhere, simultaneously

The core idea: derive macroeconomic behavior from individual decision-making. Households choose how much to consume and work. Firms choose prices and production. These choices, when aggregated, give us GDP, inflation, employment.

---

## 2. The Fundamental Question

Every period, a household must decide:

> "Should I consume today, or save and consume tomorrow?"

This is an **intertemporal choice**. The answer depends on:
- How impatient am I? (discount factor)
- What return will I get on savings? (interest rate)
- What do I expect prices to be tomorrow? (expected inflation)
- What do I expect my income to be? (expected wages)

DSGE models formalize this decision mathematically.

---

## 3. The Household's Problem

### 3.1 Utility

Households get happiness ("utility") from consumption $C$ and unhappiness from working $N$ hours:

$$U = u(C) - v(N)$$

A common specification:

$$U = \frac{C^{1-\sigma}}{1-\sigma} - \chi \frac{N^{1+\varphi}}{1+\varphi}$$

The parameter $\sigma$ controls how much extra happiness you get from extra consumption. When $\sigma$ is large, you don't value extra consumption much if you're already consuming a lot.

### 3.2 The Infinite Horizon

Households live forever (or act as dynasties). They care about utility today AND utility in all future periods:

$$\text{Total Utility} = U_0 + \beta U_1 + \beta^2 U_2 + \beta^3 U_3 + ...$$

$$= \sum_{t=0}^{\infty} \beta^t U_t$$

The parameter $\beta$ (between 0 and 1) is the **discount factor**. It captures impatience:
- $\beta = 0.99$ means you value next quarter's utility at 99% of today's
- Lower $\beta$ = more impatient

### 3.3 The Budget Constraint

Each period, your spending can't exceed your resources:

$$\text{Consumption} + \text{Savings} = \text{Wage Income} + \text{Last Period's Savings with Interest}$$

In symbols:
$$C_t + \frac{B_t}{P_t} = W_t N_t + (1+i_{t-1})\frac{B_{t-1}}{P_t}$$

Where:
- $B_t$ = bonds (savings) in dollars
- $P_t$ = price level
- $W_t$ = real wage
- $i_t$ = nominal interest rate

---

## 4. The Optimization Problem

The household wants to maximize total lifetime utility subject to the budget constraint holding every period.

$$\max \sum_{t=0}^{\infty} \beta^t U(C_t, N_t)$$

subject to: budget constraint each period.

This is a **constrained optimization problem** with infinitely many periods.

### 4.1 The Lagrangian Approach

For each period $t$, attach a multiplier $\lambda_t$ to the budget constraint:

$$\mathcal{L} = \sum_{t=0}^{\infty} \beta^t \left[ U(C_t, N_t) + \lambda_t \left( W_t N_t + (1+i_{t-1})\frac{B_{t-1}}{P_t} - C_t - \frac{B_t}{P_t} \right) \right]$$

### 4.2 First-Order Conditions

Take derivatives and set equal to zero:

**With respect to $C_t$:**
$$\frac{\partial U}{\partial C_t} = \lambda_t$$

The marginal utility of consumption equals the "shadow price" of wealth.

**With respect to $B_t$:**
$$\frac{\lambda_t}{P_t} = \beta \frac{\lambda_{t+1}(1+i_t)}{P_{t+1}}$$

The cost of saving one more dollar today equals the discounted benefit of having $(1+i_t)$ dollars tomorrow.

### 4.3 The Euler Equation

Combining these conditions:

$$\frac{\partial U}{\partial C_t} = \beta (1+i_t) \frac{P_t}{P_{t+1}} \frac{\partial U}{\partial C_{t+1}}$$

With our utility function where $\frac{\partial U}{\partial C} = C^{-\sigma}$:

$$C_t^{-\sigma} = \beta (1+i_t) \frac{1}{1+\pi_{t+1}} C_{t+1}^{-\sigma}$$

where $\pi_{t+1} = \frac{P_{t+1}}{P_t} - 1$ is the inflation rate.

**In words:** The marginal utility of consuming today equals the discounted marginal utility of saving, earning interest, and consuming tomorrow (adjusted for inflation).

This is the **Euler equation** - the fundamental equation of intertemporal choice.

---

## 5. The Problem of Expectations

Look at the Euler equation again:

$$C_t^{-\sigma} = \beta (1+i_t) \frac{1}{1+\pi_{t+1}} C_{t+1}^{-\sigma}$$

Today's consumption $C_t$ depends on **tomorrow's** consumption $C_{t+1}$ and inflation $\pi_{t+1}$.

But tomorrow hasn't happened yet! So actually it depends on what we **expect**:

$$C_t^{-\sigma} = \beta (1+i_t) E_t\left[\frac{1}{1+\pi_{t+1}} C_{t+1}^{-\sigma}\right]$$

The notation $E_t[...]$ means "expectation conditional on information available at time $t$."

This creates a problem: to know what to do today, households must forecast the future. But the future depends on what everyone does today. The solution must be **self-consistent** - expectations must match what actually happens on average.

This is **rational expectations**: agents know the model and form expectations consistent with it.

---

## 6. The Bellman Equation (Dynamic Programming)

An alternative way to write the household's problem uses **recursive** methods.

Define the **value function** $V(B_{t-1})$ as the maximum utility achievable starting with wealth $B_{t-1}$:

$$V(B_{t-1}) = \max_{C_t, N_t, B_t} \left\{ U(C_t, N_t) + \beta E_t[V(B_t)] \right\}$$

subject to the budget constraint.

**In words:** The best you can do starting today equals:
- The utility you get today from your choices, PLUS
- The discounted expected value of the best you can do starting tomorrow

This is the **Bellman equation**. It converts an infinite-horizon problem into a two-period problem (today vs. the future).

The first-order conditions from the Bellman equation give the same Euler equation as before.

---

## 7. Firms and Price Setting

### 7.1 Production

Firms produce output using labor:

$$Y_t = A_t N_t$$

where $A_t$ is productivity (technology).

### 7.2 The Price-Setting Problem

In a perfectly competitive world, firms are price-takers. But the New Keynesian model assumes **sticky prices**: firms can't change prices every period.

**Calvo pricing:** Each period, a firm gets to reset its price with probability $(1-\theta)$. With probability $\theta$, it must keep last period's price.

When a firm CAN reset, it thinks: "I might be stuck with this price for several periods. What price should I set?"

It solves:

$$\max_{P^*} E_t \sum_{k=0}^{\infty} \theta^k \times (\text{profit if still charging } P^* \text{ in period } t+k)$$

The summation goes to infinity because the firm might be stuck with this price forever (with decreasing probability).

### 7.3 The New Keynesian Phillips Curve

When you solve the firm's problem and aggregate across all firms, you get:

$$\pi_t = \beta E_t[\pi_{t+1}] + \kappa \tilde{y}_t$$

Where:
- $\pi_t$ = inflation
- $\tilde{y}_t$ = output gap (actual output minus potential)
- $\kappa$ = a slope parameter that depends on price stickiness $\theta$

**In words:** Inflation today depends on:
1. Expected future inflation (because firms setting prices today care about future costs)
2. Current demand pressure (the output gap)

---

## 8. The Log-Linear Approximation

### 8.1 The Problem

The Euler equation and Phillips curve are **nonlinear**. Nonlinear systems are hard to solve analytically.

### 8.2 The Solution: Linearize

Assume the economy fluctuates around a **steady state** - a situation where nothing changes over time.

In steady state:
- $C_t = C_{t+1} = \bar{C}$ (consumption is constant)
- $\pi_t = \bar{\pi}$ (inflation is constant, often assumed zero)
- No shocks

Then approximate the equations for small deviations from steady state.

### 8.3 Log Deviations

Define the **log deviation** of any variable:

$$\hat{x}_t = \log(X_t) - \log(\bar{X}) \approx \frac{X_t - \bar{X}}{\bar{X}}$$

This is approximately the percentage deviation from steady state.

### 8.4 Linearizing the Euler Equation

Start with: $C_t^{-\sigma} = \beta(1+i_t)E_t\left[\frac{C_{t+1}^{-\sigma}}{1+\pi_{t+1}}\right]$

After linearization (the algebra is tedious but mechanical):

$$\hat{c}_t = E_t[\hat{c}_{t+1}] - \frac{1}{\sigma}(\hat{i}_t - E_t[\hat{\pi}_{t+1}])$$

**In words:** Consumption today (relative to trend) equals:
- Expected consumption tomorrow, MINUS
- A term involving the real interest rate (nominal rate minus expected inflation)

When real interest rates are high, you save more and consume less today.

This is the **IS curve** (in a closed economy, consumption = output, so $\hat{c}_t = \hat{y}_t$).

---

## 9. The Three-Equation Model

After linearization, the basic New Keynesian model is:

**IS Curve** (from household optimization):
$$\hat{y}_t = E_t[\hat{y}_{t+1}] - \frac{1}{\sigma}(\hat{i}_t - E_t[\hat{\pi}_{t+1}])$$

**Phillips Curve** (from firm optimization):
$$\hat{\pi}_t = \beta E_t[\hat{\pi}_{t+1}] + \kappa \hat{y}_t$$

**Taylor Rule** (central bank policy):
$$\hat{i}_t = \phi_\pi \hat{\pi}_t + \phi_y \hat{y}_t$$

The Taylor Rule is NOT derived from optimization - it's an assumed policy rule.

---

## 10. Writing the Model in Matrix Form

### 10.1 Why Matrices?

We have a system of equations with:
- Current variables ($\hat{y}_t$, $\hat{\pi}_t$, $\hat{i}_t$)
- Past variables (if any)
- Expected future variables ($E_t[\hat{y}_{t+1}]$, $E_t[\hat{\pi}_{t+1}]$)
- Shocks

Matrices let us write this compactly and solve systematically.

### 10.2 The Standard Form

Stack all variables into a vector $s_t$. The model becomes:

$$A_0 s_t = A_1 s_{t-1} + B \epsilon_t + C E_t[s_{t+1}]$$

Or in "Sims form":

$$\Gamma_0 s_t = \Gamma_1 s_{t-1} + \Psi \epsilon_t + \Pi \eta_t$$

Where:
- $\Gamma_0$, $\Gamma_1$ are matrices of coefficients
- $\epsilon_t$ is the vector of shocks
- $\eta_t$ captures expectational terms

### 10.3 Example

For the 3-equation model, one way to write it:

Let $s_t = \begin{pmatrix} \hat{y}_t \\ \hat{\pi}_t \\ \hat{i}_t \end{pmatrix}$

The IS curve: $\hat{y}_t + \frac{1}{\sigma}\hat{i}_t = E_t[\hat{y}_{t+1}] + \frac{1}{\sigma}E_t[\hat{\pi}_{t+1}]$

The Phillips curve: $\hat{\pi}_t = \beta E_t[\hat{\pi}_{t+1}] + \kappa \hat{y}_t$

The Taylor rule: $\hat{i}_t = \phi_\pi \hat{\pi}_t + \phi_y \hat{y}_t$

We can write this as matrix equations relating current, past, and expected future variables.

---

## 11. Two Types of Variables

This is crucial for understanding how DSGE models are solved.

### 11.1 Backward-Looking Variables (Predetermined)

Some variables are determined by the past. Examples:
- Capital stock (yesterday's investment becomes today's capital)
- Lagged values of shocks
- Any variable that can't change instantaneously

These are called **predetermined** or **backward-looking** variables.

Their value today was already determined yesterday - they move slowly.

### 11.2 Forward-Looking Variables

Other variables depend on expectations of the future. Examples:
- Consumption (depends on expected future income)
- Prices (depend on expected future costs)
- Asset prices (depend on expected future dividends)

These can **change instantly** when expectations change. If new information arrives, people immediately revise their consumption plans, firms revise their prices.

When news arrives, these variables adjust immediately to a new value.

### 11.3 Why This Matters

The number of each type determines whether a unique solution exists.

---

## 12. The Solution Problem

### 12.1 What We Want

A solution is a rule that tells us:
- Given last period's state and today's shocks...
- ...what are today's variables?

Mathematically: $s_t = T s_{t-1} + R \epsilon_t$

We want to find the matrices $T$ and $R$.

### 12.2 The Complication

The model contains expected future values. But future values depend on the solution!

We need to find a solution that is **self-consistent**: when agents use the solution to form expectations, and we solve the model with those expectations, we get back the same solution.

### 12.3 The Approach: Eigenvalue Decomposition

The system can be analyzed by looking at its **eigenvalues**.

Think of eigenvalues as measuring how explosive or stable different "modes" of the system are:
- Eigenvalue with $|\lambda| < 1$: this mode dies out over time (stable)
- Eigenvalue with $|\lambda| > 1$: this mode explodes over time (unstable)
- Eigenvalue with $|\lambda| = 1$: this mode persists (unit root)

---

## 13. Stable vs. Unstable Dynamics

### 13.1 Backward-Looking Dynamics

Consider a simple backward-looking system:

$$x_t = \lambda x_{t-1} + \epsilon_t$$

If $|\lambda| < 1$: shocks fade away. The system is **stable**.
If $|\lambda| > 1$: shocks amplify. The system **explodes**.

For the economy to make sense, backward-looking dynamics must be stable.

### 13.2 Forward-Looking Dynamics

Now consider:

$$x_t = \frac{1}{\lambda} E_t[x_{t+1}]$$

Rearranging: $E_t[x_{t+1}] = \lambda x_t$

This looks like: $x_{t+1} = \lambda x_t + (\text{surprise})$

If $|\lambda| > 1$: solving forward gives a **stable** solution where $x_t$ depends on expected future fundamentals.

If $|\lambda| < 1$: solving forward gives an **explosive** solution - no stable equilibrium.

**The intuition is reversed for forward-looking variables!** They need "unstable" eigenvalues to have stable solutions.

### 13.3 Why?

Forward-looking variables adjust today to ensure stability. If something bad is expected, consumption drops TODAY, which prevents the bad outcome from being as severe.

The immediate adjustment absorbs the explosive tendency, keeping the economy bounded.

---

## 14. The Blanchard-Kahn Conditions

### 14.1 The Counting Rule

For a unique stable solution to exist:

**The number of unstable eigenvalues (those with $|\lambda| > 1$) must equal the number of forward-looking variables.**

### 14.2 Intuition

- Each forward-looking variable is "free" to adjust - it can move to any value instantly
- Each unstable eigenvalue represents an explosive path
- The adjustments must exactly offset the explosive tendencies
- If they match up one-to-one, there's exactly one way to do this

### 14.3 Three Cases

**Case 1: Unstable eigenvalues = Forward-looking variables**

✓ **Unique stable solution exists** (Determinacy)

This is what we want. The model pins down a unique equilibrium.

**Case 2: Unstable eigenvalues < Forward-looking variables**

✗ **Multiple stable solutions exist** (Indeterminacy)

There are "too many" forward-looking variables. Multiple equilibria are possible. The economy could coordinate on any of them, including ones driven by "sunspots" (arbitrary beliefs).

**Case 3: Unstable eigenvalues > Forward-looking variables**

✗ **No stable solution exists**

There aren't enough forward-looking variables to offset all the explosive tendencies. Something is wrong with the model specification.

---

## 15. Determinacy and Monetary Policy

### 15.1 The Taylor Rule

$$\hat{i}_t = \phi_\pi \hat{\pi}_t + \phi_y \hat{y}_t$$

How aggressively should the central bank respond to inflation?

### 15.2 The Taylor Principle

For the NK model to have a unique equilibrium:

$$\phi_\pi > 1$$

The central bank must raise nominal interest rates **more than one-for-one** with inflation.

### 15.3 Intuition

Suppose inflation rises by 1%.

If $\phi_\pi > 1$: Nominal rate rises by more than 1%. Real rate rises. Demand falls. Inflation comes back down. Stable.

If $\phi_\pi < 1$: Nominal rate rises by less than 1%. Real rate falls. Demand rises. Inflation rises further. Unstable... unless expectations coordinate on something else.

With $\phi_\pi < 1$, the model is **indeterminate**. Inflation could be anything consistent with self-fulfilling expectations.

---

## 16. Solving the Model (Conceptually)

### 16.1 Step 1: Write in Matrix Form

$$\Gamma_0 s_t = \Gamma_1 s_{t-1} + \Psi \epsilon_t + \Pi \eta_t$$

### 16.2 Step 2: Compute Eigenvalues

Find eigenvalues of the system (technically, of $\Gamma_0^{-1} \Gamma_1$ or the generalized eigenvalues).

### 16.3 Step 3: Check Blanchard-Kahn

Count eigenvalues outside the unit circle. Compare to number of forward-looking variables.

### 16.4 Step 4: Decompose

Separate the system into stable and unstable components using eigenvalue decomposition.

### 16.5 Step 5: Impose Stability

Set unstable components to zero. This pins down the forward-looking variables.

### 16.6 Step 6: Obtain Solution

The result is:

$$s_t = T s_{t-1} + R \epsilon_t$$

Where $T$ and $R$ are matrices that depend on all the structural parameters.

---

## 17. The Solution Matrices

### 17.1 The Transition Matrix $T$

$T$ governs how the state evolves over time absent new shocks.

If all eigenvalues of $T$ are inside the unit circle, the system is stable: shocks die out over time.

### 17.2 The Impact Matrix $R$

$R$ shows how shocks affect the state on impact.

Column $j$ of $R$ shows the immediate effect of a one-unit shock $j$ on each variable.

### 17.3 Impulse Responses

How does the economy respond to a shock over time?

- Period 0: $s_0 = R \epsilon_0$
- Period 1: $s_1 = T s_0 = T R \epsilon_0$
- Period 2: $s_2 = T s_1 = T^2 R \epsilon_0$
- Period $h$: $s_h = T^h R \epsilon_0$

The impulse response at horizon $h$ is $T^h R$.

---

## 18. Connecting to Data: State-Space Representation

### 18.1 The Solved Model

After solving, we have:

**Transition equation:** $s_t = T s_{t-1} + R \epsilon_t$

This describes how latent (unobserved) states evolve over time.

### 18.2 Measurement Equation

But we don't observe all state variables directly. We observe GDP, inflation, interest rates, etc.

The **measurement equation** links states to observables:

$$y_t = Z s_t + D$$

Where:
- $y_t$ = vector of observable variables
- $Z$ = matrix that selects/transforms states into observables
- $D$ = constants (means, trends)

For example:
- GDP growth = change in log output + trend growth rate
- Observed inflation = model inflation + measurement error

### 18.3 The Kalman Filter

Given the state-space form, the **Kalman filter** provides:

1. **Filtered states**: Best estimate of $s_t$ given data up to time $t$
2. **Likelihood**: Probability of observing the data given parameters

The Kalman filter recursively:
- Predicts the next state given current information
- Updates when new data arrives
- Accumulates the likelihood

---

## 19. Bayesian Estimation

### 19.1 The Goal

Estimate the structural parameters $\theta$ (discount factor, price stickiness, policy responses, shock volatilities, etc.)

### 19.2 Bayes' Theorem

$$p(\theta|Y) = \frac{p(Y|\theta) \times p(\theta)}{p(Y)}$$

Where:
- $p(\theta|Y)$ = **posterior** - what we believe about parameters after seeing data
- $p(Y|\theta)$ = **likelihood** - probability of data given parameters
- $p(\theta)$ = **prior** - what we believed before seeing data
- $p(Y)$ = normalizing constant (can ignore for MCMC)

### 19.3 The Estimation Loop

For each candidate parameter vector $\theta$:

1. **Solve the model**: Check determinacy, compute $T(\theta)$ and $R(\theta)$
2. **Run Kalman filter**: Compute likelihood $p(Y|\theta)$
3. **Evaluate prior**: Compute $p(\theta)$
4. **Compute posterior**: $p(\theta|Y) \propto p(Y|\theta) \times p(\theta)$

### 19.4 MCMC Sampling

The posterior is high-dimensional and complicated. Use **Markov Chain Monte Carlo** to sample from it.

**Metropolis-Hastings algorithm:**

1. Start at some $\theta^{(0)}$
2. For $j = 1, 2, ..., N$:
   - Propose a new $\theta^* = \theta^{(j-1)} + \text{random step}$
   - Compute acceptance probability based on posterior ratio
   - Accept or reject the proposal
3. The accepted samples approximate the posterior distribution

---

## 20. What the Results Tell Us

### 20.1 Parameter Estimates

Posterior distributions for structural parameters:
- How sticky are prices? (estimate $\theta$)
- How aggressive is monetary policy? (estimate $\phi_\pi$)
- How volatile are different shocks?

### 20.2 Impulse Response Functions

How does the economy respond to each shock over time?
- A monetary policy shock raises interest rates → reduces output and inflation
- A productivity shock raises output → lowers inflation

### 20.3 Variance Decomposition

What fraction of output fluctuations is due to each shock?
- 40% from demand shocks?
- 30% from supply shocks?
- 30% from policy shocks?

### 20.4 Historical Decomposition

What drove the economy in specific episodes?
- Was the 2008 crisis driven by financial shocks or demand shocks?

### 20.5 Forecasting

Given current state and estimated model, what do we expect in the future?

---

## 21. Summary: The Complete Workflow

```
1. SPECIFY PREFERENCES AND TECHNOLOGY
   - Utility function for households
   - Production function for firms
   - Price/wage setting frictions
   ↓
2. SET UP OPTIMIZATION PROBLEMS
   - Household maximizes lifetime utility
   - Firm maximizes profit subject to sticky prices
   ↓
3. DERIVE FIRST-ORDER CONDITIONS
   - Euler equation (consumption/saving choice)
   - Labor supply
   - Pricing equation (New Keynesian Phillips Curve)
   ↓
4. ADD POLICY RULE
   - Taylor Rule for monetary policy
   ↓
5. FIND STEADY STATE
   - Solve for values when economy is stationary
   ↓
6. LOG-LINEARIZE
   - Approximate around steady state
   - Get linear equations in percentage deviations
   ↓
7. WRITE IN MATRIX FORM
   - Γ₀ sₜ = Γ₁ sₜ₋₁ + Ψ εₜ + Π ηₜ
   ↓
8. CLASSIFY VARIABLES
   - Count backward-looking (predetermined)
   - Count forward-looking
   ↓
9. COMPUTE EIGENVALUES
   - Find eigenvalues of system
   - Count stable (|λ|<1) vs unstable (|λ|>1)
   ↓
10. CHECK BLANCHARD-KAHN
    - # unstable eigenvalues = # forward-looking variables?
    - Yes → unique solution (determinacy)
    - No → indeterminacy or no solution
    ↓
11. SOLVE FOR T AND R
    - Eigenvalue decomposition
    - Impose stability
    - Get: sₜ = T sₜ₋₁ + R εₜ
    ↓
12. ADD MEASUREMENT EQUATION
    - yₜ = Z sₜ + D
    - Link latent states to observables
    ↓
13. ESTIMATE PARAMETERS
    - Specify priors
    - Kalman filter for likelihood
    - MCMC for posterior
    ↓
14. ANALYZE RESULTS
    - Impulse responses
    - Variance decomposition
    - Forecasts
```

---

## 22. Key Equations Summary

**Euler Equation (IS Curve):**
$$\hat{y}_t = E_t[\hat{y}_{t+1}] - \frac{1}{\sigma}(\hat{i}_t - E_t[\hat{\pi}_{t+1}])$$

**New Keynesian Phillips Curve:**
$$\hat{\pi}_t = \beta E_t[\hat{\pi}_{t+1}] + \kappa \hat{y}_t$$

**Taylor Rule:**
$$\hat{i}_t = \phi_\pi \hat{\pi}_t + \phi_y \hat{y}_t$$

**Blanchard-Kahn Condition:**
Number of unstable eigenvalues = Number of forward-looking variables

**Taylor Principle (for determinacy):**
$$\phi_\pi > 1$$

**Solution Form:**
$$s_t = T s_{t-1} + R \epsilon_t$$

**Impulse Response at horizon h:**
$$IRF_h = T^h R$$

---

## 23. References

- Galí, J. (2015). *Monetary Policy, Inflation, and the Business Cycle*. Princeton University Press. [Most accessible introduction]
- Blanchard, O., & Kahn, C. (1980). "The Solution of Linear Difference Models under Rational Expectations." *Econometrica*. [The determinacy conditions]
- Sims, C. (2002). "Solving Linear Rational Expectations Models." *Computational Economics*. [The gensys algorithm]
- Smets, F., & Wouters, R. (2007). "Shocks and Frictions in US Business Cycles." *American Economic Review*. [The influential estimated model]

---

*Document prepared January 2026*
