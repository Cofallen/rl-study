---
title: 'D.1.1 Linear Algebra Basics: Vectors and Matrices'
---

# D.1.1 Vectors and Matrices

If you opened this appendix, it is probably because a formula in the main text slowed you down. Maybe it was an expectation symbol inside a Bellman equation, a KL divergence term inside PPO, or the gradient operator that suddenly appears in the policy gradient theorem.

These formulas can look intimidating, but the underlying tools are not that many: scalars, vectors, matrices, probability, expectation, derivatives, gradients, entropy, and KL divergence. Once you understand what each word means, and how they combine inside reinforcement learning formulas, the notation becomes far less mysterious.

This appendix does not organize math by algorithm. It follows a more natural learning order:

mathematical objects -> linear operations -> probability and expectation -> stochastic estimation -> recursive equations -> optimization and gradients -> distribution distance -> full RL formulas.

## Learning Route

The whole appendix can be summarized as one line:

**mathematical objects -> linear algebra -> probability and expectation -> stochastic estimation -> Bellman recursion -> optimization and gradients -> distribution distance -> RL derivations.**

## Appendix Map

| Section                                                                             | Topic                                                                                   | Main question                                                         |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [D.1 Mathematical objects and linear algebra](./linear-algebra)                     | scalars, vectors, matrices, dot products, norms, linear equations                       | How do we write states, values, and parameters as computable objects? |
| [D.2 Probability, expectation, and stochastic estimation](./probability-statistics) | probability, conditional probability, random variables, expectation, variance, sampling | How do random trajectories become average value?                      |
| [D.3 Calculus and optimization](./calculus-optimization)                            | derivatives, gradients, chain rule, Taylor expansion, optimization algorithms           | Which direction should parameters move?                               |
| [D.4 Information theory and distribution distance](./information-theory)            | self-information, entropy, cross-entropy, KL, mutual information                        | How do we measure policy randomness and policy change?                |

## Suggested Reading Order

If you want to rebuild the mathematical foundations systematically, read in this order:

1. [D.1.1 Vector and matrix basics](./linear-algebra-basics): understand scalars, vectors, matrices, and matrix multiplication.
2. [D.1.2 Bellman equations in matrix form](./linear-algebra-bellman): see how value recursion becomes a linear system.
3. [D.2.1 Probability, conditional probability, and expectation](./probability-basics): learn random variables and expectation.
4. [D.2.2 Random trajectories and state value](./probability-value): connect expectation to returns and value functions.
5. [D.2.3 Monte Carlo and importance sampling](./probability-sampling): estimate values from samples when the environment model is unknown.
6. [D.3.1 Derivatives, gradients, and the chain rule](./calculus-basics): understand parameter changes and backpropagation.
7. [D.3.2 Policy gradients and advantage functions](./calculus-policy-gradient): apply gradients to policy optimization.
8. [D.4.1 Self-information, entropy, and exploration](./information-basics): understand policy randomness.
9. [D.4.2 Cross-entropy and KL divergence](./information-cross-entropy-kl): understand distances between distributions.
10. Finally, return to each module's formula summary and exercises.

If you only need one concept, it is completely fine to jump directly to the relevant page.

## A Running Example

Several sections reuse the same tiny two-state environment. There are two states, $s_1$ and $s_2$:

- in $s_1$, the agent receives reward $2$, then moves to $s_2$
- in $s_2$, the agent receives reward $1$, then moves back to $s_1$
- the discount factor is $\gamma = 0.5$

Let the state values be $v_1$ and $v_2$. Intuitively:

$$
\begin{aligned}
v_1 &= 2 + 0.5v_2, \\
v_2 &= 1 + 0.5v_1.
\end{aligned}
$$

This same example plays different roles in different modules:

- in linear algebra, it is a two-variable linear system
- in probability, it is "immediate reward + expected next-state value"
- in stochastic estimation, it can be approximated from sampled trajectories
- in optimization, it becomes a target for a value network or policy network
- in information theory, it connects to policy distributions, exploration, and update constraints

If you can translate a complicated formula back into this two-state example, math stops being a wall and becomes a tool.

## How to Use This Appendix

The sidebar divides the content into four math modules. You do not need to read everything at once. There are three useful modes:

1. **Systematic catch-up**: start from D.1.1 and read in order.
2. **Just-in-time lookup**: if Bellman matrix form is confusing, read D.1.2; if GAE is confusing, read the probability and calculus sections; if KL constraints are confusing, read D.4.2.
3. **Quick review**: use each module's formula summary and exercises after finishing the corresponding topic.

If the sidebar feels too large, read only the first page of each module first:

- [D.1.1 Scalars, vectors, and matrices](./linear-algebra-basics)
- [D.2.1 Probability, conditional probability, and expectation](./probability-basics)
- [D.3.1 Derivatives, gradients, and the chain rule](./calculus-basics)
- [D.4.1 Self-information, entropy, and exploration](./information-basics)

After these four pages, return to the detailed topics whenever a formula in the main text needs support.

<!--@include: ./linear-algebra.md{7,}-->

> **Prerequisites**: This article does not require prior linear algebra. Read the two-state running example above first.

---

## Scalars, Sets, and Functions

A **scalar** is a single number. A reward $r=2$ is a scalar, and a discount factor $\gamma=0.9$ is also a scalar. In RL, the following values are all scalars:

| Symbol  | Meaning                            | Typical values |
| ------- | ---------------------------------- | -------------- |
| r       | Immediate reward                   | 2, -1, 0.5     |
| gamma   | Discount factor                    | 0.9, 0.99, 0.5 |
| alpha   | Learning rate                      | 0.001, 3e-4    |
| epsilon | Exploration rate or clipping range | 0.1, 0.2       |

A scalar gives the numerical value of a reward. Another basic element of an environment is the state. The collection of all possible states forms a **set**.

A **set** lists all possible elements inside braces. For example, a small environment may have three states and two actions:

$$
\mathcal{S}=\{s_1,s_2,s_3\}, \qquad \mathcal{A}=\{\text{left},\text{right}\}.
$$

The calligraphic letters $\mathcal{S}$ and $\mathcal{A}$ are conventional; writing $S$ and $A$ would not change the meaning. When we discuss "the value under state $s$," that $s$ must come from some set $\mathcal{S}$.

A set defines the available states. Next, we need to assign a value to each state, which leads to the idea of a **function**.

A value function takes a state and returns a number:

$$
v:\mathcal{S}\to\mathbb{R}, \qquad s\mapsto v(s).
$$

The notation means:

- $v:\mathcal{S}\to\mathbb{R}$ says that $v$ is a function from $\mathcal{S}$ to $\mathbb{R}$. The colon gives the type declaration, and the arrow gives the domain and codomain. $\mathbb{R}$ is the set of real numbers.
- $s\mapsto v(s)$ says that input $s$ is mapped to output $v(s)$.

In plain language, given a state $s$, the function returns a number $v(s)$. For example, $v(s_1)=3$ means that the value of state $s_1$ is $3$.

A policy function is similar. Its input is a state-action pair, and its output is a probability:

$$
\pi:\mathcal{S}\times\mathcal{A}\to[0,1], \qquad (s,a)\mapsto\pi(a\mid s).
$$

Here:

- The $\times$ in $\mathcal{S}\times\mathcal{A}$ denotes the Cartesian product: the set of all state-action pairs.
- $[0,1]$ means the output range is the real interval from $0$ to $1$, because probabilities live in this range.
- $(s,a)\mapsto\pi(a\mid s)$ means the input is a state-action pair and the output is the probability of choosing action $a$ in state $s$.

A function requires the same input to correspond to only one output. $v(s)$ gives one value for each state, so it satisfies this requirement. The transition probability $p(s'\mid s,a)$ is also a function: the input is the current state, action, and next state, and the output is the transition probability.

Scalars, sets, and functions handle one-to-one relationships: one state corresponds to one value, and one state-action pair corresponds to one probability. But reinforcement learning often needs to handle many states at once. If an environment has thousands of states, writing every $v(s_i)$ separately becomes cumbersome. Putting all state values into one column and operating on them as a whole gives us a **vector**.

---

## Vectors

Suppose an environment has three states and the current value estimates are:

| State | Value |
| ----- | ----- |
| s1    | 3     |
| s2    | 5     |
| s3    | 2     |

Put all numbers into one column and treat them as a single object:

$$
\boldsymbol{v} =
\begin{bmatrix}
3 \\
5 \\
2
\end{bmatrix}.
$$

The three numbers inside the brackets are the **components** of the vector. $\boldsymbol{v}$ is written in bold to distinguish it from a single scalar. Once vectors are introduced, we can operate on "the values of all states" at the same time.

**Addition.** Suppose every state receives an additional reward of $1$. This is equivalent to adding $1$ to every component:

$$
\boldsymbol{v}_{new} =
\begin{bmatrix}
3 \\
5 \\
2
\end{bmatrix}
+
\begin{bmatrix}
1 \\
1 \\
1
\end{bmatrix}
=
\begin{bmatrix}
4 \\
6 \\
3
\end{bmatrix}.
$$

Vector addition adds components one by one. This requires the two vectors to have the same length. Vectors with different lengths cannot be added.

**Scalar multiplication.** Multiply every component of a vector by the same scalar. For example, applying discount factor $\gamma=0.5$ gives:

$$
\gamma\boldsymbol{v} = 0.5 \times
\begin{bmatrix}
3 \\
5 \\
2
\end{bmatrix}
=
\begin{bmatrix}
1.5 \\
2.5 \\
1.0
\end{bmatrix}.
$$

This corresponds to discounting future value. In the Bellman equation $v = r + \gamma P v$, $\gamma v$ is this step: scale the future value by the discount factor before adding it to the immediate reward.

A vector can represent the values of all states, but it cannot represent how states transition into one another. From $s_1$, the next state may be $s_2$ or $s_3$, and this "from which state to which state, with what probability" relationship requires a **matrix**.

---

## Matrices

Consider two states $s_1$ and $s_2$:

- Starting from $s_1$, the next state is always $s_2$.
- Starting from $s_2$, the next state is always $s_1$.

This transition relationship can be written as a matrix:

$$
P =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}.
$$

Rows correspond to "which state we start from," and columns correspond to "which state we go to next." The first row $[0, 1]$ says: starting from $s_1$, the probability of reaching $s_1$ is $0$, and the probability of reaching $s_2$ is $1$. The second row has the analogous meaning.

If the current values of the two states are:

$$
\boldsymbol{v} =
\begin{bmatrix}
3.33 \\
2.67
\end{bmatrix},
$$

then the first component of $\boldsymbol{v}$ is the value of $s_1$, and the second component is the value of $s_2$. Each row of the transition matrix is a probability distribution over next states. Multiplying a row by $\boldsymbol{v}$ takes a probability-weighted sum of possible next-state values:

$$
\text{next-state value from }s_1
= 0 \times v(s_1) + 1 \times v(s_2) = 2.67,
$$

$$
\text{next-state value from }s_2
= 1 \times v(s_1) + 0 \times v(s_2) = 3.33.
$$

Putting the two row results back into a vector gives $P\boldsymbol{v}$: the expected next-state value from each current state.

$$
P\boldsymbol{v} =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
3.33 \\
2.67
\end{bmatrix}
=
\begin{bmatrix}
2.67 \\
3.33
\end{bmatrix}.
$$

The result matches the transition rule: from $s_1$ the next state is $s_2$, whose future value is $2.67$; from $s_2$ the next state is $s_1$, whose future value is $3.33$.

### General Case

For three states, suppose the transition relationship is:

| Current state | to s1 | to s2 | to s3 |
| ------------- | ----- | ----- | ----- |
| s1            | 0.1   | 0.7   | 0.2   |
| s2            | 0.0   | 0.3   | 0.7   |
| s3            | 0.5   | 0.5   | 0.0   |

Written as a matrix:

$$
P =
\begin{bmatrix}
0.1 & 0.7 & 0.2 \\
0.0 & 0.3 & 0.7 \\
0.5 & 0.5 & 0.0
\end{bmatrix}.
$$

The number of rows and columns both equal the number of states $n$. Moving from 2 states to 3 states changes the matrix from $2\times2$ to $3\times3$, but the structure is unchanged: row $i$ always represents the probabilities of going from $s_i$ to every possible next state. This structure holds for any number of states.

---

## Matrix Multiplication and Probability Weighting

So far we have placed state values into a vector and state transitions into a matrix. Now let us examine the computation inside matrix multiplication and why it exactly matches probability-weighted averaging.

Let

$$
A=
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix},
\qquad
x=
\begin{bmatrix}
10 \\
20
\end{bmatrix}.
$$

Then

$$
Ax=
\begin{bmatrix}
1\times10+2\times20 \\
3\times10+4\times20
\end{bmatrix}
=
\begin{bmatrix}
50 \\
110
\end{bmatrix}.
$$

Each row of the matrix takes one dot product with the vector. Matrix-vector multiplication is many weighted sums performed together.

### Probability Weighting as a Special Case

In reinforcement learning, when transition matrix $P$ multiplies value vector $v$, row $i$ says: starting from state $s_i$, average the values of all possible next states using transition probabilities as weights.

A concrete example:

$$
P=
\begin{bmatrix}
0.7 & 0.3 \\
0.2 & 0.8
\end{bmatrix},
\qquad
v=
\begin{bmatrix}
10 \\
5
\end{bmatrix}.
$$

Then

$$
Pv=
\begin{bmatrix}
0.7\times10+0.3\times5 \\
0.2\times10+0.8\times5
\end{bmatrix}
=
\begin{bmatrix}
8.5 \\
6
\end{bmatrix}.
$$

The first row $0.7\times10+0.3\times5=8.5$ means: starting from $s_1$, there is a $70\%$ chance of reaching a state with value $10$ and a $30\%$ chance of reaching a state with value $5$, so the expected future value is $8.5$.

The key property here is that every row of the matrix is a set of probabilities whose sum is $1$. Therefore, matrix multiplication implements exactly "probability times value" weighted averaging. The Bellman equation $v = r + \gamma Pv$ is essentially "immediate reward plus discounted probability-weighted future value."

Matrix multiplication is not limited to probability matrices. In neural networks, rows of weight matrices usually do not sum to $1$, but matrix multiplication is still weighted summation. Probability weighting is only one special case of matrix multiplication.

---

## Dimension Checks

The simplest way to judge whether a linear algebra formula is valid is to check dimensions.

With $n$ states, the value vector is:

$$
v\in\mathbb{R}^n.
$$

The state transition matrix is:

$$
P\in\mathbb{R}^{n\times n}.
$$

Therefore the shape of $Pv$ is:

$$
(n\times n)(n\times1)=n\times1.
$$

The result is still a value vector. Thus

$$
v=r+\gamma Pv
$$

has matching shapes on both sides and is meaningful.

### Shape Checks in Neural Networks

If a linear Q function is written as

$$
Q(s,a)=w^\top\phi(s,a),
$$

then $w$ and $\phi(s,a)$ must have the same length. If $w\in\mathbb{R}^d$, then $\phi(s,a)\in\mathbb{R}^d$, so the dot product is a scalar.

Shape checking is just as important in neural networks. Consider a simple two-layer network:

```text
input state features, 128 dimensions -> hidden layer, 64 dimensions -> action logits, 2 dimensions
```

The weight matrix shapes are:

| Layer   | Weight matrix | Shape    |
| ------- | ------------- | -------- |
| Layer 1 | W1            | 128 x 64 |
| Layer 2 | W2            | 64 x 2   |

Forward propagation:

$$
h = \sigma(W_1^\top x), \qquad z = W_2^\top h.
$$

$W_1$ is $128\times64$, input $x$ is $128\times1$, so $W_1^\top x$ is $64\times1$, and hidden state $h$ is also $64\times1$. $W_2$ is $64\times2$, so $W_2^\top h$ is $2\times1$, exactly two action logits.

Dimension checking is an effective way to read papers and write code. Many formulas look complicated, but checking input and output dimensions often reveals whether they are reasonable.

::: warning Common Pitfall
In matrix multiplication, $(A \times B)(B \times C) = A \times C$, so the inner dimensions must match. If code raises `RuntimeError: mat1 and mat2 shapes cannot be multiplied`, some tensor dimension is usually wrong.
:::

---

## Summary

This article established five basic objects in linear algebra:

| Object   | Role in RL                         | Example           |
| -------- | ---------------------------------- | ----------------- |
| Scalar   | Single reward or hyperparameter    | r=2, gamma=0.9    |
| Set      | Possible states and actions        | $S=\{s_1, s_2\}$  |
| Function | Value function and policy function | v(s), pi(a\|s)    |
| Vector   | All state values together          | **v**=[3, 5, 2]^T |
| Matrix   | Transitions among all states       | P in R^(n x n)    |

Their relationships are: scalars form vectors, vectors form matrices, and matrix-vector multiplication implements probability weighting. The next article combines these objects into a full system of equations: the matrix form of the Bellman equation $v = r + \gamma Pv$.

> **Next**: [D.1.2 Matrix Form of the Bellman Equation](./linear-algebra-bellman) shows how vectors, matrices, and matrix multiplication combine into the matrix form of the Bellman equation.
