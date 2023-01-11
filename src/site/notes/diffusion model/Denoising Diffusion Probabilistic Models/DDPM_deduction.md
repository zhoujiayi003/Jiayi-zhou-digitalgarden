---
{"dg-publish":true,"permalink":"/diffusion-model/denoising-diffusion-probabilistic-models/ddpm-deduction/","tags":["gardenEntry"]}
---


# Diffusion Process
Define how to add noise in each step, or $\mathbf{x_t}=f(\mathbf{x_{t-1}})$ :
$$q(\mathbf{x}_t|\mathbf{x}_{t-1}):=N(\mathbf{x}_{t};\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf{I})\tag{1}$$
using reparameterization trick, 
$$
\begin{eqnarray}
\mathbf{x}_t&=&\sqrt{1-\beta_t}\mathbf{x}_{t-1}+\sqrt{\beta_t}\epsilon\quad \quad\quad\quad(\epsilon \thicksim N(0,1))\\
&=&\sqrt{\alpha_t}\mathbf{x}_{t-1}+\sqrt{1-\alpha_t}\epsilon \quad\quad\quad\quad(\alpha_t:=1-\beta_t)\\
&=&\sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon)+\sqrt{1-\alpha_t}\epsilon \\
&=&\sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2}+\sqrt{\alpha_t-\alpha_t\alpha_{t-1}}\epsilon+\sqrt{1-\alpha_t}\epsilon \\
&=&\sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2}+\sqrt{1-\alpha_t\alpha_{t-1}}\epsilon\\
&=&......\\
&=&\sqrt{\bar{\alpha_t}}\mathbf{x}_{0}+\sqrt{1-\bar{\alpha_t}}\epsilon \quad\quad\quad\quad\quad (\bar{\alpha_t}:=\sideset{}{}\prod_{i=0}^t\alpha_i)\tag{2}
\end{eqnarray}
$$
In other words,
$$q(\mathbf{x}_t|\mathbf{x}_{0}):=N(\mathbf{x}_{0};\sqrt{\bar{\alpha_t}}\mathbf{x}_{0},(1-\bar{\alpha_t})\mathbf{I})\tag{3}$$

# Reverse Process
$$
\begin{eqnarray}
q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})&=&\frac{q(\mathbf{x}_{t-1},\mathbf{x}_{t}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})}\\
&=&\frac{q(\mathbf{x}_{t}|\mathbf{x}_{t-1},\mathbf{x}_{0})q(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})}\\
&=&\frac{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})q(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})} \quad\quad\tag{4}
\end{eqnarray}
$$
The last step above is based on the property of Markov process $q(\mathbf{x}_{t}|\mathbf{x}_{t-1},\mathbf{x}_{0})=q(\mathbf{x}_{t}|\mathbf{x}_{t-1})$
From Eq(2) and Eq(3) we have:



# Loss Function
