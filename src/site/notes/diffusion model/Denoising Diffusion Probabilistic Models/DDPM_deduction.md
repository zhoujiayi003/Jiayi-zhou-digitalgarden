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
Using Bayes' rule, we have:
$$
\begin{eqnarray}
q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})&=&\frac{q(\mathbf{x}_{t-1},\mathbf{x}_{t}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})}\\
&=&\frac{q(\mathbf{x}_{t}|\mathbf{x}_{t-1},\mathbf{x}_{0})q(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})}\\
&=&\frac{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})q(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})} \quad\quad\tag{4}
\end{eqnarray}
$$
The last step above is based on the property of Markov process $q(\mathbf{x}_{t}|\mathbf{x}_{t-1},\mathbf{x}_{0})=q(\mathbf{x}_{t}|\mathbf{x}_{t-1})$
From Eq(2) and Eq(3) we have:
$$
\begin{eqnarray}
q(\mathbf{x}_{t}|\mathbf{x}_{t-1})&=&\sqrt{\alpha_t}\mathbf{x}_{t-1}+\sqrt{1-\alpha_t}\epsilon \thicksim N(\mathbf{x}_{t};\sqrt{\alpha_t}\mathbf{x}_{t-1},(1-\alpha_t)\mathbf{I})\\
q(\mathbf{x}_{t-1}|\mathbf{x}_{0})&=&\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon \thicksim N(\mathbf{x}_{t-1};\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0},(1-\bar{\alpha}_{t-1})\mathbf{I})\\
q(\mathbf{x}_{t}|\mathbf{x}_{0})&=&\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}}\epsilon \thicksim N(\mathbf{x}_{t};\sqrt{\bar{\alpha_t}}\mathbf{x}_{0},(1-\bar{\alpha_t})\mathbf{I})
\end{eqnarray}
$$
so
$$
\begin{eqnarray}
q(\mathbf{x}_{t}|\mathbf{x}_{t-1})&=&\int\frac{1}{\sqrt{2\pi(1-\alpha_t)}}e^{-\frac{(\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{2(1-\alpha_t)}}d\mathbf{x}_t\\
q(\mathbf{x}_{t-1}|\mathbf{x}_{0})&=&\int\frac{1}{\sqrt{2\pi(1-\bar{\alpha}_{t-1})}}e^{-\frac{(\mathbf{x}_{t-1}-\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0})^2}{2(1-\bar{\alpha}_{t-1})}}d\mathbf{x}_{t-1}\\
q(\mathbf{x}_{t}|\mathbf{x}_{0})&=&\int\frac{1}{\sqrt{2\pi(1-\bar{\alpha}_{t})}}e^{-\frac{(\mathbf{x}_{t}-\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0})^2}{2(1-\bar{\alpha}_{t})}}d\mathbf{x}_{t}
\end{eqnarray}
$$
Back to Eq(4):
$$
\begin{eqnarray}
q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})
&=&\int f(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})d\mathbf{x}_{t-1}=\int \frac{f(\mathbf{x}_{t}|\mathbf{x}_{t-1})f(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{f(\mathbf{x}_{t}|\mathbf{x}_{0})}d\mathbf{x}_{t-1} \\
&=&\int \frac{(\frac{1}{\sqrt{2\pi(1-\alpha_t)}}e^{-\frac{(\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{2(1-\alpha_t)}})(\frac{1}{\sqrt{2\pi(1-\bar{\alpha}_{t-1})}}e^{-\frac{(\mathbf{x}_{t-1}-\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0})^2}{2(1-\bar{\alpha}_{t-1})}})}{\frac{1}{\sqrt{2\pi(1-\bar{\alpha}_{t})}}e^{-\frac{(\mathbf{x}_{t}-\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0})^2}{2(1-\bar{\alpha}_{t})}}}d\mathbf{x}_{t-1}\\
&=&\int \frac{\frac{1}{\sqrt{2\pi(1-\alpha_t)}}\frac{1}{\sqrt{2\pi(1-\bar{\alpha}_{t-1})}}}{\frac{1}{\sqrt{2\pi(1-\bar{\alpha}_{t})}}}\frac{(e^{-\frac{(\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{2(1-\alpha_t)}})(e^{-\frac{(\mathbf{x}_{t-1}-\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0})^2}{2(1-\bar{\alpha}_{t-1})}})}{e^{-\frac{(\mathbf{x}_{t}-\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0})^2}{2(1-\bar{\alpha}_{t})}}}d\mathbf{x}_{t-1}\\
&=&\frac{1}{\sqrt{2\pi \frac{(1-\alpha_{t})(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_{t})}}}\int e^{-\frac{(\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{2(1-\alpha_t)}-\frac{(\mathbf{x}_{t-1}-\sqrt{\bar\alpha_{t-1}}\mathbf{x}_{0})^2}{2(1-\bar\alpha_{t-1})}+\frac{(\mathbf{x}_{t}-\sqrt{\bar\alpha_{t}}\mathbf{x}_{0})^2}{2(1-\bar\alpha_{t})}}d\mathbf{x}_{t-1}\tag{5}\\

\end{eqnarray}
$$
Use $I$ to denote integrand function in Eq(5), 
$$
\begin{eqnarray}
I:=&exp&[-\frac{(\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{2(1-\alpha_t)}-\frac{(\mathbf{x}_{t-1}-\sqrt{\bar\alpha_{t-1}}\mathbf{x}_{0})^2}{2(1-\bar\alpha_{t-1})}+\frac{(\mathbf{x}_{t}-\sqrt{\bar\alpha_{t}}\mathbf{x}_{0})^2}{2(1-\bar\alpha_{t})}]\\
=&exp&[-\frac{1}{2}[\frac{\mathbf{x}_t^2+\alpha_t\mathbf{x}^2_{t-1}-2\sqrt{\alpha_t}\mathbf{x}_t\mathbf{x}_{t-1}}{1-\alpha_t}+
\frac{\mathbf{x}_{t-1}^2+\bar{\alpha}_{t-1}\mathbf{x}^2_{0}-2\sqrt{\bar{\alpha}_t}\mathbf{x}_{t-1}\mathbf{x}_{0}}{1-\bar\alpha_{t-1}}\\
&&-
\frac{\mathbf{x}_t^2+\bar\alpha_t\mathbf{x}^2_{0}-2\sqrt{\bar\alpha_t}\mathbf{x}_t\mathbf{x}_{0}}{1-\bar\alpha_{t}}]]\\
=&exp&[
-\frac{1}{2}[
(\frac{\alpha_t}{1-\alpha_t}+\frac{1}{1-\bar\alpha_{t-1}})\mathbf{x}^2_{t-1}-
(\frac{2\sqrt{\alpha_t}}{1-\alpha_t}\mathbf{x}_t+\frac{2\sqrt{\bar\alpha_t}}{1-\bar\alpha_{t-1}}\mathbf{x}_0)\mathbf{x}_{t-1}+C
]]\\
=&exp&[
-\frac{1}{2}[
\frac{1-\alpha_t\bar\alpha_{t-1}}{(1-\alpha_t)(1-\bar\alpha_{t-1})}\mathbf{x}^2_{t-1}-
2(\frac{\sqrt{\alpha_t}}{1-\alpha_t}\mathbf{x}_t+\frac{\sqrt{\bar\alpha_t}}{1-\bar\alpha_{t-1}}\mathbf{x}_0)\mathbf{x}_{t-1}+C
]]\\
\xlongequal{\bar\alpha_t=\alpha_t\bar\alpha_{t-1}}&exp&
[
-\frac{1}{2}[
\frac{1-\bar\alpha_{t}}{(1-\alpha_t)(1-\bar\alpha_{t-1})}\mathbf{x}^2_{t-1}-
2(\frac{\sqrt{\alpha_t}}{1-\alpha_t}\mathbf{x}_t+\frac{\sqrt{\bar\alpha_t}}{1-\bar\alpha_{t-1}}\mathbf{x}_0)\mathbf{x}_{t-1}+C
]]\\
\xlongequal{\widetilde\beta_t:=\frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_{t}}}&exp&
[
-\frac{1}{2\widetilde\beta_t}[
\mathbf{x}^2_{t-1}-
2\frac{\frac{\sqrt{\alpha_t}\mathbf{x}_t}{1-\alpha_t}+\frac{\sqrt{\bar\alpha_t}\mathbf{x}_0}{1-\bar\alpha_{t-1}}}{\frac{1-\bar\alpha_{t}}{(1-\alpha_t)(1-\bar\alpha_{t-1})}
}\mathbf{x}_{t-1}+C\widetilde\beta_t
]]\\
=&exp&
[
-\frac{1}{2\widetilde\beta_t}[
\mathbf{x}^2_{t-1}-
2\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1-\alpha_t)\sqrt{\bar\alpha_t}\mathbf{x}_0}{1-\bar\alpha_t
}\mathbf{x}_{t-1}+C\widetilde\beta_t
]]\tag{6}\\

\end{eqnarray}
$$
where
$$
\begin{eqnarray}
C\widetilde\beta_t&=&(\frac{\mathbf{x}_t^2}{1-\alpha_t}+
\frac{\bar\alpha_{t-1}\mathbf{x}_0^2}{1-\bar\alpha_{t-1}}-\frac{
\mathbf{x}_t^2+\bar\alpha_t\mathbf{x}^2_{0}-2\sqrt{\bar\alpha_t}\mathbf{x}_t\mathbf{x}_{0}}{1-\bar\alpha_t})\frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_{t}}\\
&=&\frac{(1-\bar\alpha_{t-1})\mathbf{x_t^2}}{1-\bar\alpha_t}+
\frac{(1-\alpha_t)\bar\alpha_{t-1}\mathbf{x}_0^2}{1-\bar\alpha_t}-
\frac{(\mathbf{x}_t^2+\bar\alpha_t\mathbf{x}^2_{0}-2\sqrt{\bar\alpha_t}\mathbf{x}_t\mathbf{x}_{0})(1-\alpha_t)(1-\bar\alpha_{t-1})}{(1-\bar\alpha_t)^2}\\
&=&\frac{(1-\bar\alpha_t)(1-\bar\alpha_{t-1})\mathbf{x_t^2}+
(1-\bar\alpha_t)(1-\alpha_t)\bar\alpha_{t-1}\mathbf{x}_0^2-
(\mathbf{x}_t^2+\bar\alpha_t\mathbf{x}^2_{0}-2\sqrt{\bar\alpha_t}\mathbf{x}_t\mathbf{x}_{0})(1-\alpha_t)(1-\bar\alpha_{t-1})(1-\bar\alpha_t)
}
{(1-\bar\alpha_t)^2}\\
&=&\frac{
(\alpha_t-\bar\alpha_t)(1-\bar\alpha_{t-1})\mathbf{x_t^2}+
(1-\alpha_t)(\bar\alpha_{t-1}-\bar\alpha_t)\mathbf{x}_0^2+
2\sqrt{\bar\alpha_t}(1-\alpha_t)(1-\bar\alpha_{t-1})\mathbf{x_t}\mathbf{x_0}
}
{(1-\bar\alpha_t)^2}\\
&=&(\frac{\sqrt{(\alpha_t-\bar\alpha_t)(1-\bar\alpha_{t-1})}\mathbf{x_t+\sqrt{(1-\alpha_t)(\bar\alpha_{t-1}-\bar\alpha_t)}\mathbf{x_0}}}{1-\bar\alpha_t})^2
\end{eqnarray}
$$
Recall Eq(5), Eq(6),
$$
\begin{eqnarray}
q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})
&=&\frac{1}{\sqrt{2\pi \frac{(1-\alpha_{t})(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_{t})}}}\int e^{-\frac{(\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{2(1-\alpha_t)}-\frac{(\mathbf{x}_{t-1}-\sqrt{\bar\alpha_{t-1}}\mathbf{x}_{0})^2}{2(1-\bar\alpha_{t-1})}+\frac{(\mathbf{x}_{t}-\sqrt{\bar\alpha_{t}}\mathbf{x}_{0})^2}{2(1-\bar\alpha_{t})}}d\mathbf{x}_{t-1}\\
&=&\frac{1}{\sqrt{2\pi\widetilde{\beta}_t}}
\int e^{
[
-\frac{1}{2\widetilde\beta_t}[
\mathbf{x}^2_{t-1}-
2\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1-\alpha_t)\sqrt{\bar\alpha_t}\mathbf{x}_0}{1-\bar\alpha_t
}\mathbf{x}_{t-1}+C\widetilde\beta_t
]]}
d\mathbf{x}_{t-1}\\
&=&\frac{1}{\sqrt{2\pi\widetilde{\beta}_t}}
\int e^{

-\frac{(
\mathbf{x}_{t-1}-
\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1-\alpha_t)\sqrt{\bar\alpha_t}\mathbf{x}_0}{1-\bar\alpha_t
}
)^2}{2\widetilde\beta_t}}
d\mathbf{x}_{t-1}\\

\end{eqnarray}
$$
Define 
$$\widetilde\mu_t:=\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1-\alpha_t)\sqrt{\bar\alpha_t}\mathbf{x}_0}{1-\bar\alpha_t
}$$
Finally we have the following Gaussion distribution:
$$
\begin{eqnarray}
q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})
&=&\frac{1}{\sqrt{2\pi\widetilde{\beta}_t}}
\int e^{
-\frac{(
\mathbf{x}_{t-1}-
\widetilde\mu_t)^2}{2\widetilde\beta_t}}
d\mathbf{x}_{t-1}\\
\end{eqnarray}
$$
or
$$q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})=N(\mathbf{x}_{t-1};\widetilde{\mu}_t,\widetilde{\beta}_tI)$$
$$
\mathbf{x}_{t-1}=\widetilde{\mu}_t+\sqrt{\widetilde{\beta}_t}\epsilon
$$


# Loss Function
