---
{"dg-publish":true,"permalink":"/diffusion-model/denoising-diffusion-probabilistic-models/ddpm-deduction/","tags":["gardenEntry"]}
---



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
{ #4610ab}


In other words,
$$q(\mathbf{x}_t|\mathbf{x}_{0}):=N(\mathbf{x}_{0};\sqrt{\bar{\alpha_t}}\mathbf{x}_{0},(1-\bar{\alpha_t})\mathbf{I})\tag{3}$$
{ #49f4b9}


# Reverse Process
Using Bayes' rule, we have:
$$
\begin{eqnarray}
q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})&=&\frac{q(\mathbf{x}_{t-1},\mathbf{x}_{t}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})}\\
&=&\frac{q(\mathbf{x}_{t}|\mathbf{x}_{t-1},\mathbf{x}_{0})q(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})}\\
&=&\frac{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})q(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})} \quad\quad\tag{4}
\end{eqnarray}
$$
{ #a3f8ba}


The last step above is based on the property of Markov process $q(\mathbf{x}_{t}|\mathbf{x}_{t-1},\mathbf{x}_{0})=q(\mathbf{x}_{t}|\mathbf{x}_{t-1})$
From [[diffusion model/Denoising Diffusion Probabilistic Models/DDPM_deduction#^4610ab\|Eq[2]]] and [[diffusion model/Denoising Diffusion Probabilistic Models/DDPM_deduction#^49f4b9\|Eq(3)]] we have:
$$
\begin{eqnarray}
q(\mathbf{x}_{t}|\mathbf{x}_{t-1})&=&  N(\mathbf{x}_{t};\sqrt{\alpha_t}\mathbf{x}_{t-1},(1-\alpha_t)\mathbf{I})\\
q(\mathbf{x}_{t-1}|\mathbf{x}_{0})&=& N(\mathbf{x}_{t-1};\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0},(1-\bar{\alpha}_{t-1})\mathbf{I})\\
q(\mathbf{x}_{t}|\mathbf{x}_{0})&=&N(\mathbf{x}_{t};\sqrt{\bar{\alpha_t}}\mathbf{x}_{0},(1-\bar{\alpha_t})\mathbf{I})
\end{eqnarray}
$$
or
$$
\begin{eqnarray}
\mathbf{x}_t&=&\sqrt{\alpha_t}\mathbf{x}_{t-1}+\sqrt{1-\alpha_t}\epsilon\\
\mathbf{x}_{t-1}&=&\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon\\
\mathbf{x}_t&=&\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}}\epsilon 
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
Back to [[diffusion model/Denoising Diffusion Probabilistic Models/DDPM_deduction#^a3f8ba\|Eq(4)]]:
$$
\begin{eqnarray}
q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})
&=&\int f(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})d\mathbf{x}_{t-1}=\int \frac{f(\mathbf{x}_{t}|\mathbf{x}_{t-1})f(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{f(\mathbf{x}_{t}|\mathbf{x}_{0})}d\mathbf{x}_{t-1} \\
&=&\int \frac{(\frac{1}{\sqrt{2\pi(1-\alpha_t)}}e^{-\frac{(\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{2(1-\alpha_t)}})(\frac{1}{\sqrt{2\pi(1-\bar{\alpha}_{t-1})}}e^{-\frac{(\mathbf{x}_{t-1}-\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0})^2}{2(1-\bar{\alpha}_{t-1})}})}{\frac{1}{\sqrt{2\pi(1-\bar{\alpha}_{t})}}e^{-\frac{(\mathbf{x}_{t}-\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0})^2}{2(1-\bar{\alpha}_{t})}}}d\mathbf{x}_{t-1}\\
&=&\int \frac{\frac{1}{\sqrt{2\pi(1-\alpha_t)}}\frac{1}{\sqrt{2\pi(1-\bar{\alpha}_{t-1})}}}{\frac{1}{\sqrt{2\pi(1-\bar{\alpha}_{t})}}}\frac{(e^{-\frac{(\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{2(1-\alpha_t)}})(e^{-\frac{(\mathbf{x}_{t-1}-\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0})^2}{2(1-\bar{\alpha}_{t-1})}})}{e^{-\frac{(\mathbf{x}_{t}-\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0})^2}{2(1-\bar{\alpha}_{t})}}}d\mathbf{x}_{t-1}\\
&=&\frac{1}{\sqrt{2\pi \frac{(1-\alpha_{t})(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_{t})}}}\int e^{-\frac{(\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{2(1-\alpha_t)}-\frac{(\mathbf{x}_{t-1}-\sqrt{\bar\alpha_{t-1}}\mathbf{x}_{0})^2}{2(1-\bar\alpha_{t-1})}+\frac{(\mathbf{x}_{t}-\sqrt{\bar\alpha_{t}}\mathbf{x}_{0})^2}{2(1-\bar\alpha_{t})}}d\mathbf{x}_{t-1}\tag{5}\\

\end{eqnarray}
$$
{ #c9714c}


Use $I$ to denote integrand function, 
$$
\begin{eqnarray}
I:=&exp&[-\frac{(\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{2(1-\alpha_t)}-\frac{(\mathbf{x}_{t-1}-\sqrt{\bar\alpha_{t-1}}\mathbf{x}_{0})^2}{2(1-\bar\alpha_{t-1})}+\frac{(\mathbf{x}_{t}-\sqrt{\bar\alpha_{t}}\mathbf{x}_{0})^2}{2(1-\bar\alpha_{t})}]\\
=&exp&[-\frac{1}{2}[\frac{\mathbf{x}_t^2+\alpha_t\mathbf{x}^2_{t-1}-2\sqrt{\alpha_t}\mathbf{x}_t\mathbf{x}_{t-1}}{1-\alpha_t}+
\frac{\mathbf{x}_{t-1}^2+\bar{\alpha}_{t-1}\mathbf{x}^2_{0}-2\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{t-1}\mathbf{x}_{0}}{1-\bar\alpha_{t-1}}
-
\frac{\mathbf{x}_t^2+\bar\alpha_t\mathbf{x}^2_{0}-2\sqrt{\bar\alpha_t}\mathbf{x}_t\mathbf{x}_{0}}{1-\bar\alpha_{t}}]]\\
=&exp&[
-\frac{1}{2}[
(\frac{\alpha_t}{1-\alpha_t}+\frac{1}{1-\bar\alpha_{t-1}})\mathbf{x}^2_{t-1}-
(\frac{2\sqrt{\alpha_t}}{1-\alpha_t}\mathbf{x}_t+\frac{2\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\mathbf{x}_0)\mathbf{x}_{t-1}+C
]]\\
=&exp&[
-\frac{1}{2}[
\frac{1-\alpha_t\bar\alpha_{t-1}}{(1-\alpha_t)(1-\bar\alpha_{t-1})}\mathbf{x}^2_{t-1}-
2(\frac{\sqrt{\alpha_t}}{1-\alpha_t}\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\mathbf{x}_0)\mathbf{x}_{t-1}+C
]]\\
\xlongequal{\bar\alpha_t=\alpha_t\bar\alpha_{t-1}}&exp&
[
-\frac{1}{2}[
\frac{1-\bar\alpha_{t}}{(1-\alpha_t)(1-\bar\alpha_{t-1})}\mathbf{x}^2_{t-1}-
2(\frac{\sqrt{\alpha_t}}{1-\alpha_t}\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\mathbf{x}_0)\mathbf{x}_{t-1}+C
]]\\
\xlongequal{\widetilde\beta_t:=\frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_{t}}}&exp&
[
-\frac{1}{2\widetilde\beta_t}[
\mathbf{x}^2_{t-1}-
2\frac{\frac{\sqrt{\alpha_t}\mathbf{x}_t}{1-\alpha_t}+\frac{\sqrt{\bar\alpha_{t-1}}\mathbf{x}_0}{1-\bar\alpha_{t-1}}}{\frac{1-\bar\alpha_{t}}{(1-\alpha_t)(1-\bar\alpha_{t-1})}
}\mathbf{x}_{t-1}+C\widetilde\beta_t
]]\\
=&exp&
[
-\frac{1}{2\widetilde\beta_t}[
\mathbf{x}^2_{t-1}-
2\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}\mathbf{x}_0}{1-\bar\alpha_t
}\mathbf{x}_{t-1}+C\widetilde\beta_t
]]\tag{6}\\

\end{eqnarray}
$$
{ #e426f0}


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
&=&(\frac{\sqrt{(\alpha_t-\bar\alpha_t)(1-\bar\alpha_{t-1})}\mathbf{x_t+\sqrt{(1-\alpha_t)(\bar\alpha_{t-1}-\bar\alpha_t)}\mathbf{x_0}}}{1-\bar\alpha_t})^2\\
&=&(\frac{\sqrt{\alpha_t(1-\bar\alpha_{t-1})(1-\bar\alpha_{t-1})}\mathbf{x_t+\sqrt{\bar\alpha_{t-1}(1-\alpha_t)(1-\alpha_t)}\mathbf{x_0}}}{1-\bar\alpha_t})^2\\
&=&(\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}\mathbf{x_t+(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}\mathbf{x_0}}}{1-\bar\alpha_t})^2\\
\end{eqnarray}
$$
Recall [[diffusion model/Denoising Diffusion Probabilistic Models/DDPM_deduction#^c9714c\|Eq(5)]], [[diffusion model/Denoising Diffusion Probabilistic Models/DDPM_deduction#^e426f0\|Eq(6)]],
$$
\begin{eqnarray}
q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})
&=&\frac{1}{\sqrt{2\pi \frac{(1-\alpha_{t})(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_{t})}}}\int e^{-\frac{(\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{2(1-\alpha_t)}-\frac{(\mathbf{x}_{t-1}-\sqrt{\bar\alpha_{t-1}}\mathbf{x}_{0})^2}{2(1-\bar\alpha_{t-1})}+\frac{(\mathbf{x}_{t}-\sqrt{\bar\alpha_{t}}\mathbf{x}_{0})^2}{2(1-\bar\alpha_{t})}}d\mathbf{x}_{t-1}\\
&=&\frac{1}{\sqrt{2\pi\widetilde{\beta}_t}}
\int e^{
[
-\frac{1}{2\widetilde\beta_t}[
\mathbf{x}^2_{t-1}-
2\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}\mathbf{x}_0}{1-\bar\alpha_t
}\mathbf{x}_{t-1}+C\widetilde\beta_t
]]}
d\mathbf{x}_{t-1}\\
&=&\frac{1}{\sqrt{2\pi\widetilde{\beta}_t}}
\int e^{

-\frac{(
\mathbf{x}_{t-1}-
\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}\mathbf{x}_0}{1-\bar\alpha_t
}
)^2}{2\widetilde\beta_t}}
d\mathbf{x}_{t-1}\\

\end{eqnarray}
$$
Define 
$$\widetilde\mu_t:=\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}\mathbf{x}_0}{1-\bar\alpha_t
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
$$
\begin{eqnarray}
q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})&=&N(\mathbf{x}_{t-1};\widetilde{\mu}_t,\widetilde{\beta}_tI)\\
\mathbf{x}_{t-1}&=&\widetilde{\mu}_t+\sqrt{\widetilde{\beta}_t}\epsilon\\
&=&\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}\mathbf{x}_0}{1-\bar\alpha_t}+\sqrt{\frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_{t}}}\epsilon\\
&=&\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}\mathbf{x}_0}{1-\bar\alpha_t}+\sqrt{\frac{(1-\bar\alpha_{t-1})\beta_t}{1-\bar\alpha_{t}}}\epsilon\\


\end{eqnarray}
$$
Thanks to [[diffusion model/Denoising Diffusion Probabilistic Models/DDPM_deduction#^4610ab\|Eq(2)]], we can represent $\mathbf{x}_0=\frac{1}{\sqrt{\bar\alpha_t}}(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\widetilde\epsilon)$, where $\widetilde\epsilon(\mathbf{x}_t,t)$ denote the noise  estimated by model (Unet+attention)
$$
\begin{eqnarray}
\mathbf{x}_{t-1}
&=&\frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\widetilde\epsilon)+
\sqrt{\frac{(1-\bar\alpha_{t-1})\beta_t}{1-\bar\alpha_{t}}}\epsilon\\


\end{eqnarray}
$$
> [!faq] 
>  **Q1:Why using estimation $\widetilde\epsilon(\mathbf{x}_t,t)$ instead of $\epsilon$? **
>  
>  Because during diffusion process, the noise image is sampled by
>   $$\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_{0}+\sqrt{1-\bar{\alpha_t}}\epsilon $$
>   Of course we can reverse this process with this specific noise image $\mathbf{x}_t$ and noise $\epsilon$ to reconstruct this specfic $\mathbf{x}_0$ by
>   $$\mathbf{x}_0=\frac{1}{\sqrt{\bar\alpha_t}}(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\epsilon)$$
>   However, in the reverse process, we want to construct a real image from **random noise $\epsilon_r$**. Notice that the random noise $\mathbf{x}_t=\epsilon_r$ **does not equal** to the specific noise image $\mathbf{x}_t$, so we are **unable** to reconstruct $\mathbf{x}_0$ by 
>   $$\mathbf{x}_0=\frac{1}{\sqrt{\bar\alpha_t}}(\epsilon_r-\sqrt{1-\bar\alpha_t}\epsilon)$$
>   So a simple idea comes out that we can generate many noise images $\mathbf{x}_{it}$ from different real images $\mathbf{x}_{i0}$ and different diffusion steps $t$ with different noise $\epsilon_i$.
>   $$\mathbf{x}_{it}=\sqrt{\bar{\alpha}_{it}}\mathbf{x}_{i0}+\sqrt{1-\bar{\alpha_{it}}}\epsilon_{i}\quad(i=1,2,...,batchsize) $$
>    Then utilize deep learning methods (Unet+attention) to mining the relationship between $\mathbf{x}_{it}$ and $\epsilon_{i}$. That is to say, learning
>    $$\epsilon_{i}(\mathbf{x}_{it},t)$$ 
>    Consequencely, with random noise $\mathbf{x}_t=\epsilon_r$, we can find the correspound $\epsilon$ which is able to construct $\mathbf{x}_0$
>    
 

# Loss Function
