# AutoStein: a JAX package for Stein based methods with AutoDiff.
This package including following methods:
* AutoKSD: Kernelized Stein Discrepancy [[1](https://arxiv.org/pdf/1602.03253.pdf),[2](https://arxiv.org/pdf/1602.02964.pdf)].
* AutoSVGD: Stein Variational Gradient Descent [[1](https://arxiv.org/pdf/1608.04471.pdf)].

### AutoKSD
The following code shows the simplicity of  AutoKSD.
``` python
## Define the kernel function
rbf=lambda x,y:jnp.exp(-1*jnp.sum((x-y)**2))

## Define the score function of p
p_score=grad(lambda x:norm.logpdf(x,loc=0., scale=1.).sum())

## Initialize the KSD
ksd=KSD(rbf,p_score)

## Samples from q
q_samples=random.normal(key,[100,2])

## Compute statsitics
print(ksd.U_stats(q_samples))
print(ksd.V_stats(q_samples))
```
### AutoSVGD
The following code shows the simplicity of AutoKSD.
``` python
## Define the kernel function
rbf=lambda x,y:jnp.exp(-1*jnp.sum((x-y)**2))

## Define the score function of p
p_score=grad(lambda x:norm.logpdf(x,loc=0., scale=1.).sum())

## Initialize the SVGD
svgd=SVGD(rbf,p_score)

## Initialize the samples
samples=random.normal(key,[100,2])

## Update samples
for i in range(0,100):
    samples=svgd.update(samples,lr=0.1)
```
