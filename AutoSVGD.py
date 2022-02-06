import jax.numpy as jnp
from jax import grad, jit,vmap
from functools import partial



class SVGD:
    def __init__(self, kernel,score):
        self.batch_k = jit(vmap(kernel,in_axes=(0, 0),out_axes=0))
        self.batch_grad_x = jit(vmap(grad(kernel,0),in_axes=(0, 0),out_axes=0))
        self.score = jit(vmap(score,in_axes=0,out_axes=0))
        
    @partial(jit, static_argnums=(0,))
    def svg(self, x, y):
        return (self.batch_k(x,y) * self.score(x)+self.batch_grad_x(x,y)).mean()
    
    def update(self,samples,lr):
        n=samples.shape[0]
        for i in range(0,n):
            delta=lr*self.svg(samples, jnp.repeat(samples[i:i+1],n,axis=0))
            samples=samples.at[i].add(delta)
        return samples


