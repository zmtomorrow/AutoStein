import jax.numpy as jnp
from jax import grad, jit, jacfwd, jacrev,vmap
from functools import partial


batch_dot=jit(vmap(jnp.dot,in_axes=(0,0)))

class KSD:
    def __init__(self, kernel,score):
        self.batch_k = jit(vmap(kernel,in_axes=(0, 0),out_axes=0))
        self.batch_grad_x = jit(vmap(grad(kernel,0),in_axes=(0, 0),out_axes=0))
        self.batch_grad_y = jit(vmap(grad(kernel,1),in_axes=(0, 0),out_axes=0))
        self.batch_trace_grad_xy=jit(vmap(lambda x,y: jacfwd(jacrev(kernel,0),1)(x,y).trace(),in_axes=(0, 0),out_axes=0))
        self.score = jit(vmap(score,in_axes=0,out_axes=0))
        self.V=0
        self.U=0
        
    @partial(jit, static_argnums=(0,))
    def batch_h(self, x, y):
        grad_log_px = self.score(x)
        grad_log_py = self.score(y)
        a = self.batch_k(x,y) * batch_dot(grad_log_py, grad_log_px)
        b = batch_dot(self.batch_grad_x(x,y), grad_log_py)
        c = batch_dot(self.batch_grad_y(x,y), grad_log_px)
        d = self.batch_trace_grad_xy(x,y)
        return a+b+c+d
    
    def ksd_sum(self,samples,n):
        stein_sum=0
        for i in range(0,n):
            x=jnp.repeat(samples[i:i+1],samples.shape[0],axis=0)
            stein_sum += self.batch_h(x,samples).sum()
        return stein_sum

    def V_stats(self,samples):
        n=samples.shape[0]
        self.V= self.ksd_sum(samples,n)/(n**2)
        return self.V
    
    def U_stats(self,samples):
        n=samples.shape[0]
        self.U= (self.ksd_sum(samples,n)-self.batch_h(samples,samples).sum())/(n*(n-1))
        return self.U

