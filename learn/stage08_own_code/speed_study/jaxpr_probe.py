"""Print what JAX hands to folx for det and for slogdet of an 18 x 18 matrix (CPU; Stage 8 speed study)."""
import jax, jax.numpy as jnp
x = jnp.eye(18) + 0.01 * jnp.arange(324.0).reshape(18, 18) / 324
for name, f in [("det", jnp.linalg.det), ("slogdet", lambda a: jnp.linalg.slogdet(a)[1])]:
    seen = []
    def walk(j, depth=0):
        for e in j.eqns:
            label = "  " * depth + e.primitive.name + (f"[name={e.params.get('name')}]" if "name" in e.params else "")
            if label not in seen:
                seen.append(label)
            for p in ("jaxpr", "call_jaxpr"):
                if p in e.params and depth < 2:
                    inner = e.params[p]; walk(getattr(inner, "jaxpr", inner), depth + 1)
    walk(jax.make_jaxpr(f)(x).jaxpr)
    print(name, "|", " ; ".join("depth%d:%s" % ((len(s) - len(s.lstrip())) // 2, s.strip()) for s in seen[:16]))
