from joblib.parallel import register_parallel_backend


def register_ray():
    """Register Ray Backend to be called with parallel_backend("ray")."""
    try:
        from ray.util.joblib.ray_backend import RayBackend

        register_parallel_backend("ray", RayBackend)
    except ImportError:
        msg = (
            "To use the ray backend you must install ray."
            "Try running 'uv pip install --system ray'."
            "See https://docs.ray.io/en/master/installation.html"
            "for more information."
        )
        raise ImportError(msg)


__all__ = ["register_ray"]
