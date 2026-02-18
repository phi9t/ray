(serving-llms)=

# Serving LLMs

Ray Serve LLM provides a high-performance, scalable framework for deploying Large Language Models (LLMs) in production. It specializes Ray Serve primitives for distributed LLM serving workloads, offering enterprise-grade features with OpenAI API compatibility.

## Why Ray Serve LLM?

Ray Serve LLM excels at highly distributed multi-node inference workloads:

- **Advanced parallelism strategies**: Seamlessly combine pipeline parallelism, tensor parallelism, expert parallelism, and data parallel attention for models of any size.
- **Prefill-decode disaggregation**: Separates and optimizes prefill and decode phases independently for better resource utilization and cost efficiency.
- **Custom request routing**: Implements prefix-aware, session-aware, or custom routing logic to maximize cache hits and reduce latency.
- **Multi-node deployments**: Serves massive models that span multiple nodes with automatic placement and coordination.
- **Production-ready**: Has built-in autoscaling, monitoring, fault tolerance, and observability.

## Features

- ⚡️ Automatic scaling and load balancing
- 🌐 Unified multi-node multi-model deployment
- 🔌 OpenAI-compatible API
- 🔄 Multi-LoRA support with shared base models
- 🚀 Engine-agnostic architecture (vLLM, SGLang, etc.)
- 📊 Built-in metrics and Grafana dashboards
- 🎯 Advanced serving patterns (PD disaggregation, data parallel attention)

## Requirements

```bash
uv pip install --system ray[serve,llm]
```

```{toctree}
:hidden:

Quickstart <quick-start>
Examples <examples>
User Guides <user-guides/index>
Architecture <architecture/index>
Benchmarks <benchmarks>
Troubleshooting <troubleshooting>
```

## Next steps

- {doc}`Quickstart <quick-start>` - Deploy your first LLM with Ray Serve
- {doc}`Examples <examples>` - Production-ready deployment tutorials
- {doc}`User Guides <user-guides/index>` - Practical guides for advanced features
- {doc}`Architecture <architecture/index>` - Technical design and implementation details
- {doc}`Troubleshooting <troubleshooting>` - Common issues and solutions