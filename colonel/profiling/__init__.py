"""Workload-specific profiling adapters.

A *flavor* here describes the workload we are running (generic process,
vLLM engine, …), as opposed to an *evaluator* (which describes the
profiler — nsys, ncu). Flavors are composed with evaluators at the
CLI / metadata layer; they do not replace evaluators.
"""
