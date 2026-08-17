"""Install the bounded GLM-5.2 expert intervention in vLLM worker processes."""

from qsrt.glm52_expert_intervention_runtime import install_vllm_patch


install_vllm_patch()
