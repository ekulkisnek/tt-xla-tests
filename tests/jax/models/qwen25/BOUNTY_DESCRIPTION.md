Add a JAX Qwen2.5-7B Model to TT-xla Model Demos
Background:
TT-Forge is a growing collection of model demos showcasing the capabilities of AI models running on Tenstorrent hardware. This bounty aims to bring up the tensor parallelized (not data parallelized) implementation of the Qwen2.5-7B model using the JAX framework. This is an opportunity for AI developers to contribute to cutting-edge research and earn rewards.

Requirements:

Tensor Parallel Implementation in JAX: Ensure that the model is implemented with tensor parallelism, not data parallelism.
Must provide the model implementation, which can (and is preferred to) be an extension of an open source single-device model implementation.
Some examples on an open source JAX models implementation can be found here. https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/modeling_flax_auto.py#L27
Comprehensive Documentation: Include a detailed README.md explaining the model architecture, setup, and usage.
Compatibility and Optimization: Test and optimize for multi-device environments.
Sample Inputs/Outputs: Provide examples demonstrating the model’s functionality.
Dependency Management: Clearly document installation procedures and dependencies.
1 or more of these mesh shapes should be targeted:
2x4
1x8
1x32
8x4
Contribution Guidelines:

Fork the TT-xla repository.
Create a directory in the tt-xla/tests/jax/models folder with the naming convention qwen2_5 and add the model implementation there.
Follow the repository’s coding standards and guidelines as stated in CONTRIBUTING.md.
Submit a pull request with a detailed description and relevant information to help reviewers evaluate your contribution.
Evaluation Criteria:

Tenstorrent Hardware not required for this bounty.
Please use the JAX multidevice simulation guide here. https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html#setup
Implementation Quality: Code structure, readability, and adherence to best practices.
Completeness: We will need to verify the model works.
Tensor parallel implementation should produce the same GSM8K score as the single-device implementation of the same model. https://github.com/openai/grade-school-math
Documentation: Clarity and completeness of the accompanying documentation.
Code must contain descriptive comments that explains the rationale for design decisions made.