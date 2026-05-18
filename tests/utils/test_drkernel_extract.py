import pytest

from slime_plugins.drkernel.extract import extract_assistant_response, extract_kernel_submission, strip_think_blocks


@pytest.mark.unit
def test_extract_cuda_agent_submission_formats_complete_sections():
    response = """
<think>draft</think>
Here is the implementation.

### CUDA_KERNELS
```cuda
extern "C" __global__ void add_one(float* x) {}
```

### APPLY_BINDINGS
```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
```

### MODEL_NEW
```python
class ModelNew:
    pass
```
"""

    submission = extract_kernel_submission(response)

    assert submission.backend == "cuda_agent"
    assert submission.sections == {
        "CUDA_KERNELS": 'extern "C" __global__ void add_one(float* x) {}',
        "APPLY_BINDINGS": "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}",
        "MODEL_NEW": "class ModelNew:\n    pass",
    }
    assert submission.code == (
        "### CUDA_KERNELS\n"
        "```cpp\n"
        'extern "C" __global__ void add_one(float* x) {}\n'
        "```\n\n"
        "### APPLY_BINDINGS\n"
        "```cpp\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}\n"
        "```\n\n"
        "### MODEL_NEW\n"
        "```python\n"
        "class ModelNew:\n"
        "    pass\n"
        "```"
    )


@pytest.mark.unit
def test_extract_kernel_submission_uses_last_complete_group():
    response = """
### CUDA_KERNELS
```cpp
old kernel
```
### APPLY_BINDINGS
```cpp
old binding
```
### MODEL_NEW
```python
old model
```
### CUDA_KERNELS
```cpp
new kernel
```
### APPLY_BINDINGS
```cpp
new binding
```
### MODEL_NEW
```python
new model
```
"""

    submission = extract_kernel_submission(response)

    assert submission is not None
    assert submission.sections == {
        "CUDA_KERNELS": "new kernel",
        "APPLY_BINDINGS": "new binding",
        "MODEL_NEW": "new model",
    }


@pytest.mark.unit
def test_extract_returns_empty_result_for_incomplete_cuda_agent_submission():
    response = """
### CUDA_KERNELS
```cpp
only kernels
```
"""

    assert extract_kernel_submission(response) is None


@pytest.mark.unit
def test_extract_does_not_fallback_to_plain_code_block():
    assert extract_kernel_submission("```python\nx = 1\n```") is None


@pytest.mark.unit
def test_strip_think_blocks_prefers_text_after_last_closing_tag():
    response = "<think>bad answer</think>final <think>hidden</think>answer"

    assert strip_think_blocks(response) == "answer"


@pytest.mark.unit
def test_extract_assistant_response_uses_last_assistant_message():
    messages = [
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "retry"},
        {"role": "assistant", "content": "second"},
    ]

    assert extract_assistant_response(messages) == "second"
