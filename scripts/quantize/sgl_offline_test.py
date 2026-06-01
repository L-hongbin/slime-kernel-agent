# launch the offline engine


import sglang as sgl


def main():
    llm_path = "checkpoints/quantized/Qwen3.6-27B-smooth-bf16-nonla"
    llm_path = "checkpoints/quantized/Qwen3.6-27B-smooth-w8a8-nonla"

    print(f"{llm_path=}")
    llm = sgl.Engine(
        model_path=llm_path,
        trust_remote_code=True,
        disable_cuda_graph=True,
        # disable_radix_cache=True,
        # disable_cuda_graph_padding=False,
        # disable_outlines_disk_cache=False,
        # disable_custom_all_reduce=False,
        # disable_overlap_schedule=False,
        # enable_nan_detection=False,
        # enable_p2p_check=False,
        # triton_attention_reduce_in_fp32=False,
        tp_size=2,
    )

    prompts = [
        "Hello, my name is",
        # "The president of the United States is",
        # "The capital of France is",
        # "The future of AI is",
    ]

    sampling_params = {"temperature": 0.001, "top_p": 0.95, "max_new_tokens": 32}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs, strict=False):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    # set_env()
    main()
