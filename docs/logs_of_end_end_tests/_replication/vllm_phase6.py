import time
from vllm import LLM, SamplingParams


def main():
    print("Building vLLM engine for Qwen2.5-1.5B-Instruct on XPU...")
    t0 = time.time()
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", dtype="bfloat16",
              gpu_memory_utilization=0.4, enforce_eager=True, max_model_len=2048)
    print(f"Engine built in {time.time()-t0:.1f}s")
    prompts = ["Explain reinforcement learning.", "What is DeepSpeed?"]
    sp = SamplingParams(temperature=0.7, max_tokens=40)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    print(f"Generated in {time.time()-t0:.1f}s")
    for o in outs:
        print("PROMPT:", o.prompt)
        print("OUTPUT:", o.outputs[0].text.strip()[:200])
        print("---")
    print("PHASE6_RESULT=PASS")


if __name__ == "__main__":
    main()
