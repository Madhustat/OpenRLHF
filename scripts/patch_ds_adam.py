import pathlib
f = pathlib.Path("/opt/venv/lib/python3.12/site-packages/deepspeed/runtime/engine.py")
src = f.read_text()
old = '''                else:
                    from deepspeed.ops.adam import FusedAdam

                    optimizer = FusedAdam(
                        model_parameters,
                        **optimizer_parameters,
                        adam_w_mode=effective_adam_w_mode,
                    )'''
new = '''                else:
                    try:
                        from deepspeed.ops.adam import FusedAdam
                        optimizer = FusedAdam(
                            model_parameters,
                            **optimizer_parameters,
                            adam_w_mode=effective_adam_w_mode,
                        )
                    except Exception:
                        import torch
                        optimizer = torch.optim.AdamW(
                            model_parameters,
                            **{k: v for k, v in optimizer_parameters.items()
                               if k in ("lr", "betas", "eps", "weight_decay")}
                        )'''
assert old in src, "block not found — check indentation"
patched = src.replace(old, new, 1)
assert patched != src, "no change made"
f.write_text(patched)
print("patched engine.py OK: FusedAdam -> torch.optim.AdamW fallback on JIT failure")
