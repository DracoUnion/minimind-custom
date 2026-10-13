# -*- coding: utf-8 -*-
"""
model-diffusion Block Diffusion 架构测试

基于 DiffusionGemma 改造的 MiniMind diffusion 模型，验证要点:
  1. config 新增字段 (canvas_length / num_diffusion_steps / use_prefix_ar_loss)
  2. Encoder 编码 prefix 写共享 KV cache; Decoder 双向注意力只读该 cache
  3. Encoder 与 Decoder 共享同一组 Transformer 层权重; embed 与 lm_head 绑定
  4. forward(input_ids, labels): 自动把序列尾部 canvas_length 个 token 切为 canvas
     作为去噪目标, 前面作为 prefix; loss 在 canvas 上计算
  5. use_prefix_ar_loss: 可选叠加 prefix 的自回归损失 (类似 DiffusionGemma encoder AR loss)
  6. self-conditioning: 传入上一轮 logits 注入 decoder 输入
  7. generate: 随机初始化 canvas 的多步去噪采样, 返回 prefix+canvas 完整序列
  8. 纯 decoder 模式 (input_ids=None): 无前缀纯去噪
  9. MoE 模式: forward / loss / backward / aux_loss 正常
  10. flash_attn=True 路径 (encoder 预填充走 SDPA) 正常

运行:  python test/test_diffusion.py
"""
import os, sys, types, importlib
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, pkgdir="model-diffusion"):
    pkg = types.ModuleType(pkgdir)
    pkg.__path__ = [os.path.join(ROOT, pkgdir)]
    sys.modules[pkgdir] = pkg
    return importlib.import_module(f"{pkgdir}.{name}")


def _make_config(cm, **overrides):
    base = dict(vocab_size=6400, hidden_size=256, num_hidden_layers=4,
                num_attention_heads=8, num_key_value_heads=4, head_dim=32,
                intermediate_size=512, dropout=0.0, flash_attn=False,
                canvas_length=32)
    base.update(overrides)
    return cm.MiniMindConfig(**base)


def test_forward_and_loss():
    cm, mm = _load_mod("config_minimind"), _load_mod("model_minimind")
    torch.manual_seed(0)
    model = mm.MiniMindForCausalLM(_make_config(cm))
    model.eval()

    input_ids = torch.randint(0, 6400, (2, 64))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)
    # canvas 长度 = 32, logits 只覆盖 canvas
    assert out.logits.shape == (2, 32, 6400)
    assert out.aux_loss.item() == 0.0
    print("[OK] forward + canvas loss")

    # 共享权重: encoder 与 decoder 同一层指向同一参数
    assert model.model.encoder.layers[0].self_attn.q_proj.weight.data_ptr() == \
           model.model.decoder.layers[0].self_attn.q_proj.weight.data_ptr()
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    print("[OK] encoder/decoder 共享层权重 + embed/lm_head 绑定")


def test_backward():
    cm, mm = _load_mod("config_minimind"), _load_mod("model_minimind")
    torch.manual_seed(0)
    model = mm.MiniMindForCausalLM(_make_config(cm))
    model.train()

    input_ids = torch.randint(0, 6400, (2, 64))
    out = model(input_ids, labels=input_ids.clone())
    out.loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("[OK] backward 梯度传播")


def test_prefix_ar_loss_switch():
    cm, mm = _load_mod("config_minimind"), _load_mod("model_minimind")
    torch.manual_seed(0)
    input_ids = torch.randint(0, 6400, (2, 64))
    labels = input_ids.clone()

    with_ar = mm.MiniMindForCausalLM(_make_config(cm, use_prefix_ar_loss=True)).eval()
    without_ar = mm.MiniMindForCausalLM(_make_config(cm, use_prefix_ar_loss=False)).eval()
    l1 = with_ar(input_ids, labels=labels).loss
    l2 = without_ar(input_ids, labels=labels).loss
    # 开启 AR loss 时整体 loss 更大 (额外叠加了 prefix 损失)
    assert l1.item() > l2.item(), (l1.item(), l2.item())
    print("[OK] use_prefix_ar_loss 开关 (AR loss 叠加后总 loss 更大)")


def test_self_conditioning():
    cm, mm = _load_mod("config_minimind"), _load_mod("model_minimind")
    torch.manual_seed(0)
    model = mm.MiniMindForCausalLM(_make_config(cm)).eval()

    prefix = torch.randint(0, 6400, (2, 16))
    canvas = torch.randint(0, 6400, (2, 32))
    prev_logits = torch.randn(2, 32, 6400) * 0.1

    out = model(input_ids=prefix, decoder_input_ids=canvas,
                self_conditioning_logits=prev_logits)
    assert out.logits.shape == (2, 32, 6400)

    # 逐样本 mask: False 样本 self-conditioning 信号置零
    out2 = model(input_ids=prefix, decoder_input_ids=canvas,
                 self_conditioning_logits=prev_logits,
                 self_conditioning_mask=torch.tensor([True, False]))
    assert out2.logits.shape == (2, 32, 6400)
    print("[OK] self-conditioning logits / mask")


def test_generate():
    cm, mm = _load_mod("config_minimind"), _load_mod("model_minimind")
    torch.manual_seed(1)
    model = mm.MiniMindForCausalLM(_make_config(cm)).eval()

    prefix = torch.randint(0, 6400, (2, 16))
    out = model.generate(inputs=prefix, num_diffusion_steps=8, temperature=0.9, seed=42)
    assert out.shape == (2, 48), out.shape
    assert torch.equal(out[:, :16], prefix)  # prefix 原样保留
    print("[OK] generate 去噪采样 (prefix + canvas)")


def test_pure_decoder():
    cm, mm = _load_mod("config_minimind"), _load_mod("model_minimind")
    torch.manual_seed(0)
    model = mm.MiniMindForCausalLM(_make_config(cm, hidden_size=128,
                                                num_hidden_layers=2,
                                                head_dim=32)).eval()
    canvas = torch.randint(0, 6400, (1, 32))
    out = model(input_ids=None, decoder_input_ids=canvas)
    assert out.logits.shape == (1, 32, 6400)
    print("[OK] 纯 decoder 模式 (无前缀)")


def test_moe():
    cm, mm = _load_mod("config_minimind"), _load_mod("model_minimind")
    torch.manual_seed(0)
    cfg = _make_config(cm, use_moe=True, num_experts=4, num_experts_per_tok=2)
    model = mm.MiniMindForCausalLM(cfg).train()
    input_ids = torch.randint(0, 6400, (2, 64))
    out = model(input_ids, labels=input_ids.clone())
    out.loss.backward()
    assert out.aux_loss.item() > 0  # MoE 负载均衡辅助损失
    print("[OK] MoE 模式 (aux_loss={:.5f})".format(out.aux_loss.item()))


def test_flash_attn():
    cm, mm = _load_mod("config_minimind"), _load_mod("model_minimind")
    torch.manual_seed(0)
    model = mm.MiniMindForCausalLM(_make_config(cm, flash_attn=True)).eval()
    input_ids = torch.randint(0, 6400, (2, 16))
    out = model(input_ids, labels=input_ids.clone())
    assert out.loss is not None and torch.isfinite(out.loss)
    print("[OK] flash_attn=True (encoder 预填充 SDPA)")


def test_attention_mask():
    cm, mm = _load_mod("config_minimind"), _load_mod("model_minimind")
    torch.manual_seed(0)
    model = mm.MiniMindForCausalLM(_make_config(cm)).eval()
    input_ids = torch.randint(0, 6400, (2, 64))
    labels = input_ids.clone()
    mask = torch.ones(2, 64, dtype=torch.bool)
    out = model(input_ids, attention_mask=mask, labels=labels)
    assert torch.isfinite(out.loss)
    print("[OK] attention_mask 前向")


if __name__ == "__main__":
    test_forward_and_loss()
    test_backward()
    test_prefix_ar_loss_switch()
    test_self_conditioning()
    test_generate()
    test_pure_decoder()
    test_moe()
    test_flash_attn()
    test_attention_mask()
    print("\nAll model-diffusion tests passed.")
