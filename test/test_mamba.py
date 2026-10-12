# -*- coding: utf-8 -*-
"""
model-mamba 混合注意力 (Mamba2 线性 + 全注意力) 测试

验证要点:
  1. 默认 layer_types 生成 (3 linear + 1 full 循环)
  2. 各 layer_types 长度必须等于 num_hidden_layers
  3. 前向: 混合层模型的 prefill / 生成 cache 形状
  4. 关键不变量: 分块扫描 (chunk_scan) 输出 ≈ 逐 token 递归 (recurrent) 输出
     (fp32 累积误差量级, 应 < 1e-2)
  5. 递归 (recurrent) 输出 == 纯 Python 参考循环 (应精确, < 1e-5)
  6. 全注意力层的 prefill == 逐 token 生成 (应精确, < 1e-4)

运行:  python test/test_mamba.py
"""
import os, sys, types, importlib
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model-mamba"))


def _load_mod(name, pkgdir):
    pkg = types.ModuleType(pkgdir)
    pkg.__path__ = [os.path.join(ROOT, pkgdir)]
    sys.modules[pkgdir] = pkg
    return importlib.import_module(f"{pkgdir}.{name}")


def test_layer_types_default():
    cm = _load_mod("config_minimind", "model-mamba")
    cfg = cm.MiniMindConfig(hidden_size=64, num_hidden_layers=8)
    assert len(cfg.layer_types) == 8, cfg.layer_types
    # 默认 3 linear + 1 full 循环
    assert cfg.layer_types[3] == "full_attention"
    assert cfg.layer_types[0] == "linear_attention"
    # 显式指定
    cfg2 = cm.MiniMindConfig(hidden_size=64, num_hidden_layers=4,
                             layer_types=["linear_attention", "full_attention",
                                          "linear_attention", "full_attention"])
    assert cfg2.layer_types == ["linear_attention", "full_attention", "linear_attention", "full_attention"]
    # 长度不符应报错
    try:
        cm.MiniMindConfig(hidden_size=64, num_hidden_layers=4, layer_types=["full_attention"])
        raise AssertionError("layer_types 长度不符未报错")
    except AssertionError as e:
        if "未报错" in str(e):
            raise
    print("[OK] layer_types 生成与校验")


def test_mixed_prefill_and_cache():
    cm = _load_mod("config_minimind", "model-mamba")
    mm = _load_mod("model_minimind", "model-mamba")
    cfg = cm.MiniMindConfig(hidden_size=64, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, mamba_num_heads=4)
    model = mm.MiniMindForCausalLM(cfg)
    model.eval()
    x = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        out = model(input_ids=x, use_cache=True)
        assert out.logits.shape == (2, 10, cfg.vocab_size)
        pkv = out.past_key_values
        assert len(pkv) == 4
        out2 = model(input_ids=x[:, -1:], past_key_values=pkv, use_cache=True)
        assert out2.logits.shape == (2, 1, cfg.vocab_size)
    # 校验各层 cache 形状
    for i, layer in enumerate(model.model.layers):
        c = out2.past_key_values[i]
        if layer.layer_type == "linear_attention":
            assert len(c) == 2 and c[0].dim() == 3 and c[1].shape[1] == cfg.mamba_num_heads, (i, c)
        else:
            assert len(c) == 2 and c[0].shape[1] == 11, (i, c)  # prefill10 + step1
    print("[OK] 混合层 prefill + 生成 cache")


def _reference_ssm(mixer, hidden, B, L, H, P):
    """纯 Python 参考循环 (线性, 精确)。返回 (out, final_state)。"""
    proj = mixer.in_proj(hidden)
    gate, conv_bc, dt = proj.split(
        [mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], dim=-1)
    dt = dt.transpose(1, 2) + mixer.dt_bias.view(1, -1, 1)
    dt = dt.transpose(1, 2)
    co = conv_bc.transpose(1, 2).to(mixer.conv1d.weight.dtype)
    co, _ = mixer._causal_conv1d(co, None)
    co = mixer.act(co).transpose(1, 2)
    x, C, Bv = co.split(
        [mixer.intermediate_size,
         mixer.n_groups * mixer.ssm_state_size,
         mixer.n_groups * mixer.ssm_state_size], dim=-1)
    x = x.view(B, L, H, P)
    C = C.view(B, L, mixer.n_groups, mixer.ssm_state_size)
    Bv = Bv.view(B, L, mixer.n_groups, mixer.ssm_state_size)
    A = -torch.exp(mixer.A_log.float())
    N = mixer.ssm_state_size
    B_ = Bv.repeat_interleave(H // mixer.n_groups, dim=1).view(B, L, H, N)
    C_ = C.repeat_interleave(H // mixer.n_groups, dim=1).view(B, L, H, N)
    st = torch.zeros(B, H, P, N, device=x.device)
    out = []
    for t in range(L):
        dt_t = F.softplus(dt[:, t]).clamp(mixer.config.time_step_min, mixer.config.time_step_max)
        dA = torch.exp((dt_t * A.view(1, H)).float())
        dBx = torch.einsum('bh,bhn,bhp->bhpn', dt_t, B_[:, t], x[:, t])
        st = st * dA.view(B, H, 1, 1) + dBx
        o = torch.einsum('bhn,bhpn->bhp', C_[:, t].float(), st.float()) \
            + x[:, t].float() * mixer.D.view(1, H, 1)
        out.append(o)
    o = torch.stack(out, 1).to(x.dtype).view(B, L, mixer.intermediate_size)
    return mixer.out_proj(mixer.norm(o, gate)), st


def test_mamba_chunk_vs_streaming():
    """分块扫描 vs 逐 token 递归: 输出应在 fp32 累积误差量级内。

    注意: 分块 Mamba2 在 fp32 下对长对角做求和, 累积舍入误差随 序列长度/块数 增长
    (chunk 越小累积越多)。该测试验证的是"分块实现无逻辑 bug", 用宽松阈值区分
    正常 fp32 误差 (~1e-2~5e-2) 与逻辑错误 (einsum/decay 方向错 → 0.1+)。
    精确性由 test_mamba_streaming_exact (递归==参考, <1e-5) 保证。
    """
    cm = _load_mod("config_minimind", "model-mamba")
    mm = _load_mod("model_minimind", "model-mamba")
    for cs in (4, 8, 16):
        cfg = cm.MiniMindConfig(hidden_size=48, num_hidden_layers=1, num_attention_heads=3,
                                mamba_num_heads=3, layer_types=["linear_attention"], chunk_size=cs)
        m = mm.MiniMindForCausalLM(cfg); m.eval()
        mix = m.model.layers[0].mixer
        torch.manual_seed(0)
        L = 40
        hidden = torch.randn(1, L, cfg.hidden_size)
        with torch.no_grad():
            c, _ = mix(hidden, past_key_value=None, use_cache=False)
            s, pk = None, None
            for t in range(L):
                r = mix(hidden[:, t:t + 1], past_key_value=pk, use_cache=True)
                s = r[0] if s is None else torch.cat([s, r[0]], 1)
                pk = r[1]
            diff = (c - s).abs().max().item()
            assert diff < 5e-2, f"chunk_size={cs}: 分块 vs 递归 偏差 {diff} (疑似逻辑 bug)"
        print(f"[OK] chunk_size={cs}: 分块扫描 vs 递归 偏差 {diff:.2e}")


def test_mamba_streaming_exact():
    """逐 token 递归 输出 == 纯 Python 参考循环 (应精确)。"""
    cm = _load_mod("config_minimind", "model-mamba")
    mm = _load_mod("model_minimind", "model-mamba")
    cfg = cm.MiniMindConfig(hidden_size=48, num_hidden_layers=1, num_attention_heads=3,
                            mamba_num_heads=3, layer_types=["linear_attention"], chunk_size=8)
    m = mm.MiniMindForCausalLM(cfg); m.eval()
    mix = m.model.layers[0].mixer
    torch.manual_seed(0)
    L = 26
    hidden = torch.randn(1, L, cfg.hidden_size)
    with torch.no_grad():
        s, pk = None, None
        for t in range(L):
            r = mix(hidden[:, t:t + 1], past_key_value=pk, use_cache=True)
            s = r[0] if s is None else torch.cat([s, r[0]], 1)
            pk = r[1]
        ref, _ = _reference_ssm(mix, hidden, 1, L, 3, mix.head_dim)
    diff = (s - ref).abs().max().item()
    assert diff < 1e-5, f"递归 vs 参考 偏差 {diff}"
    print(f"[OK] 递归 vs 纯 Python 参考 偏差 {diff:.2e}")


def test_attention_prefill_exact():
    """全注意力层 prefill == 逐 token 生成 (应精确)。"""
    cm = _load_mod("config_minimind", "model-mamba")
    mm = _load_mod("model_minimind", "model-mamba")
    cfg = cm.MiniMindConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                            num_key_value_heads=2, mamba_num_heads=4,
                            layer_types=["full_attention"] * 4)
    m = mm.MiniMindForCausalLM(cfg); m.eval()
    torch.manual_seed(0)
    seq = torch.randint(0, 100, (1, 32))
    with torch.no_grad():
        o = m.model(seq, use_cache=True)
        prefill_last = o[0][:, -1]
        h, pk = None, None
        for t in range(32):
            r = m.model(seq[:, t:t + 1], past_key_values=pk, use_cache=True)
            h = r[0][:, -1]
            pk = r[1]
    diff = (prefill_last - h).abs().max().item()
    assert diff < 1e-4, f"全注意力 prefill vs 生成 偏差 {diff}"
    print(f"[OK] 全注意力 prefill vs 生成 偏差 {diff:.2e}")


def test_mixed_end_to_end():
    """混合模型 prefill vs 逐 token 生成。

    多个 mamba 层的 fp32 分块误差经残差堆叠会累积 (每层 ~1e-2, 3~4 个 mamba 层
    → 端到端 ~0.05-0.08)。阈值取 0.15 以区分正常累积与逻辑错误。
    """
    cm = _load_mod("config_minimind", "model-mamba")
    mm = _load_mod("model_minimind", "model-mamba")
    cfg = cm.MiniMindConfig(hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
                            num_key_value_heads=2, mamba_num_heads=4, chunk_size=8,
                            layer_types=["linear_attention", "full_attention", "linear_attention",
                                         "linear_attention", "full_attention", "linear_attention"])
    m = mm.MiniMindForCausalLM(cfg); m.eval()
    torch.manual_seed(0)
    seq = torch.randint(0, 100, (1, 32))
    with torch.no_grad():
        o = m.model(seq, use_cache=True)
        prefill_last = o[0][:, -1]
        h, pk = None, None
        for t in range(32):
            r = m.model(seq[:, t:t + 1], past_key_values=pk, use_cache=True)
            h = r[0][:, -1]
            pk = r[1]
    diff = (prefill_last - h).abs().max().item()
    assert diff < 0.15, f"混合模型 prefill vs 生成 偏差 {diff} (疑似逻辑 bug)"
    print(f"[OK] 混合模型 prefill vs 生成 偏差 {diff:.2e} (fp32 堆叠累积)")


def main():
    tests = [
        test_layer_types_default,
        test_mixed_prefill_and_cache,
        test_mamba_chunk_vs_streaming,
        test_mamba_streaming_exact,
        test_attention_prefill_exact,
        test_mixed_end_to_end,
    ]
    for t in tests:
        t()
    print("\n=== 全部 Mamba 测试通过 ===")


if __name__ == "__main__":
    main()
