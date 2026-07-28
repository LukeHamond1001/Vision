"""Structural tests for the learned-latent contract (SPEC §11)."""

import unittest

import torch

from iga.envs.hdcharge import HDSensor, NonlinearObservationLatent
from iga.pretrain import EncoderMLP, geodesic_pairs, pretrain_mlp_encoder


class TestLearnedLatentContract(unittest.TestCase):
    def _tiny_encoder(self):
        torch.manual_seed(0)
        obs = torch.randn(500, 32) * 0.3
        return pretrain_mlp_encoder(obs, band_dims=[6, 2], taus=[10.0, 300.0],
                                    lags=[15, 15], segment_len=100,
                                    context_amp=0.3, geo=None, lam_geo=0.0,
                                    epochs=20, seed=0)

    def test_R1_frozen_after_pretraining(self):
        enc = self._tiny_encoder()
        for p in enc.parameters():
            self.assertFalse(p.requires_grad, "SPEC R1: encoder must be frozen")
        lat = NonlinearObservationLatent(enc, [6, 2], HDSensor())
        for t in lat.frozen_tensors():
            self.assertFalse(t.requires_grad)

    def test_R1_embed_delta_raises_loudly(self):
        """Nonlinear encoders MUST raise on one-step lookahead rather than
        silently approximate (C4 forward-model precondition)."""
        lat = NonlinearObservationLatent(self._tiny_encoder(), [6, 2], HDSensor())
        with self.assertRaises(NotImplementedError):
            lat.embed_delta(torch.tensor([0.1, 0.0]))

    def test_R2_geodesic_pairs_contract(self):
        torch.manual_seed(1)
        obs = torch.randn(800, 32) * 0.3
        node_idx, gi, gj, d = geodesic_pairs(obs, n_nodes=100, k=6, n_pairs=500)
        self.assertTrue(torch.isfinite(d).all())      # disconnected pairs dropped
        self.assertTrue((d > 0).all())                # no self-pairs
        self.assertTrue((gi != gj).all())
        self.assertLessEqual(int(node_idx.max()), 799)

    def test_scale_normalization_via_out_scale(self):
        enc = self._tiny_encoder()
        lat = NonlinearObservationLatent(enc, [6, 2], HDSensor())
        s0 = lat.step_scale(0)
        lat._post_scale(s0)
        self.assertAlmostEqual(lat.step_scale(0), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
