"""
基础设施层单元测试 — SignalExporter / SignalBatch / OMSAdapter / ProductionGateway
"""

import os
import json
import tempfile
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.infrastructure.signal_exporter import (
    SignalExporter,
    SignalBatch,
    SignalEntry,
    Order,
    OMSAdapter,
    ProductionGateway,
)


class TestSignalEntry:
    """信号条目测试"""

    def test_create(self):
        entry = SignalEntry(
            instrument="AAPL",
            action="BUY",
            target_weight=0.05,
            score=0.8,
        )
        assert entry.instrument == "AAPL"
        assert entry.action == "BUY"


class TestSignalBatch:
    """信号批次测试"""

    def test_create_empty(self):
        batch = SignalBatch(model_name="lgb_v1", approved_by="pm_zhang")
        assert batch.model_name == "lgb_v1"
        assert batch.signals == []

    def test_validate_empty_signals(self):
        batch = SignalBatch(approved_by="pm")
        valid, msg = batch.validate()
        assert not valid
        assert "为空" in msg

    def test_validate_no_approval(self):
        batch = SignalBatch(signals=[SignalEntry("AAPL", "BUY", 1.0, 0.8)])
        valid, msg = batch.validate()
        assert not valid
        assert "审批" in msg

    def test_validate_weights_overflow(self):
        batch = SignalBatch(
            approved_by="pm",
            signals=[
                SignalEntry("A", "BUY", 0.6, 0.8),
                SignalEntry("B", "BUY", 0.6, 0.7),
            ],
        )
        valid, msg = batch.validate()
        assert not valid

    def test_validate_duplicate_instruments(self):
        batch = SignalBatch(
            approved_by="pm",
            signals=[
                SignalEntry("A", "BUY", 0.4, 0.8),
                SignalEntry("A", "BUY", 0.3, 0.7),
            ],
        )
        valid, msg = batch.validate()
        assert not valid

    def test_validate_valid(self):
        batch = SignalBatch(
            approved_by="pm",
            signals=[
                SignalEntry("AAPL", "BUY", 0.5, 0.8),
                SignalEntry("MSFT", "BUY", 0.3, 0.7),
            ],
        )
        valid, msg = batch.validate()
        assert valid

    def test_to_payload(self):
        batch = SignalBatch(
            model_name="lgb",
            approved_by="pm",
            signals=[SignalEntry("AAPL", "BUY", 1.0, 0.9)],
        )
        payload = batch.to_payload()
        assert payload["model_name"] == "lgb"
        assert len(payload["signals"]) == 1

    def test_from_payload(self):
        payload = {
            "batch_id": "test_001",
            "generated_at": "2024-01-01T00:00:00",
            "signals": [{"instrument": "AAPL", "action": "BUY", "target_weight": 1.0, "score": 0.9}],
        }
        batch = SignalBatch.from_payload(payload)
        assert batch.batch_id == "test_001"
        assert len(batch.signals) == 1


class TestSignalExporter:
    """信号导出器测试"""

    @pytest.fixture
    def exporter(self):
        return SignalExporter()

    @pytest.fixture
    def sample_predictions(self):
        np.random.seed(42)
        instruments = [f"STOCK_{i:03d}" for i in range(50)]
        scores = np.random.randn(50) * 0.05 + 0.01
        return pd.Series(scores, index=instruments)

    def test_build_batch(self, exporter, sample_predictions):
        batch = exporter.build_batch(
            sample_predictions,
            model_name="test_model",
            approved_by="pm_test",
            top_k=10,
        )
        assert isinstance(batch, SignalBatch)
        assert batch.model_name == "test_model"
        assert len(batch.signals) == 10

    def test_build_batch_with_weights(self, exporter, sample_predictions):
        weights = pd.Series(1.0 / 50, index=sample_predictions.index)
        batch = exporter.build_batch(
            sample_predictions,
            weights=weights,
            approved_by="pm",
            top_k=5,
        )
        assert batch.approved_by == "pm"

    def test_build_batch_min_score(self, exporter, sample_predictions):
        batch = exporter.build_batch(
            sample_predictions,
            approved_by="pm",
            top_k=50,
            min_score=0.05,  # 高阈值 → 少量标的
        )
        assert len(batch.signals) <= 50

    def test_sign_and_verify(self, exporter, sample_predictions):
        batch = exporter.build_batch(sample_predictions, approved_by="pm", top_k=5)
        assert exporter.verify_signature(batch)

    def test_encrypt_decrypt_no_encryptor(self, exporter, sample_predictions):
        batch = exporter.build_batch(sample_predictions, approved_by="pm", top_k=5)
        encrypted = exporter.encrypt_batch(batch)
        decrypted = exporter.decrypt_batch(encrypted)
        assert decrypted.batch_id == batch.batch_id

    def test_encrypt_decrypt_with_aes(self, sample_predictions):
        from src.security import AES256Encryptor
        aes = AES256Encryptor()
        exporter = SignalExporter(encryptor=aes)
        
        batch = exporter.build_batch(sample_predictions, approved_by="pm", top_k=5)
        encrypted = exporter.encrypt_batch(batch)
        decrypted = exporter.decrypt_batch(encrypted)
        assert decrypted.batch_id == batch.batch_id
        assert len(decrypted.signals) == len(batch.signals)

    def test_push_dry_run(self, exporter, sample_predictions):
        batch = exporter.build_batch(sample_predictions, approved_by="pm", top_k=5)
        result = exporter.push(batch, dry_run=True)
        assert result["success"]
        assert "DRY RUN" in result["message"]

    def test_push_no_url(self, exporter, sample_predictions):
        batch = exporter.build_batch(sample_predictions, approved_by="pm", top_k=5)
        result = exporter.push(batch)
        assert not result["success"]
        assert "URL" in result["message"]

    def test_export_to_file(self, exporter, sample_predictions):
        batch = exporter.build_batch(sample_predictions, approved_by="pm", top_k=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "signals.json")
            result = exporter.export_to_file(batch, path, encrypt=False)
            assert os.path.exists(result)


class TestOMSAdapter:
    """OMS 适配器测试"""

    def test_submit_order(self):
        oms = OMSAdapter()
        order = Order(order_id="ord_001", instrument="AAPL", side="BUY", quantity=100)
        assert oms.submit_order(order)
        assert order.status == "SUBMITTED"

    def test_cancel_order(self):
        oms = OMSAdapter()
        oms.submit_order(Order(order_id="ord_001", instrument="AAPL", side="BUY", quantity=100))
        assert oms.cancel_order("ord_001")
        assert not oms.cancel_order("nonexistent")

    def test_get_orders(self):
        oms = OMSAdapter()
        oms.submit_order(Order(order_id="o1", instrument="A", side="BUY", quantity=10))
        oms.submit_order(Order(order_id="o2", instrument="B", side="SELL", quantity=5))
        assert len(oms.get_orders()) == 2
        assert len(oms.get_orders("SUBMITTED")) == 2

    def test_emergency_stop(self):
        oms = OMSAdapter()
        oms.submit_order(Order(order_id="o1", instrument="A", side="BUY", quantity=10))
        oms.submit_order(Order(order_id="o2", instrument="B", side="BUY", quantity=5))
        cancelled = oms.emergency_stop()
        assert cancelled == 2

    def test_convert_signals(self):
        oms = OMSAdapter()
        batch = SignalBatch(
            batch_id="test",
            approved_by="pm",
            signals=[
                SignalEntry("AAPL", "BUY", 0.5, 0.8),
                SignalEntry("MSFT", "BUY", 0.5, 0.7),
            ],
        )
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        orders = oms.convert_signals(batch, capital=100000, prices=prices)
        assert len(orders) == 2


class TestProductionGateway:
    """生产网关测试"""

    def test_approve_without_permission(self):
        gw = ProductionGateway()
        with patch.object(gw, "rbac") as mock_rbac:
            mock_rbac.can_push_signal.return_value = False
            result = gw.approve_and_push(
                pd.DataFrame(), approved_by="researcher_li"
            )
            assert not result["success"]

    def test_emergency_shutdown_no_permission(self):
        gw = ProductionGateway()
        with patch.object(gw, "rbac") as mock_rbac:
            mock_rbac.can_emergency_stop.return_value = False
            result = gw.emergency_shutdown("unauthorized_user")
            assert not result["success"]

    def test_emergency_shutdown_with_permission(self):
        gw = ProductionGateway()
        with patch.object(gw, "rbac") as mock_rbac:
            mock_rbac.can_emergency_stop.return_value = True
            result = gw.emergency_shutdown("pm_zhang")
            assert result["success"]
