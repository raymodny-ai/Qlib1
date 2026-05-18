"""
基础设施层单元测试 — TrainerDispatcher / GPUPool / EarlyStopping / CheckpointManager
"""

import os
import tempfile
import pytest
import numpy as np
from unittest.mock import patch
from src.infrastructure.trainer import (
    TrainerDispatcher,
    GPUPool,
    EarlyStopping,
    CheckpointManager,
    Checkpoint,
    TrainingResult,
)


class TestEarlyStopping:
    """早停控制器测试"""

    def test_init_default(self):
        stopper = EarlyStopping(patience=10)
        assert stopper.patience == 10
        assert stopper.best_score is None
        assert not stopper.should_stop

    def test_improving_loss_min_mode(self):
        stopper = EarlyStopping(patience=3, mode="min")
        # 损失递减
        assert not stopper.update(1.0, 0)
        assert stopper.best_score == 1.0
        assert not stopper.update(0.5, 1)
        assert stopper.best_score == 0.5

    def test_stops_after_patience_min(self):
        stopper = EarlyStopping(patience=2, mode="min")
        stopper.update(1.0, 0)
        assert not stopper.update(1.1, 1)
        assert stopper.update(1.2, 2)  # counter=2 >= patience=2 → 停止
        assert stopper.should_stop

    def test_max_mode(self):
        stopper = EarlyStopping(patience=3, mode="max")
        stopper.update(0.5, 0)
        assert not stopper.update(0.6, 1)  # 改善
        assert not stopper.update(0.55, 2)  # 不改善
        assert not stopper.update(0.54, 3)
        assert stopper.update(0.53, 4)  # patience=3 耗尽

    def test_min_delta(self):
        stopper = EarlyStopping(patience=2, min_delta=0.01, mode="min")
        stopper.update(1.0, 0)
        # 改善幅度小于 min_delta 不计为改善
        assert not stopper.update(0.995, 1)  # counter=1
        assert stopper.update(1.001, 2)      # counter=2 → 停止
        assert stopper.should_stop

    def test_reset(self):
        stopper = EarlyStopping(patience=3)
        stopper.update(1.0, 0)
        stopper.update(2.0, 1)
        stopper.reset()
        assert stopper.best_score is None
        assert stopper.counter == 0
        assert not stopper.should_stop


class TestCheckpointManager:
    """检查点管理器测试"""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(checkpoint_dir=tmp)
            ckpt = Checkpoint(epoch=10, model_state={"w": [1, 2, 3]}, metrics={"loss": 0.01})
            path = mgr.save(ckpt)
            assert os.path.exists(path)

            loaded = mgr.load_latest()
            assert loaded is not None
            assert loaded.epoch == 10
            assert loaded.model_state == {"w": [1, 2, 3]}

    def test_load_latest_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(checkpoint_dir=tmp)
            assert mgr.load_latest() is None

    def test_max_keep(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(checkpoint_dir=tmp, max_keep=2)
            for i in range(5):
                ckpt = Checkpoint(epoch=i, model_state={}, metrics={})
                mgr.save(ckpt)
            # 应只保留最后 2 个
            checkpoints = mgr.list_checkpoints()
            assert len(checkpoints) <= 2

    def test_load_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(checkpoint_dir=tmp)
            for i, loss in enumerate([0.5, 0.3, 0.4]):
                ckpt = Checkpoint(epoch=i, model_state={}, metrics={"loss": loss})
                mgr.save(ckpt)
            best = mgr.load_best("loss", mode="min")
            assert best is not None
            assert best.metrics["loss"] == 0.3

    def test_list_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(checkpoint_dir=tmp)
            mgr.save(Checkpoint(epoch=0, model_state={}, metrics={}))
            clist = mgr.list_checkpoints()
            assert len(clist) == 1


class TestGPUPool:
    """GPU 资源池测试"""

    def test_empty_pool(self):
        pool = GPUPool(gpu_ids=[])
        assert pool.available_count == 0

    def test_devices_info(self):
        pool = GPUPool(gpu_ids=[])
        info = pool.devices_info
        assert isinstance(info, list)


class TestTrainerDispatcher:
    """训练调度器测试"""

    def test_init(self):
        trainer = TrainerDispatcher(gpu_ids=[])
        assert trainer.gpu_pool is not None
        assert trainer.checkpoint_manager is not None

    def test_status(self):
        trainer = TrainerDispatcher(gpu_ids=[])
        status = trainer.status
        assert "gpu_pool" in status
        assert "checkpoints" in status

    def test_fit_with_mock_model(self):
        """使用简单模型测试 fit 流程"""
        class MockModel:
            model_name = "MockModel"
            def __init__(self):
                self.is_fitted = False
            def fit(self, X, y, **kwargs):
                self.is_fitted = True
                return self
            def predict(self, X):
                return np.zeros(len(X))
        
        # patch GPUPool 以避免 GPU 检测延迟
        with patch("src.infrastructure.trainer.GPUPool", autospec=True) as mock_pool_cls:
            mock_pool = mock_pool_cls.return_value
            mock_pool.acquire.return_value = None
            mock_pool.release.return_value = None
            mock_pool.available_count = 0
            mock_pool.devices_info = []
            
            trainer = TrainerDispatcher(gpu_ids=[])
            X = np.random.randn(100, 5)
            y = np.random.randn(100)
            
            result = trainer.fit(MockModel(), X, y, verbose=False)
            assert isinstance(result, TrainingResult)
            assert result.model_name == "MockModel"

    def test_training_result_to_dict(self):
        result = TrainingResult(
            model_name="Test",
            epochs_completed=50,
            best_score=0.045,
        )
        d = result.to_dict()
        assert d["model_name"] == "Test"
