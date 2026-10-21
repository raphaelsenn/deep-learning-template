import pytest
import torch
import torch.nn as nn

from src import TemplateModel


class TestTemplateModel:
    @pytest.fixture
    def model(self) -> TemplateModel:
        return TemplateModel(
            in_channels=1,
            hidden_dim=256,
            out_dim=10,
        )

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_output_shape(
        self,
        model: TemplateModel,
        batch_size: int,
    ) -> None:
        x = torch.randn(batch_size, 1, 28, 28)
        output = model(x)
        assert output.shape == (batch_size, 10)

    def test_initialization(self, model: TemplateModel) -> None:
        assert isinstance(model, nn.Module)
        assert isinstance(model.conv_net_2d, nn.Sequential)
        assert isinstance(model.cls_mlp, nn.Sequential)
        assert model.conv_net_2d[0].in_channels == 1
        assert model.cls_mlp[0].in_features == 120
        assert model.cls_mlp[-1].out_features == 10

    def test_forward(self, model: TemplateModel) -> None:
        x = torch.randn(8, 1, 28, 28)
        output = model(x)
        assert output.shape == (8, 10)
        assert torch.isfinite(output).all()
