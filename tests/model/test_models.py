import numpy as np
import pandas as pd
import pytest


def test_trainer_fit_diffusion_1d(dummy_trainer_diffusion_1d):
    trainer = dummy_trainer_diffusion_1d
    trainer.fit()
    assert trainer.model is not None


def test_trainer_get_data_generator_diffusion_1d(dummy_trainer_diffusion_1d):
    trainer = dummy_trainer_diffusion_1d
    dg = trainer.get_data_generator()
    assert dg.model_type == "diffusion_ts"
    assert dg.model is not None


def test_trainer_fit_acgan_1d(dummy_trainer_acgan_1d):
    trainer = dummy_trainer_acgan_1d
    trainer.fit()
    assert trainer.model is not None


def test_trainer_get_data_generator_acgan_1d(dummy_trainer_acgan_1d):
    trainer = dummy_trainer_acgan_1d
    dg = trainer.get_data_generator()
    assert dg.model_type == "acgan"
    assert dg.model is not None


def test_trainer_fit_diffusion_2d(dummy_trainer_diffusion_2d):
    trainer = dummy_trainer_diffusion_2d
    trainer.fit()
    assert trainer.model is not None


def test_trainer_get_data_generator_diffusion_2d(dummy_trainer_diffusion_2d):
    trainer = dummy_trainer_diffusion_2d
    dg = trainer.get_data_generator()
    assert dg.model_type == "diffusion_ts"
    assert dg.model is not None


def test_trainer_fit_acgan_2d(dummy_trainer_acgan_2d):
    trainer = dummy_trainer_acgan_2d
    trainer.fit()
    assert trainer.model is not None


def test_trainer_get_data_generator_acgan_2d(dummy_trainer_acgan_2d):
    trainer = dummy_trainer_acgan_2d
    dg = trainer.get_data_generator()
    assert dg.model_type == "acgan"
    assert dg.model is not None
