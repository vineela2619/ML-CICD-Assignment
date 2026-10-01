from model import train_model


def test_model_training():
    model, accuracy = train_model()

    assert model is not None
    assert 0 <= accuracy <= 1


def test_model_accuracy():
    model, accuracy = train_model()

    assert accuracy >= 0.90
