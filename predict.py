"""Прогноз цены квадратного метра для новых квартир.

Модель создаёт ноутбук (раздел «Модель для предсказаний») и сохраняет её в models/price_model.joblib.
На вход — таблица квартир в формате исходного CSV; обязательны столбцы city, lat, lon, area,
остальные можно не заполнять.

    import pandas as pd
    from predict import predict_price

    flats = pd.read_csv('new_flats.csv')
    flats['predicted_price_sq'] = predict_price(flats)

Из командной строки:

    python predict.py new_flats.csv predictions.csv
"""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import build_model_frame

MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'price_model.joblib'

_model = None


def load_model(path=MODEL_PATH):
    """Загружает сохранённую модель (один раз за сеанс)."""
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model


def predict_price(flats, model=None):
    """Прогноз цены квадратного метра, руб. Среднее LightGBM и CatBoost, обученных на логарифме цены."""
    model = model or load_model()
    X = build_model_frame(flats)[model['features']]
    with warnings.catch_warnings():
        # LightGBM 4.6 с новым scikit-learn ложно предупреждает об именах признаков: модель обучалась
        # на той же матрице после OneHotEncoder, что и здесь, — прогноз от этого не меняется
        warnings.filterwarnings('ignore', message='X does not have valid feature names')
        pred_lgbm = np.exp(model['lgbm'].predict(X))
    pred_catboost = np.exp(model['catboost'].predict(X))
    return (pred_lgbm + pred_catboost) / 2


def main():
    parser = argparse.ArgumentParser(description='Прогноз цены квадратного метра для квартир из CSV.')
    parser.add_argument('input', help='CSV с квартирами в формате исходных данных')
    parser.add_argument('output', nargs='?', help='куда сохранить CSV с прогнозом (по умолчанию — вывести на экран)')
    args = parser.parse_args()

    flats = pd.read_csv(args.input)
    flats['predicted_price_sq'] = predict_price(flats).round()
    if args.output:
        flats.to_csv(args.output, index=False)
        print(f'Прогноз для {len(flats)} квартир сохранён в {args.output}')
    else:
        print(flats[['city', 'area', 'predicted_price_sq']].to_string(index=False))


if __name__ == '__main__':
    main()
