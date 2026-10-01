"""Признаки для модели цены квадратного метра.

Общий код для ноутбука и predict.py. На вход — таблица квартир в формате исходного CSV
(data/flats.csv), на выход — признаки модели.
"""
import numpy as np
import pandas as pd

# Центры городов. Посёлки Новой Москвы и неизвестные города считаем от центра Москвы
CITY_CENTERS = {
    'Казань': (55.7963, 49.1088),
    'Москва': (55.7558, 37.6173),
    'Санкт-Петербург': (59.9386, 30.3141),
    'Новосибирск': (55.0302, 82.9204),
}
ROTATION_ANGLES = (15, 30, 45, 60)

# Столбцы-списки исходных данных: префикс, число позиций и значения, для которых делаем флаги
# (значения, встречающиеся в обучающих данных не реже 20 раз)
LIST_COLUMNS = {
    'window_view': ('all_data.object_info.window_view', 5, ['Двор', 'Улица', 'Парк', 'Лес', 'Водоем']),
    'security': ('all_data.house.security', 4, ['Домофон', 'Закрытая территория', 'Кодовая дверь', 'Консьерж']),
    'parking': ('all_data.house.parking', 5, ['Во дворе', 'Подземная', 'Со шлагбаумом', 'Наземная', 'Охраняемая']),
    'yard': ('all_data.house.yard', 2, ['Детская площадка', 'Спортивная площадка']),
    'infrastructure': ('all_data.house.infrastructure', 5,
                       ['Школа', 'Детский сад', 'Торговый центр', 'Парк', 'Фитнес']),
}

EXTRA_FEATURES = (
    ['dist_center_km', 'bearing_center']
    + [f'rot{angle}_{axis}' for angle in ROTATION_ANGLES for axis in 'xy']
    + ['floor_ratio', 'is_first_floor', 'is_last_floor', 'area_per_room', 'kitchen_area_clean',
       'living_share', 'ceiling_height', 'sale_type']
    + [f'{group}_{value}' for group, (_, _, values) in LIST_COLUMNS.items() for value in values]
)

# Модель для предсказаний: исходные столбцы как есть (без импутации) и дополнительные признаки
MODEL_BASE_FEATURES = [
    'city', 'lat', 'lon', 'area', 'floor', 'bathroom_type', 'balconies', 'is_apartment', 'rooms',
    'house_floors', 'house_wall_type', 'lifts', 'freight_lifts', 'time_on_foot_to_subway', 'build_year',
    'has_gas', 'urgent', 'duplicates_count', 'is_auction', 'all_data.legal_options.approve',
    'all_data.object_info.garage_type.display_name',
]
MODEL_FEATURES = MODEL_BASE_FEATURES + EXTRA_FEATURES
CATEGORICAL_FEATURES = ['city', 'bathroom_type', 'is_apartment', 'house_wall_type', 'has_gas',
                        'all_data.object_info.garage_type.display_name', 'sale_type']
REQUIRED_COLUMNS = ['city', 'lat', 'lon', 'area']
MISSING_CATEGORY = 'нет данных'


def column(raw, name):
    """Столбец исходной таблицы; если его нет, — пустой столбец."""
    if name in raw.columns:
        return raw[name]
    return pd.Series(np.nan, index=raw.index, dtype=object)


def to_float(s):
    """Число из исходного столбца: дробная часть может быть через запятую, логические значения — строками."""
    if s.dtype == bool:
        return s.astype(float)
    if not pd.api.types.is_numeric_dtype(s):
        text = s.astype(str).str.strip().str.lower()
        s = text.map({'true': '1', 'false': '0'}).fillna(text.str.replace(',', '.', regex=False))
    return pd.to_numeric(s, errors='coerce')


def to_category(s):
    """Категория строкой; пропуск — отдельное значение «нет данных»."""
    def value(v):
        if pd.isna(v):
            return MISSING_CATEGORY
        v = str(v).strip()
        return {'true': 'True', 'false': 'False'}.get(v.lower(), v)
    return s.astype(object).map(value)


def make_extra_features(df, raw):
    """Дополнительные признаки (раздел «Дополнительные признаки» в ноутбуке).

    df  — квартиры с числовыми lat, lon, area, kitchen_area, floor, house_floors, rooms и столбцом city;
    raw — те же строки в исходном формате: из него берутся столбцы, которых в df уже нет
          (жилая площадь, высота потолков, тип продажи, столбцы-списки).
    """
    out = pd.DataFrame(index=df.index)

    # Расстояние и направление от центра города
    centers = df['city'].map(lambda c: CITY_CENTERS.get(c, CITY_CENTERS['Москва']))
    lat0 = np.array([c[0] for c in centers])
    lon0 = np.array([c[1] for c in centers])
    lat1, lon1, lat2, lon2 = map(np.radians, [df['lat'].values, df['lon'].values, lat0, lon0])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    out['dist_center_km'] = 6371 * 2 * np.arcsin(np.sqrt(a))
    out['bearing_center'] = np.degrees(np.arctan2(df['lon'] - lon0, df['lat'] - lat0))

    # Повёрнутые координаты: деревья режут пространство только вдоль осей
    dx = (df['lon'] - lon0) * np.cos(np.radians(lat0))
    dy = df['lat'] - lat0
    for angle in ROTATION_ANGLES:
        t = np.radians(angle)
        out[f'rot{angle}_x'] = dx * np.cos(t) - dy * np.sin(t)
        out[f'rot{angle}_y'] = dx * np.sin(t) + dy * np.cos(t)

    # Этаж и площадь; площадь кухни 0 — это пропуск, а не кухня в 0 м²
    out['floor_ratio'] = df['floor'] / df['house_floors']
    out['is_first_floor'] = (df['floor'] == 1).astype(int)
    out['is_last_floor'] = (df['floor'] == df['house_floors']).astype(int)
    out['area_per_room'] = df['area'] / df['rooms'].replace(0, np.nan)
    out['kitchen_area_clean'] = df['kitchen_area'].replace(0, np.nan)

    # Столбцы с большим числом пропусков: бустингу пропуски не мешают
    out['living_share'] = to_float(column(raw, 'all_data.object_info.living_area')) / df['area']
    ceiling = to_float(column(raw, 'all_data.house.ceiling_height'))
    out['ceiling_height'] = ceiling.where(ceiling.between(2, 6))  # значения вне 2–6 м — ошибки ввода
    out['sale_type'] = column(raw, 'sale_type').fillna(MISSING_CATEGORY)

    # Столбцы-списки: по флагу на каждое значение, NaN — если группа в объявлении не заполнена
    for group, (prefix, n, values) in LIST_COLUMNS.items():
        items = pd.DataFrame({i: column(raw, f'{prefix}[{i}].display_name') for i in range(n)})
        specified = items.notna().any(axis=1)
        for value in values:
            out[f'{group}_{value}'] = items.eq(value).any(axis=1).astype(float).where(specified)

    return out[EXTRA_FEATURES]


def build_model_frame(raw):
    """Таблица квартир в формате исходного CSV → признаки модели для предсказаний (MODEL_FEATURES)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f'В таблице нет обязательных столбцов: {missing}')

    df = pd.DataFrame(index=raw.index)
    df['city'] = column(raw, 'city')
    for col in ['lat', 'lon', 'area', 'kitchen_area', 'floor', 'house_floors', 'rooms']:
        df[col] = to_float(column(raw, col))

    X = pd.DataFrame(index=raw.index)
    for col in MODEL_BASE_FEATURES:
        X[col] = to_category(column(raw, col)) if col in CATEGORICAL_FEATURES else to_float(column(raw, col))
    X['build_year'] = X['build_year'].replace(23, 2023)  # та же чистка, что в ноутбуке

    X = pd.concat([X, make_extra_features(df, raw)], axis=1)
    X['sale_type'] = to_category(X['sale_type'])
    return X[MODEL_FEATURES]
