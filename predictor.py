import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from prophet import Prophet

def predecir_y_comparar_modelos(df_historico, df_reales=None):
    meses = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
             'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']

    df_historico = df_historico.dropna(how='all').dropna(axis=1, how='all')

    if 'artículo' not in df_historico.columns or 'año' not in df_historico.columns:
        raise ValueError("❌ El archivo debe tener columnas 'artículo' y 'año' separadas.")

    df_historico['articulo'] = df_historico['artículo']
    df_historico['año'] = pd.to_numeric(df_historico['año'], errors='coerce')

    if df_historico['año'].isna().any():
        raise ValueError("⚠️ Algunos registros no tienen año válido.")

    df_historico['año'] = df_historico['año'].astype(int)
    año_a_predecir = df_historico['año'].max() + 1

    resultados_lr = {}
    resultados_rf = {}
    resultados_xgb = {}
    resultados_prophet = {}
    metricas = {}
    advertencias = []

    for articulo in df_historico['articulo'].unique():
        df_art = df_historico[df_historico['articulo'] == articulo].copy()
        df_art = df_art[df_art['año'].notna()]
        df_art['año'] = df_art['año'].astype(int)

        if df_art['año'].nunique() < 2:
            advertencias.append(f"⚠️ El artículo '{articulo}' tiene solo un año de datos. No se puede predecir.")
            continue

        registros = []
        for _, fila in df_art.iterrows():
            for i, mes in enumerate(meses):
                registros.append({
                    'mes': i + 1,
                    'año': fila['año'],
                    'ventas': fila[mes]
                })

        df_train = pd.DataFrame(registros)

        if df_train.isnull().any().any():
            raise ValueError(f"❌ Datos faltantes detectados en el artículo: {articulo}")

        X = df_train[['año', 'mes']]
        y = df_train['ventas']
        X_pred = pd.DataFrame({'año': [año_a_predecir]*12, 'mes': list(range(1, 13))})

        # LINEAR REGRESSION
        modelo_lr = LinearRegression().fit(X, y)
        y_pred_lr = modelo_lr.predict(X_pred)
        resultados_lr[articulo] = dict(zip(meses, [round(x, 2) for x in y_pred_lr]))

        # RANDOM FOREST
        modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        y_pred_rf = modelo_rf.predict(X_pred)
        resultados_rf[articulo] = dict(zip(meses, [round(x, 2) for x in y_pred_rf]))

        # XGBOOST
        modelo_xgb = XGBRegressor(n_estimators=100, random_state=42).fit(X, y)
        y_pred_xgb = modelo_xgb.predict(X_pred)
        resultados_xgb[articulo] = dict(zip(meses, [round(x, 2) for x in y_pred_xgb]))

        # PROPHET
        df_prophet = df_train.copy()
        df_prophet['ds'] = pd.to_datetime(df_prophet['año'].astype(str) + "-" + df_prophet['mes'].astype(str) + "-01")
        df_prophet = df_prophet.rename(columns={'ventas': 'y'})
        modelo_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        modelo_prophet.fit(df_prophet[['ds', 'y']])
        future = modelo_prophet.make_future_dataframe(periods=12, freq='MS')
        forecast = modelo_prophet.predict(future)
        forecast_pred = forecast[-12:]['yhat'].values
        resultados_prophet[articulo] = dict(zip(meses, [round(x, 2) for x in forecast_pred]))

        # MÉTRICAS SI HAY REALES
        if df_reales is not None:
            fila_real = df_reales[(df_reales['artículo'] == articulo) & (df_reales['año'] == año_a_predecir)]
            if not fila_real.empty:
                reales = [fila_real.iloc[0][mes] for mes in meses]
                mae_lr = mean_absolute_error(reales, y_pred_lr)
                mae_rf = mean_absolute_error(reales, y_pred_rf)
                mae_xgb = mean_absolute_error(reales, y_pred_xgb)
                mae_prophet = mean_absolute_error(reales, forecast_pred)

                mejor = min([
                    (mae_lr, 'Linear Regression'),
                    (mae_rf, 'Random Forest'),
                    (mae_xgb, 'XGBoost'),
                    (mae_prophet, 'Prophet')
                ], key=lambda x: x[0])

                metricas[articulo] = {
                    'mae_lr': round(mae_lr, 2),
                    'mae_rf': round(mae_rf, 2),
                    'mae_xgb': round(mae_xgb, 2),
                    'mae_prophet': round(mae_prophet, 2),
                    'modelo_recomendado': mejor[1]
                }

    if advertencias:
        print("\n".join(advertencias))

    return resultados_lr, resultados_rf, resultados_xgb, resultados_prophet, metricas