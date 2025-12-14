import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX


def predecir_y_comparar_modelos(df_historico,
                                df_reales=None):  # df_reales ya no se usa, pero se mantiene para evitar error de argumento.
    meses = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
             'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']

    df_historico = df_historico.dropna(how='all').dropna(axis=1, how='all')

    # Normalización de nombres de columnas
    df_historico.columns = [
        col.lower().replace(' ', '_').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace(
            'ú', 'u') for col in df_historico.columns]

    if 'articulo' not in df_historico.columns:
        if 'artículo' in df_historico.columns:
            df_historico = df_historico.rename(columns={'artículo': 'articulo'})
        else:
            raise ValueError("❌ El archivo debe tener columnas 'articulo' y 'año'.")

    df_historico['año'] = pd.to_numeric(df_historico['año'], errors='coerce')

    if df_historico['año'].isna().any():
        raise ValueError("⚠️ Algunos registros no tienen año válido.")

    df_historico['año'] = df_historico['año'].astype(int)
    año_a_predecir = df_historico['año'].max() + 1

    # Identificar el año de validación (último año histórico completo)
    año_de_validacion = df_historico['año'].max()

    resultados_lr = {}
    resultados_rf = {}
    resultados_xgb = {}
    resultados_prophet = {}
    resultados_sarima = {}
    metricas = {}
    advertencias = []

    for articulo in df_historico['articulo'].unique():
        df_art = df_historico[df_historico['articulo'] == articulo].copy()

        # 1. Preparar datos para VALIDACIÓN (año T)
        df_validacion = df_art[df_art['año'] == año_de_validacion]
        if df_validacion.empty:
            advertencias.append(
                f"⚠️ El artículo '{articulo}' no tiene datos para el último año ({año_de_validacion}) para validar.")
            continue

        # Obtener los valores reales del año de validación
        reales_validacion = [df_validacion.iloc[0].get(mes, float('nan')) for mes in meses]
        reales_validacion = [x for x in reales_validacion if pd.notna(x)]

        if len(reales_validacion) != 12:
            advertencias.append(
                f"⚠️ El artículo '{articulo}' no tiene 12 meses de datos en el último año ({año_de_validacion}) para validar.")
            continue

        # 2. Preparar datos para ENTRENAMIENTO (hasta año T-1)
        df_art_train = df_art[df_art['año'] < año_de_validacion].copy()

        if df_art_train['año'].nunique() < 2:
            advertencias.append(
                f"⚠️ El artículo '{articulo}' tiene muy pocos años históricos antes de {año_de_validacion}.")
            continue

        registros_train = []
        for _, fila in df_art_train.iterrows():
            for i, mes in enumerate(meses):
                if mes in fila and pd.notna(fila[mes]):
                    registros_train.append({
                        'mes': i + 1,
                        'año': fila['año'],
                        'ventas': fila[mes]
                    })

        df_train = pd.DataFrame(registros_train)
        df_train = df_train.dropna(subset=['ventas'])

        if df_train.empty:
            advertencias.append(f"⚠️ El artículo '{articulo}' no tiene suficientes datos de entrenamiento.")
            continue

        # Crear series de tiempo para SARIMA
        df_ts = df_train.copy()
        df_ts['ds'] = pd.to_datetime(df_ts['año'].astype(str) + "-" + df_ts['mes'].astype(str) + "-01")
        df_ts = df_ts.set_index('ds')
        ts_data = df_ts['ventas']

        X_train = df_train[['año', 'mes']]
        y_train = df_train['ventas']

        # Puntos a PREDECIR
        X_pred_futuro = pd.DataFrame({'año': [año_a_predecir] * 12, 'mes': list(range(1, 13))})
        X_pred_validacion = pd.DataFrame({'año': [año_de_validacion] * 12, 'mes': list(range(1, 13))})

        # --- MODELADO y PREDICCIÓN ---

        # LINEAR REGRESSION
        modelo_lr = LinearRegression().fit(X_train, y_train)
        y_pred_lr_futuro = modelo_lr.predict(X_pred_futuro)
        y_pred_lr_validacion = modelo_lr.predict(X_pred_validacion)
        resultados_lr[articulo] = dict(zip(meses, [round(x, 2) for x in y_pred_lr_futuro]))

        # RANDOM FOREST
        modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        y_pred_rf_futuro = modelo_rf.predict(X_pred_futuro)
        y_pred_rf_validacion = modelo_rf.predict(X_pred_validacion)
        resultados_rf[articulo] = dict(zip(meses, [round(x, 2) for x in y_pred_rf_futuro]))

        # XGBOOST
        modelo_xgb = XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        y_pred_xgb_futuro = modelo_xgb.predict(X_pred_futuro)
        y_pred_xgb_validacion = modelo_xgb.predict(X_pred_validacion)
        resultados_xgb[articulo] = dict(zip(meses, [round(x, 2) for x in y_pred_xgb_futuro]))

        # PROPHET (SECCIÓN CORREGIDA)
        df_prophet = pd.DataFrame({
            'ds': pd.to_datetime(df_train['año'].astype(str) + "-" + df_train['mes'].astype(str) + "-01"),
            'y': df_train['ventas']
        })

        modelo_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        modelo_prophet.fit(df_prophet)

        # Predicción Futura
        future_futuro = modelo_prophet.make_future_dataframe(periods=12, freq='MS', include_history=False)
        forecast_futuro = modelo_prophet.predict(future_futuro)
        y_pred_prophet_futuro = forecast_futuro['yhat'].values
        resultados_prophet[articulo] = dict(zip(meses, [round(x, 2) for x in y_pred_prophet_futuro]))

        # Predicción Validación
        future_validacion = pd.DataFrame({'ds': pd.to_datetime(
            X_pred_validacion['año'].astype(str) + "-" + X_pred_validacion['mes'].astype(str) + "-01")})
        forecast_validacion = modelo_prophet.predict(future_validacion)
        y_pred_prophet_validacion = forecast_validacion['yhat'].values

        # SARIMA
        y_pred_sarima_futuro = [0] * 12
        y_pred_sarima_validacion = [0] * 12

        try:
            modelo_sarima = SARIMAX(ts_data,
                                    order=(1, 0, 1),
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit(disp=False)

            # Predicción Futura
            start_futuro = len(ts_data)
            end_futuro = len(ts_data) + 12 - 1
            sarima_forecast_futuro = modelo_sarima.get_prediction(start=start_futuro, end=end_futuro)
            y_pred_sarima_futuro = sarima_forecast_futuro.predicted_mean.values
            resultados_sarima[articulo] = dict(zip(meses, [round(x, 2) for x in y_pred_sarima_futuro]))

            # Predicción Validación
            sarima_forecast_validacion = modelo_sarima.get_prediction(start=start_futuro, end=end_futuro)
            y_pred_sarima_validacion = sarima_forecast_validacion.predicted_mean.values

        except Exception as e:
            advertencias.append(f"❌ Error al ajustar SARIMA para '{articulo}': {e}.")

        # 3. CÁLCULO DE MÉTRICAS (Validación Cruzada)

        mae_lr = mean_absolute_error(reales_validacion, y_pred_lr_validacion)
        mae_rf = mean_absolute_error(reales_validacion, y_pred_rf_validacion)
        mae_xgb = mean_absolute_error(reales_validacion, y_pred_xgb_validacion)
        mae_prophet = mean_absolute_error(reales_validacion, y_pred_prophet_validacion)

        try:
            mae_sarima = mean_absolute_error(reales_validacion, y_pred_sarima_validacion)
        except ValueError:
            mae_sarima = float('inf')

        mejor = min([
            (mae_lr, 'Linear Regression'),
            (mae_rf, 'Random Forest'),
            (mae_xgb, 'XGBoost'),
            (mae_prophet, 'Prophet'),
            (mae_sarima, 'SARIMA')
        ], key=lambda x: x[0])

        # Se guarda la métrica de validación interna
        metricas[articulo] = {
            'año_validacion': año_de_validacion,
            'mae_lr': round(mae_lr, 2),
            'mae_rf': round(mae_rf, 2),
            'mae_xgb': round(mae_xgb, 2),
            'mae_prophet': round(mae_prophet, 2),
            'mae_sarima': round(mae_sarima, 2) if mae_sarima != float('inf') else 'Error',
            'modelo_recomendado': mejor[1]
        }

    if advertencias:
        print("\n".join(advertencias))

    return resultados_lr, resultados_rf, resultados_xgb, resultados_prophet, resultados_sarima, metricas